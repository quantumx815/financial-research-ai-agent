import json
import os
from dotenv import load_dotenv
from google import genai

from services.research_service import build_research_package

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def _parse_analysis_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return json.loads(cleaned)


def generate_financial_analysis(company_data, news, rag_context=None):
    research_package = build_research_package(company_data, news)

    rag_evidence_text = ""
    if rag_context and rag_context.results:
        rag_evidence_text = "\n\nSEC Filing Evidence:\n"
        rag_evidence_text += (
            "The following excerpts were retrieved from official SEC filings "
            "using semantic search.\n"
            "Use this evidence to strengthen and verify your analysis where relevant.\n"
            "Do not claim an SEC filing supports a statement if the evidence does not support it.\n"
            "If SEC evidence is unavailable, rely on the available financial data and recent news.\n"
            "Never fabricate missing evidence.\n\n"
        )
        for idx, result in enumerate(rag_context.results, 1):
            rag_evidence_text += (
                f"[{idx}] {result.document_type} ({result.document_year})\n"
                f"Document ID: {result.document_id}\n"
                f"Chunk Index: {result.chunk_index}\n"
                f"Source: {result.source} - {result.source_url}\n"
                f"Text: {result.chunk_text}\n\n"
            )

    prompt = f"""
You are a financial research assistant.

Analyze the following company information, recent news, and SEC filing evidence.
The news articles have already been ranked by local relevance score.
Higher-relevance articles should be weighted more heavily when discussing important recent news, positive factors, and risks.
Do not completely ignore lower-relevance articles if they contain useful context.

Company Information:
{research_package["company"]}

Financial Metrics:
{research_package["financial_metrics"]}

Recent News:
{research_package["recent_news"]}
{rag_evidence_text}
Financial Health Analysis Requirements:
For each financial health category below, do not simply restate the metric. Instead, provide:
1. The relevant metric or value from the research package.
2. What that metric indicates about the company.
3. Why that information matters to the company's overall financial condition.

Growth:
- Use revenue and revenue growth.
- Explain whether the available growth information suggests strong, moderate, or weak expansion, and why that matters for the company's trajectory.

Profitability:
- Use net income and profit margin.
- Explain what the profitability profile indicates about the company's ability to convert revenue into earnings, and why that matters.

Efficiency:
- Use return on equity (ROE).
- Explain what ROE indicates about capital efficiency and management effectiveness.
- Mention that ROE can be influenced by capital structure, including leverage or share buybacks, when relevant.
- Do not overstate extremely high or low ROE values.

Valuation:
- Use P/E ratio, current price, 52-week high, and 52-week low.
- Discuss valuation only as a research consideration.
- Do not provide a price target.
- Do not label the stock as cheap, expensive, a good buy, or a good sell unless phrased neutrally as a valuation observation supported by the supplied data.

Leverage:
- Use debt-to-equity.
- Explain what the leverage position indicates about financial risk or flexibility based on the available value.
- Do not apply a universal "good" or "bad" threshold unless the context supports it.

Overall Financial Health:
- Provide a balanced qualitative assessment: Strong, Moderate, or Weak.
- Base this assessment on the combination of the available categories above.
- Briefly explain the main reasons supporting the overall assessment.
- Do not create an arbitrary numerical financial-health score.
- If important metrics are unavailable, explicitly mention that the assessment is based only on the available metrics.

Opportunity Analysis Requirements:
- Identify potential opportunities only when supported by the provided financial data, news, or SEC filing evidence.
- For each important opportunity:
  1. State the opportunity.
  2. Explain the evidence supporting it from the research package.
  3. Explain why it could matter to the company.
- Connect financial metrics and news when appropriate. For example, strong revenue growth together with demand-related news may support continued expansion.
- When SEC filing evidence is provided, prioritize it for risk and opportunity analysis when it directly supports a finding.
- Do not present an opportunity as a guaranteed future outcome.
- Use cautious language such as: may, could, suggests, indicates, appears.
- Do not use unsupported certainty such as: definitely, guaranteed, will certainly.

Risk Analysis Requirements:
- Identify potential risks only when supported by the provided financial data, news, or SEC filing evidence.
- For each important risk:
  1. State the risk.
  2. Explain the evidence supporting it from the research package.
  3. Explain why it could matter to the company.
- Connect financial metrics and news when appropriate.
- When SEC filing evidence is provided, prioritize it for risk and opportunity analysis when it directly supports a finding.
- Do not present uncertain information as confirmed fact.
- Use cautious language such as: may, could, suggests, indicates, appears.
- Do not use unsupported certainty such as: definitely, guaranteed, will certainly.

General Instructions:
- Base all analysis strictly on the values provided above. If a metric is missing or None, do not invent it.
- Distinguish between facts directly provided in the research package and reasonable interpretations based on that information. Do not present speculation as confirmed fact.
- Risks and opportunities must be supported by the provided financial data, news, or SEC filing evidence.
- Do NOT provide investment recommendations. Do not use phrases such as buy signal, sell signal, buy recommendation, sell recommendation, hold recommendation, good time to buy, good entry point, investors should buy, investors should sell, or price target.
- Use neutral research terminology such as: positive factor, potential opportunity, risk factor, valuation consideration, research observation, market consideration.
- Clearly state that this output is informational/educational research and is not financial advice or a buy/sell/hold recommendation.

OUTPUT FORMAT:
Return ONLY valid JSON. Do not include Markdown formatting, code fences, or any text before or after the JSON.

Use exactly this JSON structure:

{{
  "company_overview": {{
    "summary": ""
  }},
  "financial_health": {{
    "overall_assessment": "Strong",
    "overall_reasoning": "",
    "growth": {{
      "assessment": "",
      "evidence": "",
      "significance": ""
    }},
    "profitability": {{
      "assessment": "",
      "evidence": "",
      "significance": ""
    }},
    "efficiency": {{
      "assessment": "",
      "evidence": "",
      "significance": ""
    }},
    "valuation": {{
      "assessment": "",
      "evidence": "",
      "significance": ""
    }},
    "leverage": {{
      "assessment": "",
      "evidence": "",
      "significance": ""
    }}
  }},
  "recent_news": [
    {{
      "title": "",
      "summary": "",
      "relevance": "",
      "publisher": "",
      "published_at": "",
      "url": ""
    }}
  ],
  "opportunities": [
    {{
      "point": "",
      "evidence": "",
      "why_it_matters": ""
    }}
  ],
  "risks": [
    {{
      "point": "",
      "evidence": "",
      "why_it_matters": ""
    }}
  ],
  "overall_perspective": {{
    "summary": ""
  }},
  "disclaimer": ""
}}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return _parse_analysis_response(response.text)
