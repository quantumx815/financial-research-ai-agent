"use client";

import { useState, useRef } from "react";

interface CompanyData {
  symbol: string;
  company_name: string;
  sector: string;
  industry: string;
  current_price: number;
  market_cap: number;
  currency: string;
}

interface NewsItem {
  title: string;
  summary: string;
  publisher: string;
  published_at: string;
  url: string;
}

interface FinancialHealth {
  overall_assessment: string;
  overall_reasoning: string;
  growth: {
    assessment: string;
    evidence: string;
    significance: string;
  };
  profitability: {
    assessment: string;
    evidence: string;
    significance: string;
  };
  efficiency: {
    assessment: string;
    evidence: string;
    significance: string;
  };
  valuation: {
    assessment: string;
    evidence: string;
    significance: string;
  };
  leverage: {
    assessment: string;
    evidence: string;
    significance: string;
  };
}

interface RecentNews {
  title: string;
  summary: string;
  relevance: string;
  publisher: string;
  published_at: string;
  url: string;
}

interface Opportunity {
  point: string;
  evidence: string;
  why_it_matters: string;
}

interface Risk {
  point: string;
  evidence: string;
  why_it_matters: string;
}

interface AnalysisResponse {
  company_overview: {
    summary: string;
  };
  financial_health: FinancialHealth;
  recent_news: RecentNews[];
  opportunities: Opportunity[];
  risks: Risk[];
  overall_perspective: {
    summary: string;
  };
  disclaimer: string;
}

interface AnalysisData {
  symbol: string;
  analysis: AnalysisResponse;
}

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm">
      <h3 className="text-base font-semibold text-slate-900 mb-3">
        {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function MetricBlock({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
        {label}
      </p>
      <p className="text-sm text-slate-700 whitespace-pre-wrap">{value}</p>
    </div>
  );
}

export default function Home() {
  const [company, setCompany] = useState("");
  const [data, setData] = useState<CompanyData | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [analysisError, setAnalysisError] = useState("");

  const searchIdRef = useRef(0);
  const analysisControllerRef = useRef<AbortController | null>(null);

  const searchCompany = async () => {
    if (!company.trim()) {
      setError("Please enter a stock symbol");
      return;
    }

    const currentSearchId = ++searchIdRef.current;

    setLoading(true);
    setData(null);
    setNews([]);
    setError("");
    setAnalysisData(null);
    setAnalysisError("");
    setAnalysisLoading(true);

    const symbol = company.trim().toUpperCase();

    try {
      const companyResponse = await fetch(
        `http://127.0.0.1:8000/api/company/${encodeURIComponent(symbol)}`
      );

      if (!companyResponse.ok) {
        throw new Error("Company not found");
      }

      const companyResult = await companyResponse.json();

      if (currentSearchId !== searchIdRef.current) return;

      const newsResponse = await fetch(
        `http://127.0.0.1:8000/api/company/${encodeURIComponent(symbol)}/news`
      );

      if (!newsResponse.ok) {
        throw new Error("News could not be retrieved");
      }

      const newsResult = await newsResponse.json();

      if (currentSearchId !== searchIdRef.current) return;

      setData(companyResult);
      setNews(newsResult.news || []);
      setLoading(false);
    } catch (error) {
      if (currentSearchId !== searchIdRef.current) return;
      setError("Unable to retrieve company information.");
      setLoading(false);
    }

    if (currentSearchId !== searchIdRef.current) return;

    if (analysisControllerRef.current) {
      analysisControllerRef.current.abort();
    }

    const controller = new AbortController();
    analysisControllerRef.current = controller;

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 30000);

    try {
      const analysisResponse = await fetch(
        `http://127.0.0.1:8000/api/company/${encodeURIComponent(symbol)}/analysis`,
        { signal: controller.signal }
      );

      clearTimeout(timeoutId);

      if (currentSearchId !== searchIdRef.current) return;

      if (analysisResponse.ok) {
        const analysisResult = await analysisResponse.json();
        if (currentSearchId !== searchIdRef.current) return;
        setAnalysisData(analysisResult);
      } else {
        if (currentSearchId !== searchIdRef.current) return;
        setAnalysisError(
          "Unable to generate AI financial research. Please try again."
        );
      }
    } catch (analysisError) {
      clearTimeout(timeoutId);

      if (currentSearchId !== searchIdRef.current) return;

      if (
        analysisError instanceof Error &&
        analysisError.name === "AbortError"
      ) {
        setAnalysisError(
          "AI analysis is taking longer than expected. Please try again."
        );
      } else {
        setAnalysisError(
          "Unable to generate AI financial research. Please try again."
        );
      }
    } finally {
      setAnalysisLoading(false);
      analysisControllerRef.current = null;
    }
  };

  const formatMarketCap = (value: number) => {
    if (value >= 1_000_000_000_000) {
      return `$${(value / 1_000_000_000_000).toFixed(2)} Trillion`;
    }

    if (value >= 1_000_000_000) {
      return `$${(value / 1_000_000_000).toFixed(2)} Billion`;
    }

    if (value >= 1_000_000) {
      return `$${(value / 1_000_000).toFixed(2)} Million`;
    }

    return `$${value.toLocaleString()}`;
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-semibold text-sm">
              F
            </div>
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight text-slate-900">
              Financial Research AI Agent
            </h1>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <section className="mb-10 sm:mb-14">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm">
            <div className="mb-4">
              <h2 className="text-base font-semibold text-slate-900">
                Company Search
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Enter a stock symbol to retrieve financial data and latest news.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3">
              <input
                type="text"
                placeholder="Enter stock symbol (e.g. AAPL)"
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    searchCompany();
                  }
                }}
                className="flex-1 rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-colors"
              />
              <button
                onClick={searchCompany}
                disabled={loading}
                className="inline-flex items-center justify-center rounded-xl bg-indigo-600 px-6 py-3 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </div>
          </div>
        </section>

        {loading && (
          <div className="mb-10 sm:mb-14">
            <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm flex flex-col items-center justify-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
              <p className="text-sm font-medium text-slate-600">
                Loading financial data and news...
              </p>
            </div>
          </div>
        )}

        {error && !loading && (
          <div className="mb-10 sm:mb-14">
            <div className="rounded-2xl border border-red-200 bg-red-50 p-6 shadow-sm">
              <p className="text-sm font-medium text-red-800">{error}</p>
            </div>
          </div>
        )}

        {data && !loading && (
          <>
            <section className="mb-10 sm:mb-14">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-slate-900">
                  Company Overview
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Key financial metrics for {data.company_name}
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Symbol
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    {data.symbol}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Company Name
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    {data.company_name}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Sector
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    {data.sector}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Industry
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    {data.industry}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Current Price
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    ${data.current_price.toFixed(2)}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-1">
                    Market Cap
                  </p>
                  <p className="text-xl font-semibold text-slate-900">
                    {formatMarketCap(data.market_cap)}
                  </p>
                </div>
              </div>
            </section>

            <section className="mb-10 sm:mb-14">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-slate-900">
                  Latest News
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Recent headlines and updates for {data.symbol}
                </p>
              </div>

              {news.length === 0 ? (
                <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
                  <p className="text-sm text-slate-500">No news available.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                  {news.map((item, index) => (
                    <article
                      key={index}
                      className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 sm:p-6 shadow-sm hover:shadow-md transition-shadow"
                    >
                      <h3 className="text-base font-semibold text-slate-900 mb-2 line-clamp-2">
                        {item.title}
                      </h3>
                      <p className="text-sm text-slate-600 mb-4 line-clamp-3 flex-1">
                        {item.summary}
                      </p>
                      <div className="mt-auto">
                        <div className="flex items-center justify-between text-xs text-slate-500 mb-3">
                          <span className="font-medium text-slate-700">
                            {item.publisher}
                          </span>
                          <span>
                            {new Date(item.published_at).toLocaleString()}
                          </span>
                        </div>
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors"
                        >
                          Read full article
                          <svg
                            className="ml-1 h-4 w-4"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M13 7l5 5m0 0l-5 5m5-5H6"
                            />
                          </svg>
                        </a>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>

            <section className="mb-10 sm:mb-14">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-slate-900">
                  AI Financial Research
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Gemini-powered analysis for {data.symbol}
                </p>
              </div>

              {analysisLoading && (
                <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm flex flex-col items-center justify-center gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
                  <p className="text-sm font-medium text-slate-600">
                    Generating AI financial research...
                  </p>
                </div>
              )}

              {analysisError && !analysisLoading && (
                <div className="rounded-2xl border border-red-200 bg-red-50 p-6 shadow-sm">
                  <p className="text-sm font-medium text-red-800">
                    {analysisError}
                  </p>
                </div>
              )}

              {analysisData && !analysisLoading && (
                <div className="space-y-6">
                  <SectionCard title="Company Overview">
                    <p className="text-sm text-slate-700 whitespace-pre-wrap">
                      {analysisData.analysis.company_overview.summary}
                    </p>
                  </SectionCard>

                  <SectionCard title="Financial Health">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <MetricBlock
                        label="Overall Assessment"
                        value={analysisData.analysis.financial_health.overall_assessment}
                      />
                      <MetricBlock
                        label="Overall Reasoning"
                        value={analysisData.analysis.financial_health.overall_reasoning}
                      />
                    </div>
                    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      <MetricBlock
                        label="Growth"
                        value={
                          analysisData.analysis.financial_health.growth.assessment +
                          "\n\n" +
                          analysisData.analysis.financial_health.growth.evidence +
                          "\n\n" +
                          analysisData.analysis.financial_health.growth.significance
                        }
                      />
                      <MetricBlock
                        label="Profitability"
                        value={
                          analysisData.analysis.financial_health.profitability.assessment +
                          "\n\n" +
                          analysisData.analysis.financial_health.profitability.evidence +
                          "\n\n" +
                          analysisData.analysis.financial_health.profitability.significance
                        }
                      />
                      <MetricBlock
                        label="Efficiency"
                        value={
                          analysisData.analysis.financial_health.efficiency.assessment +
                          "\n\n" +
                          analysisData.analysis.financial_health.efficiency.evidence +
                          "\n\n" +
                          analysisData.analysis.financial_health.efficiency.significance
                        }
                      />
                      <MetricBlock
                        label="Valuation"
                        value={
                          analysisData.analysis.financial_health.valuation.assessment +
                          "\n\n" +
                          analysisData.analysis.financial_health.valuation.evidence +
                          "\n\n" +
                          analysisData.analysis.financial_health.valuation.significance
                        }
                      />
                      <MetricBlock
                        label="Leverage"
                        value={
                          analysisData.analysis.financial_health.leverage.assessment +
                          "\n\n" +
                          analysisData.analysis.financial_health.leverage.evidence +
                          "\n\n" +
                          analysisData.analysis.financial_health.leverage.significance
                        }
                      />
                    </div>
                  </SectionCard>

                  <SectionCard title="Important Recent News">
                    <div className="space-y-4">
                      {analysisData.analysis.recent_news.map((item, index) => (
                        <div
                          key={index}
                          className="rounded-xl border border-slate-100 bg-slate-50 p-4"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="text-sm font-semibold text-slate-900">
                              {item.title}
                            </h4>
                            <span className="text-xs font-medium text-indigo-700 bg-indigo-50 px-2 py-1 rounded-lg">
                              {item.relevance}
                            </span>
                          </div>
                          {item.summary && (
                            <p className="text-sm text-slate-600 mb-2">
                              {item.summary}
                            </p>
                          )}
                          <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>{item.publisher}</span>
                            <span>
                              {new Date(item.published_at).toLocaleString()}
                            </span>
                          </div>
                          {item.url && (
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center text-xs font-medium text-indigo-600 hover:text-indigo-700 mt-2 transition-colors"
                            >
                              Read full article
                              <svg
                                className="ml-1 h-3 w-3"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                strokeWidth={2}
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                                />
                              </svg>
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  </SectionCard>

                  <SectionCard title="Potential Positive Factors">
                    <div className="space-y-4">
                      {analysisData.analysis.opportunities.map((item, index) => (
                        <div
                          key={index}
                          className="rounded-xl border border-slate-100 bg-slate-50 p-4"
                        >
                          <h4 className="text-sm font-semibold text-slate-900 mb-1">
                            {item.point}
                          </h4>
                          <p className="text-sm text-slate-600 mb-1">
                            <span className="font-medium text-slate-700">
                              Evidence:
                            </span>{" "}
                            {item.evidence}
                          </p>
                          <p className="text-sm text-slate-600">
                            <span className="font-medium text-slate-700">
                              Why it matters:
                            </span>{" "}
                            {item.why_it_matters}
                          </p>
                        </div>
                      ))}
                    </div>
                  </SectionCard>

                  <SectionCard title="Potential Risks">
                    <div className="space-y-4">
                      {analysisData.analysis.risks.map((item, index) => (
                        <div
                          key={index}
                          className="rounded-xl border border-slate-100 bg-slate-50 p-4"
                        >
                          <h4 className="text-sm font-semibold text-slate-900 mb-1">
                            {item.point}
                          </h4>
                          <p className="text-sm text-slate-600 mb-1">
                            <span className="font-medium text-slate-700">
                              Evidence:
                            </span>{" "}
                            {item.evidence}
                          </p>
                          <p className="text-sm text-slate-600">
                            <span className="font-medium text-slate-700">
                              Why it matters:
                            </span>{" "}
                            {item.why_it_matters}
                          </p>
                        </div>
                      ))}
                    </div>
                  </SectionCard>

                  <SectionCard title="Overall Research Perspective">
                    <p className="text-sm text-slate-700 whitespace-pre-wrap">
                      {analysisData.analysis.overall_perspective.summary}
                    </p>
                  </SectionCard>

                  <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <p className="text-xs text-slate-500">
                      {analysisData.analysis.disclaimer}
                    </p>
                  </div>
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
