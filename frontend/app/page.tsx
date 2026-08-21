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

interface AnalysisData {
  symbol: string;
  analysis: string;
}

const LINE_RE = /\n/;

function splitBlocks(text: string): string[] {
  return text.split(/\n{2,}/);
}

function parseInline(segment: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let remaining = segment;
  let key = 0;

  while (remaining.length > 0) {
    const boldItalic = remaining.match(/^\*\*\*(.+?)\*\*\*/);
    if (boldItalic) {
      nodes.push(
        <strong key={key++} className="font-semibold">
          <em>{boldItalic[1]}</em>
        </strong>
      );
      remaining = remaining.slice(boldItalic[0].length);
      continue;
    }

    const bold = remaining.match(/^\*\*(.+?)\*\*/);
    if (bold) {
      nodes.push(
        <strong key={key++} className="font-semibold text-slate-900">
          {bold[1]}
        </strong>
      );
      remaining = remaining.slice(bold[0].length);
      continue;
    }

    const italic = remaining.match(/^\*(.+?)\*/);
    if (italic) {
      nodes.push(
        <em key={key++} className="italic">
          {italic[1]}
        </em>
      );
      remaining = remaining.slice(italic[0].length);
      continue;
    }

    const nextMarker = remaining.search(/\*\*|\*|___/);
    const sliceEnd = nextMarker === -1 ? remaining.length : nextMarker;
    const textSlice = remaining.slice(0, sliceEnd);

    if (textSlice.length > 0) {
      nodes.push(textSlice);
      key++;
    }

    remaining = remaining.slice(sliceEnd);
  }

  return nodes;
}

function parseLines(raw: string): React.ReactNode[] {
  const lines = raw.split(LINE_RE);
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^#{1,6}\s/.test(line)) {
      const match = line.match(/^(#{1,6})\s+(.*)/);
      if (match) {
        const level = match[1].length;
        const content = parseInline(match[2]);

        const sizeClasses =
          level <= 2
            ? "text-xl font-semibold text-slate-900 mt-5 mb-2"
            : level === 3
            ? "text-lg font-semibold text-slate-900 mt-4 mb-2"
            : "text-base font-semibold text-slate-900 mt-3 mb-1";

        if (level <= 2) {
          elements.push(
            <h2 key={i} className={sizeClasses}>
              {content}
            </h2>
          );
        } else if (level === 3) {
          elements.push(
            <h3 key={i} className={sizeClasses}>
              {content}
            </h3>
          );
        } else {
          elements.push(
            <h4 key={i} className={sizeClasses}>
              {content}
            </h4>
          );
        }
        i++;
        continue;
      }
    }

    if (/^\s*[-*]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s/, ""));
        i++;
      }

      elements.push(
        <ul key={`ul-${i}`} className="list-disc list-inside mb-3 space-y-1">
          {items.map((item, idx) => (
            <li key={idx} className="text-sm text-slate-700">
              {parseInline(item)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (/^\s*\d+\.\s/.test(line)) {
      const items: string[] = [];
      while (
        i < lines.length &&
        /^\s*\d+\.\s/.test(lines[i])
      ) {
        items.push(lines[i].replace(/^\s*\d+\.\s/, ""));
        i++;
      }

      elements.push(
        <ol key={`ol-${i}`} className="list-decimal list-inside mb-3 space-y-1">
          {items.map((item, idx) => (
            <li key={idx} className="text-sm text-slate-700">
              {parseInline(item)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    if (/^\s*[-]{3,}\s*$/.test(line)) {
      elements.push(
        <hr
          key={`hr-${i}`}
          className="my-4 border-slate-200"
        />
      );
      i++;
      continue;
    }

    if (line.trim().length === 0) {
      i++;
      continue;
    }

    const content = parseInline(line);
    elements.push(
      <p key={i} className="text-sm text-slate-700 mb-2">
        {content}
      </p>
    );
    i++;
  }

  return elements;
}

function renderMarkdown(text: string): React.ReactNode[] {
  const blocks = splitBlocks(text);
  const elements: React.ReactNode[] = [];
  let key = 0;

  for (const block of blocks) {
    if (block.trim().length === 0) {
      continue;
    }

    const hasMultipleLines = LINE_RE.test(block);

    if (hasMultipleLines) {
      elements.push(...parseLines(block));
    } else {
      elements.push(
        <p key={key++} className="text-sm text-slate-700 mb-2">
          {parseInline(block)}
        </p>
      );
    }
  }

  return elements;
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
                <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 shadow-sm">
                  {renderMarkdown(analysisData.analysis)}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
