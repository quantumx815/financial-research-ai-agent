import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rag_service import rag_service, RetrievalResponse


def print_result(label, response: RetrievalResponse):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"Query: {response.query}")
    print(f"Company: {response.company_symbol}")
    print(f"Top-K: {response.top_k}")
    print(f"Total Found: {response.total_found}")
    print(f"Error: {response.error}")
    print(f"Results:")
    for i, result in enumerate(response.results, 1):
        print(f"  [{i}] chunk_id={result.chunk_id}")
        print(f"      text={result.chunk_text[:120]}...")
        print(f"      company={result.company_symbol} type={result.document_type} year={result.document_year}")
        print(f"      source={result.source} url={result.source_url}")
        print(f"      document_id={result.document_id} chunk_index={result.chunk_index}")
        print(f"      distance={result.distance}")


def main():
    print("Stage 7.5 — RAG Retrieval Integration Tests")
    print("="*60)

    initial_count = rag_service.get_collection().count()
    print(f"\nInitial ChromaDB document count: {initial_count}")

    test_cases = [
        ("AAPL financial risks", "AAPL", 5),
        ("TSLA financial risks", "TSLA", 5),
        ("NVDA revenue and business risks", "NVDA", 5),
        ("AAPL risks (Top-K=3)", "AAPL", 3),
    ]

    responses = []
    for query_text, symbol, top_k in test_cases:
        response = rag_service.query(query_text, symbol, n_results=top_k)
        responses.append((query_text, symbol, response))
        print_result(f"{query_text} | {symbol} | top_k={top_k}", response)

    final_count = rag_service.get_collection().count()
    print(f"\nFinal ChromaDB document count: {final_count}")

    if initial_count != final_count:
        print(f"\nWARNING: Document count changed from {initial_count} to {final_count}!")
    else:
        print(f"\nOK: Document count unchanged at {initial_count}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for query_text, symbol, response in responses:
        status = "OK" if response.error is None else f"ERROR: {response.error}"
        print(f"  {symbol}: {response.total_found} results | {status}")

    cross_company_ok = True
    for query_text, symbol, response in responses:
        for result in response.results:
            if result.company_symbol != symbol.upper():
                print(f"  CROSS-COMPANY LEAK: {result.company_symbol} found in {symbol} query")
                cross_company_ok = False
    if cross_company_ok:
        print("  No cross-company leakage detected.")

    if all(r.error is None for _, _, r in responses):
        print("\nStage 7.5 retrieval layer is ready for the next stage.")
    else:
        print("\nStage 7.5 has errors that need to be resolved.")


if __name__ == "__main__":
    main()
