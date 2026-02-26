"""CLI entry point for the FRTR benchmark runner.

Usage:
    # Index all workbooks and run full benchmark
    uv run python -m frtr --max-questions 5

    # Use vLLM deployed model
    uv run python -m frtr --base-url http://gpu-server:8000/v1 --llm-model Qwen/Qwen2.5-VL-72B-Instruct

    # Index only (skip evaluation)
    uv run python -m frtr --index-only

    # Run benchmark on pre-built index
    uv run python -m frtr --skip-index --base-url http://gpu-server:8000/v1 --llm-model my-model

    # Use specific embedding backend
    uv run python -m frtr --embedding clip
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from .config import EmbeddingProvider, FRTRConfig

console = Console()


def _create_embedder(config: FRTRConfig):
    """Instantiate the configured embedding backend."""
    if config.embedding_provider == EmbeddingProvider.CLIP:
        from .embeddings.clip_embedder import CLIPEmbedder
        return CLIPEmbedder(model_name=config.clip_model_name)
    elif config.embedding_provider == EmbeddingProvider.OPENAI:
        from .embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(
            api_key=config.get_openai_api_key(),
            model=config.openai_embedding_model,
        )
    elif config.embedding_provider == EmbeddingProvider.BEDROCK:
        from .embeddings.bedrock_embedder import BedrockEmbedder
        return BedrockEmbedder(
            model_id=config.bedrock_model_id,
            region=config.bedrock_region,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")


@click.command()
@click.option("--data-dir", type=click.Path(exists=True), default=None,
              help="Directory containing .xlsx workbook files")
@click.option("--index-dir", type=click.Path(), default=None,
              help="Directory for persisted index")
@click.option("--embedding", type=click.Choice(["clip", "openai", "bedrock"]),
              default="clip", help="Embedding backend")
@click.option("--llm-model", default="gpt-4o", help="Model name (for vLLM use deployed model path)")
@click.option("--base-url", default=None, help="OpenAI-compatible API base URL (e.g. http://host:8000/v1)")
@click.option("--index-only", is_flag=True, help="Only build the index, skip evaluation")
@click.option("--skip-index", is_flag=True, help="Skip indexing, use existing index")
@click.option("--max-questions", type=int, default=None,
              help="Limit number of questions (for testing)")
@click.option("--output", type=click.Path(), default=None,
              help="Path to save JSON results")
@click.option("--workbook", type=str, default=None,
              help="Evaluate only this workbook (filename stem)")
def main(
    data_dir,
    index_dir,
    embedding,
    llm_model,
    base_url,
    index_only,
    skip_index,
    max_questions,
    output,
    workbook,
):
    """FRTR: From Rows to Reasoning – Benchmark Runner.

    Implements the full FRTR pipeline: indexing, hybrid retrieval with
    RRF fusion, and LLM-based reasoning over multimodal spreadsheet data.
    """
    # Build config
    config_kwargs = {}
    if data_dir:
        config_kwargs["data_dir"] = Path(data_dir)
    if index_dir:
        config_kwargs["index_dir"] = Path(index_dir)
    config_kwargs["embedding_provider"] = EmbeddingProvider(embedding)
    config_kwargs["llm_model"] = llm_model
    if base_url:
        config_kwargs["llm_base_url"] = base_url

    config = FRTRConfig(**config_kwargs)

    console.print("[bold]FRTR: From Rows to Reasoning[/bold]")
    console.print(f"  Data dir:    {config.data_dir}")
    console.print(f"  Index dir:   {config.index_dir}")
    console.print(f"  Embedding:   {config.embedding_provider.value}")
    console.print(f"  LLM model:   {config.llm_model}")
    if config.llm_base_url:
        console.print(f"  LLM API:     {config.llm_base_url}")
    console.print()

    # Create embedder
    console.print("[bold cyan]Initializing embedding model...[/bold cyan]")
    embedder = _create_embedder(config)
    console.print(f"  Embedding dimension: {embedder.dimension}")

    # Create vector store
    from .vectordb import HybridVectorStore

    store = HybridVectorStore(persist_dir=config.index_dir)

    if not skip_index:
        # Stage 1: Index all workbooks
        console.print("\n[bold cyan]Stage 1: Indexing workbooks...[/bold cyan]")
        from .indexer import index_all_workbooks

        all_chunks = index_all_workbooks(config, embedder)
        console.print(f"  Total chunks: {len(all_chunks)}")

        # Add to vector store
        console.print("[bold cyan]Building hybrid index...[/bold cyan]")
        store.add_chunks(all_chunks)
        store.save_bm25()
        console.print(f"  ChromaDB entries: {store.size}")
        console.print(f"  Index persisted to: {config.index_dir}")
    else:
        console.print("[bold cyan]Loading existing index...[/bold cyan]")
        loaded = store.load_bm25()
        if not loaded:
            console.print("[red]Error: No existing BM25 index found. Run without --skip-index first.[/red]")
            sys.exit(1)
        console.print(f"  ChromaDB entries: {store.size}")

    if index_only:
        console.print("\n[green]Indexing complete. Exiting (--index-only).[/green]")
        return

    # Stage 2 + 3: Retrieval and Reasoning
    console.print("\n[bold cyan]Stage 2+3: Running evaluation...[/bold cyan]")

    from .evaluator import (
        Evaluator,
        load_questions,
        print_report,
        save_report,
    )
    from .reasoner import LLMReasoner
    from .retriever import HybridRetriever

    retriever = HybridRetriever(store, embedder, config)
    reasoner = LLMReasoner(config)
    evaluator = Evaluator(retriever, reasoner, config)

    # Load questions
    questions = load_questions(config)
    if workbook:
        questions = [q for q in questions if q.workbook_name == workbook]
    console.print(f"  Total questions: {len(questions)}")

    # Run benchmark
    report = evaluator.run_benchmark(
        questions=questions,
        max_questions=max_questions,
    )

    # Print report
    print_report(report)

    # Save results
    if output:
        save_report(report, Path(output))
    else:
        default_output = config.index_dir / "results.json"
        save_report(report, default_output)


if __name__ == "__main__":
    main()
