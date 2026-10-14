#!/usr/bin/env python3
"""python -m knowledge.cli <add|add-dir|list|remove|rebuild|check|query|
source-discover|source-validate|source-sync|source-ingest|source-reindex|
source-status> ...

Section 4's required user workflow, so adding a reference document never
requires hand-editing a Python file. Every subcommand loads the persistent
index, does its one job, and (for mutating commands) saves it back --
exactly the same load/mutate/save shape as program_model.py's own
train_program_model.py script, not a new persistence convention.

The source-* commands (knowledge/sources.py) are the automated-acquisition
workflow -- discover candidate documentation sources, validate their
license/reachability, sync (download) approved ones, and ingest them
through this SAME CLI's existing `add`/`add-dir` machinery underneath.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.config import KnowledgeConfig
from knowledge.index import KnowledgeIndex
from knowledge.ingestion import add_directory, add_document
from knowledge.ingestion import remove_document as _remove_document
from knowledge.document_parser import DocumentParseError
from knowledge.retrieval import retrieve
from knowledge import sources as _sources


def _cmd_add(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    try:
        record = add_document(index, Path(args.path), source_or_publisher=args.source, license_notes=args.license, config=config)
    except (DocumentParseError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1
    index.save()
    print(f"Added {record.filename!r} ({record.document_id[:12]}...): {record.chunk_count} chunks, "
          f"{record.page_count or 'n/a'} pages, via {record.extraction_method}")
    return 0


def _cmd_add_dir(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    report = add_directory(
        index, Path(args.path), source_or_publisher=args.source, license_notes=args.license,
        config=config, recursive=not args.no_recursive,
    )
    index.save()
    print(f"Added: {len(report.added)}  Skipped (duplicates): {len(report.skipped_duplicates)}  Failed: {len(report.failed)}")
    for record in report.added:
        print(f"  [ok] {record.filename} -- {record.chunk_count} chunks")
    for path in report.skipped_duplicates:
        print(f"  [dup] {path}")
    for issue in report.failed:
        print(f"  [FAIL] {issue.source_path}: {issue.reason}")
    return 0 if not report.failed else 1


def _cmd_list(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    docs = index.list_documents()
    if not docs:
        print("No documents indexed.")
        return 0
    for d in docs:
        print(f"{d.document_id[:12]}...  {d.filename:40s}  {d.chunk_count:4d} chunks  "
              f"{d.extraction_method:10s}  {d.title or ''}")
    return 0


def _cmd_remove(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    matches = [d for d in index.documents if d.startswith(args.document_id)]
    if not matches:
        print(f"No document matching id prefix {args.document_id!r}")
        return 1
    if len(matches) > 1:
        print(f"Ambiguous id prefix {args.document_id!r} matches {len(matches)} documents -- use a longer prefix")
        return 1
    removed = _remove_document(index, matches[0], config=config)
    index.save()
    print(f"Removed {matches[0][:12]}..." if removed else "Not found")
    return 0 if removed else 1


def _cmd_rebuild(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    index.rebuild()
    index.save()
    print(f"Rebuilt index: {len(index.chunks)} chunks, vocabulary size {index.embedding_backend.vocabulary_size()}")
    return 0


def _cmd_check(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    issues = index.integrity_check()
    if not issues:
        print(f"OK -- {len(index.documents)} documents, {len(index.chunks)} chunks")
        return 0
    print(f"{len(issues)} issue(s) found:")
    for issue in issues:
        print(f"  - {issue}")
    return 1


def _cmd_query(args, config: KnowledgeConfig) -> int:
    index = KnowledgeIndex.load(config=config)
    results = retrieve(index, args.text, top_k=args.top_k)
    if not results:
        print("No results.")
        return 0
    for r in results:
        c = r.chunk
        loc = f"p.{c.page_start}" if c.page_start else (c.section or "")
        print(f"[{r.score:.3f}] {c.title or c.source_path} {loc} ({c.concept_type}, domains={list(c.domains)})")
        if c.source_url:
            print(f"    source: {c.source_url}" + (f" @ {c.source_revision[:12]}" if c.source_revision else ""))
        print(f"    {c.text[:200]}{'...' if len(c.text) > 200 else ''}")
    return 0


def _cmd_source_discover(args, config: KnowledgeConfig) -> int:
    manifest = _sources.SourceManifest.load(config=config)
    for e in manifest.list():
        print(f"{e.source_id:18s} license={e.license:40s} status={e.license_status:13s} "
              f"sync={e.sync_status:12s} ingest={e.ingest_status}")
        print(f"    {e.canonical_url}")
    return 0


def _cmd_source_validate(args, config: KnowledgeConfig) -> int:
    manifest = _sources.SourceManifest.load(config=config)
    all_ok = True
    for e in manifest.list():
        ok, message = _sources.validate_source(e)
        all_ok = all_ok and ok
        print(f"{'[ok]' if ok else '[FAIL]'} {e.source_id}: {message}")
    return 0 if all_ok else 1


def _cmd_source_sync(args, config: KnowledgeConfig) -> int:
    manifest = _sources.SourceManifest.load(config=config)
    results = (
        [_sources.sync_source(args.source_id, manifest)] if args.source_id else _sources.sync_all(manifest)
    )
    ok = True
    for r in results:
        ok = ok and r.ok
        if r.ok:
            print(f"[ok] {r.source_id}: {r.files_fetched} fetched, {r.files_unchanged} unchanged, revision={r.revision}")
        else:
            print(f"[FAIL] {r.source_id}: {r.error}")
    return 0 if ok else 1


def _cmd_source_ingest(args, config: KnowledgeConfig) -> int:
    manifest = _sources.SourceManifest.load(config=config)
    index = KnowledgeIndex.load(config=config)
    results = (
        [_sources.ingest_source(args.source_id, manifest, index)] if args.source_id
        else _sources.ingest_all(manifest, index)
    )
    index.save()
    ok = True
    for r in results:
        rep = r.report
        ok = ok and not rep.failed
        print(f"{r.source_id}: added={len(rep.added)} duplicates={len(rep.skipped_duplicates)} "
              f"failed={len(rep.failed)} pruned_stale={len(r.pruned_stale_documents)}")
        for issue in rep.failed:
            print(f"  [FAIL] {issue.source_path}: {issue.reason}")
    return 0 if ok else 1


def _cmd_source_reindex(args, config: KnowledgeConfig) -> int:
    """Re-syncs + re-ingests (prune-aware) one or all sources in one step --
    the mission's "reindex" workflow, built from sync_source + ingest_source
    rather than a third code path."""
    manifest = _sources.SourceManifest.load(config=config)
    index = KnowledgeIndex.load(config=config)
    source_ids = [args.source_id] if args.source_id else [e.source_id for e in manifest.list()]
    ok = True
    for source_id in source_ids:
        sync_result = _sources.sync_source(source_id, manifest)
        if not sync_result.ok:
            print(f"[FAIL] {source_id} sync: {sync_result.error}")
            ok = False
            continue
        ingest_result = _sources.ingest_source(source_id, manifest, index)
        rep = ingest_result.report
        ok = ok and not rep.failed
        print(f"{source_id}: synced (revision={sync_result.revision}), "
              f"added={len(rep.added)} duplicates={len(rep.skipped_duplicates)} "
              f"failed={len(rep.failed)} pruned_stale={len(ingest_result.pruned_stale_documents)}")
    index.save()
    return 0 if ok else 1


def _cmd_source_status(args, config: KnowledgeConfig) -> int:
    manifest = _sources.SourceManifest.load(config=config)
    index = KnowledgeIndex.load(config=config)
    usage = _sources.storage_usage(config)
    print(f"Storage: raw_sources={usage['raw_sources_bytes']/1e6:.1f}MB  "
          f"documents={usage['documents_bytes']/1e6:.1f}MB  index={usage['index_bytes']/1e6:.1f}MB")
    for e in manifest.list():
        doc_count = sum(1 for d in index.documents.values() if d.source_id == e.source_id)
        chunk_count = sum(1 for c in index.chunks if c.source_id == e.source_id)
        print(f"\n{e.source_id} -- {e.title}")
        print(f"  license: {e.license} ({e.license_status})")
        print(f"  sync: {e.sync_status}  last_synced_at={e.last_synced_at}  revision={e.revision}  file_count={e.file_count}")
        print(f"  ingest: {e.ingest_status}  last_ingested_at={e.last_ingested_at}  "
              f"documents_indexed={doc_count}  chunks_indexed={chunk_count}")
        if e.error:
            print(f"  error: {e.error}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="ingest one document")
    p_add.add_argument("path")
    p_add.add_argument("--source", default=None, help="source/publisher note")
    p_add.add_argument("--license", default=None, help="license/usage note")

    p_add_dir = sub.add_parser("add-dir", help="ingest every supported file in a directory")
    p_add_dir.add_argument("path")
    p_add_dir.add_argument("--source", default=None)
    p_add_dir.add_argument("--license", default=None)
    p_add_dir.add_argument("--no-recursive", action="store_true")

    sub.add_parser("list", help="list indexed documents")

    p_remove = sub.add_parser("remove", help="remove a document by id (or id prefix)")
    p_remove.add_argument("document_id")

    sub.add_parser("rebuild", help="rebuild the index's embedding vectors")
    sub.add_parser("check", help="run integrity checks")

    p_query = sub.add_parser("query", help="query the knowledge base")
    p_query.add_argument("text")
    p_query.add_argument("--top-k", type=int, default=5)

    sub.add_parser("source-discover", help="list candidate documentation sources and their status")
    sub.add_parser("source-validate", help="check source reachability + recorded license status (no download)")

    p_source_sync = sub.add_parser("source-sync", help="download/update approved sources (all, or one by id)")
    p_source_sync.add_argument("source_id", nargs="?", default=None)

    p_source_ingest = sub.add_parser("source-ingest", help="ingest synced source files into the index")
    p_source_ingest.add_argument("source_id", nargs="?", default=None)

    p_source_reindex = sub.add_parser("source-reindex", help="sync + ingest (prune-aware) in one step")
    p_source_reindex.add_argument("source_id", nargs="?", default=None)

    sub.add_parser("source-status", help="report per-source sync/ingest status, counts, and storage usage")

    args = ap.parse_args()
    config = KnowledgeConfig()
    handlers = {
        "add": _cmd_add, "add-dir": _cmd_add_dir, "list": _cmd_list, "remove": _cmd_remove,
        "rebuild": _cmd_rebuild, "check": _cmd_check, "query": _cmd_query,
        "source-discover": _cmd_source_discover, "source-validate": _cmd_source_validate,
        "source-sync": _cmd_source_sync, "source-ingest": _cmd_source_ingest,
        "source-reindex": _cmd_source_reindex, "source-status": _cmd_source_status,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
