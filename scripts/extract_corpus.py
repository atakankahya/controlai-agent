#!/usr/bin/env python3
"""Extract page-level text from inventoried PDF, HTML, code, and transcript files."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sources"
DEFAULT_INVENTORY = PROJECT_ROOT / "data" / "processed" / "inventory.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "extracted_corpus.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "extraction_summary.json"

QUARANTINED_ROLES = {"assignment", "solution", "exam"}


def normalized_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str | None:
    if not text:
        return None
    canonical = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_policy(role: str) -> str:
    if role in QUARANTINED_ROLES:
        return "quarantine_problem_or_solution"
    if role == "code":
        return "tool_example_candidate"
    return "knowledge_candidate"


def read_member(input_dir: Path, record: dict) -> bytes:
    container = input_dir / record["container"]
    if record["member_path"] == record["container"]:
        return container.read_bytes()
    with zipfile.ZipFile(container) as archive:
        member_path = record["member_path"]
        for _ in range(8):
            member = archive.getinfo(member_path)
            data = archive.read(member)
            if not stat.S_ISLNK(member.external_attr >> 16):
                return data
            target = data.decode("utf-8").strip()
            member_path = posixpath.normpath(
                posixpath.join(posixpath.dirname(member_path), target)
            )
            if member_path.startswith("../") or member_path not in archive.namelist():
                raise ValueError(f"Archive symlink escaped or is missing: {target}")
        raise ValueError(f"Too many archive symlink levels: {record['member_path']}")


def base_output(record: dict) -> dict:
    return {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "source_title": record.get("source_title"),
        "source_authors": record.get("source_authors", []),
        "corpus_tier": record.get("corpus_tier"),
        "source_coverage": record.get("source_coverage", []),
        "container": record["container"],
        "member_path": record["member_path"],
        "extension": record["extension"],
        "content_role": record["content_role"],
        "split_policy": split_policy(record["content_role"]),
    }


def finish_record(output: dict, text: str) -> dict:
    text = normalized_text(text)
    output.update(
        {
            "text": text,
            "text_sha256": text_hash(text),
            "characters": len(text),
            "words": len(text.split()),
            "extraction_status": "ok" if text else "empty",
        }
    )
    return output


def extract_pdf(data: bytes, record: dict) -> list[dict]:
    outputs = []
    with pymupdf.open(stream=data, filetype="pdf") as document:
        page_count = document.page_count
        for page_index, page in enumerate(document):
            output = base_output(record)
            output.update(
                {
                    "unit_id": f"{record['document_id']}:page:{page_index + 1}",
                    "page_number": page_index + 1,
                    "page_count": page_count,
                }
            )
            text = page.get_text("text", sort=True)
            finish_record(output, text)
            output["needs_ocr_review"] = output["characters"] < 40
            outputs.append(output)
    return outputs


def extract_html(data: bytes, record: dict) -> list[dict]:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "template", "svg", "noscript"]):
        tag.decompose()
    content = soup.find("main") or soup.find("article") or soup.body or soup
    output = base_output(record)
    output.update({"unit_id": record["document_id"], "page_number": None, "page_count": None})
    return [finish_record(output, content.get_text("\n", strip=True))]


def extract_plain(data: bytes, record: dict) -> list[dict]:
    text = data.decode("utf-8", errors="replace")
    output = base_output(record)
    output.update({"unit_id": record["document_id"], "page_number": None, "page_count": None})
    return [finish_record(output, text)]


def extract_notebook(data: bytes, record: dict) -> list[dict]:
    """Keep notebook explanations and source code, but discard outputs and metadata.

    Notebook outputs often contain base64-encoded plots that can be megabytes long.
    They are not useful language-model training text and can dominate token counts.
    """
    raw_text = data.decode("utf-8", errors="replace")
    try:
        notebook = json.loads(raw_text)
    except json.JSONDecodeError:
        if raw_text.startswith("version https://git-lfs.github.com/spec/v1"):
            output = base_output(record)
            output.update(
                {
                    "unit_id": record["document_id"],
                    "page_number": None,
                    "page_count": None,
                    "artifact_status": "git_lfs_pointer",
                }
            )
            return [finish_record(output, "")]
        raise
    sections = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code"}:
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str) or not source.strip():
            continue
        label = "Markdown" if cell_type == "markdown" else "Code"
        sections.append(f"## {label} cell {index}\n{source.strip()}")

    output = base_output(record)
    output.update({"unit_id": record["document_id"], "page_number": None, "page_count": None})
    return [finish_record(output, "\n\n".join(sections))]


def extract_record(input_dir: Path, record: dict) -> list[dict]:
    data = read_member(input_dir, record)
    if record["extension"] == ".pdf":
        return extract_pdf(data, record)
    if record["extension"] in {".html", ".htm"}:
        return extract_html(data, record)
    if record["extension"] == ".ipynb":
        return extract_notebook(data, record)
    return extract_plain(data, record)


def load_inventory(path: Path, max_documents: int | None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["exact_duplicate_of"]:
                continue
            records.append(record)
            if max_documents is not None and len(records) >= max_documents:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse documents already present in the output JSONL and extract only new inventory records",
    )
    args = parser.parse_args()

    inventory = load_inventory(args.inventory, args.max_documents)
    extracted = []
    reused_document_ids = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                extracted.append(row)
                reused_document_ids.add(row["document_id"])
        inventory = [
            record for record in inventory if record["document_id"] not in reused_document_ids
        ]
        print(
            f"Reusing {len(reused_document_ids)} documents; "
            f"extracting {len(inventory)} new documents"
        )
    failures = []
    for index, record in enumerate(inventory, start=1):
        try:
            extracted.extend(extract_record(args.input_dir, record))
        except Exception as error:
            failures.append(
                {
                    "document_id": record["document_id"],
                    "source_id": record["source_id"],
                    "member_path": record["member_path"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if index % 25 == 0 or index == len(inventory):
            print(f"Processed {index}/{len(inventory)} documents")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in extracted:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    status_counts = Counter(record["extraction_status"] for record in extracted)
    summary = {
        "documents_reused": len(reused_document_ids),
        "new_documents_attempted": len(inventory),
        "documents_total": len({record["document_id"] for record in extracted}),
        "documents_failed": len(failures),
        "extracted_units": len(extracted),
        "characters": sum(record["characters"] for record in extracted),
        "words": sum(record["words"] for record in extracted),
        "empty_units": status_counts["empty"],
        "pages_needing_ocr_review": sum(record.get("needs_ocr_review", False) for record in extracted),
        "by_split_policy": dict(sorted(Counter(r["split_policy"] for r in extracted).items())),
        "failures": failures,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Corpus:  {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
