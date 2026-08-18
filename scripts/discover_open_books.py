#!/usr/bin/env python3
"""Harvest and rank English open-access control/foundations books from OAI-PMH catalogs."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider, DropItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sources" / "discovered_open_books.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "sources" / "discovered_open_books_summary.json"

CATALOGS = {
    "doab": "https://directory.doabooks.org/oai/request",
    "oapen": "https://library.oapen.org/oai/request",
}

# Specific control phrases are weighted more strongly than broad mathematical foundations.
PHRASE_WEIGHTS = {
    "automatic control": 12,
    "control engineering": 12,
    "control system": 12,
    "feedback control": 12,
    "robust control": 12,
    "optimal control": 12,
    "model predictive control": 14,
    "predictive control": 11,
    "nonlinear control": 12,
    "adaptive control": 12,
    "digital control": 10,
    "process control": 10,
    "distributed control": 10,
    "networked control": 10,
    "sliding mode": 9,
    "system identification": 11,
    "state estimation": 10,
    "kalman filter": 10,
    "observer design": 10,
    "dynamical system": 8,
    "dynamic system": 7,
    "system theory": 8,
    "signals and systems": 10,
    "signal processing": 7,
    "mechatronics": 7,
    "robotics": 6,
    "autonomous system": 6,
    "trajectory optimization": 9,
    "convex optimization": 8,
    "numerical optimization": 7,
    "optimization algorithm": 5,
    "linear algebra": 6,
    "differential equation": 6,
    "numerical method": 5,
    "stochastic process": 6,
    "time series": 4,
    "mathematical modeling": 5,
    "mathematical modelling": 5,
}

NEGATIVE_PHRASES = {
    "access control",
    "birth control",
    "disease control",
    "infection control",
    "pest control",
    "social control",
    "tobacco control",
    "crime control",
    "border control",
    "arms control",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def score_record(title: str, subjects: list[str], description: str) -> tuple[int, list[str]]:
    fields = [
        (normalize(title), 3),
        (normalize(" ".join(subjects)), 2),
        (normalize(description), 1),
    ]
    score = 0
    matches = []
    for phrase, weight in PHRASE_WEIGHTS.items():
        field_multiplier = max((multiplier for text, multiplier in fields if phrase in text), default=0)
        if field_multiplier:
            score += weight * field_multiplier
            matches.append(phrase)

    combined = " ".join(text for text, _ in fields)
    negative_matches = [phrase for phrase in NEGATIVE_PHRASES if phrase in combined]
    score -= 18 * len(negative_matches)
    return max(score, 0), sorted(matches)


def texts(node: scrapy.Selector, local_name: str) -> list[str]:
    return unique(node.xpath(f".//*[local-name()='{local_name}']/text()").getall())


class JsonlCatalogPipeline:
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        pipeline.crawler = crawler
        return pipeline

    def open_spider(self) -> None:
        spider = self.crawler.spider
        self.output_path = Path(spider.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.known_urls = set()
        if self.output_path.exists():
            with self.output_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        self.known_urls.add(json.loads(line)["pdf_url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        self.stream = self.output_path.open("a", encoding="utf-8")
        self.added = 0
        self.duplicates = 0

    def process_item(self, item: dict) -> dict:
        if item["pdf_url"] in self.known_urls:
            self.duplicates += 1
            raise DropItem("duplicate PDF URL")
        self.known_urls.add(item["pdf_url"])
        self.stream.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        self.added += 1
        return item

    def close_spider(self) -> None:
        spider = self.crawler.spider
        self.stream.close()
        summary = {
            "catalog": spider.catalog,
            "endpoint": spider.endpoint,
            "pages_harvested_this_run": spider.pages_seen,
            "records_seen_this_run": spider.records_seen,
            "candidates_seen_this_run": spider.candidates_seen,
            "candidates_added_this_run": self.added,
            "duplicate_pdf_urls_this_run": self.duplicates,
            "total_unique_candidates": len(self.known_urls),
            "next_resumption_token": spider.next_resumption_token,
            "minimum_score": spider.min_score,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        Path(spider.summary_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


class OpenBooksOaiSpider(scrapy.Spider):
    name = "open_books_oai"

    custom_settings = {
        "ITEM_PIPELINES": {JsonlCatalogPipeline: 300},
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 1.0,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 8.0,
        "RETRY_TIMES": 4,
        "DOWNLOAD_TIMEOUT": 60,
        "USER_AGENT": "controlai-open-corpus-research/0.1",
        "LOG_LEVEL": "INFO",
        "DEFAULT_DROPITEM_LOG_LEVEL": "DEBUG",
    }

    def __init__(
        self,
        catalog: str,
        output_path: str,
        summary_path: str,
        max_pages: int,
        max_candidates: int,
        min_score: int,
        from_date: str | None = None,
        resume_token: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.catalog = catalog
        self.endpoint = CATALOGS[catalog]
        self.output_path = output_path
        self.summary_path = summary_path
        self.max_pages = int(max_pages)
        self.max_candidates = int(max_candidates)
        self.min_score = int(min_score)
        self.from_date = from_date
        self.resume_token = resume_token
        self.pages_seen = 0
        self.records_seen = 0
        self.candidates_seen = 0
        self.next_resumption_token = resume_token

    async def start(self):
        if self.resume_token:
            params = {"verb": "ListRecords", "resumptionToken": self.resume_token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            if self.from_date:
                params["from"] = self.from_date
        yield scrapy.Request(f"{self.endpoint}?{urlencode(params)}", callback=self.parse_records)

    def parse_records(self, response: scrapy.http.Response):
        self.pages_seen += 1
        records = response.xpath("//*[local-name()='record']")
        for record in records:
            self.records_seen += 1
            titles = texts(record, "title")
            title = titles[0] if titles else "Untitled"
            subjects = texts(record, "subject")
            descriptions = texts(record, "description")
            description = "\n".join(descriptions)
            languages = [normalize(value) for value in texts(record, "language")]
            resource_types = [normalize(value) for value in texts(record, "resourceType")]
            identifiers = texts(record, "identifier")
            pdf_urls = unique(
                value for value in identifiers if value.startswith("http") and ".pdf" in value.lower()
            )
            landing_urls = unique(
                value for value in identifiers if value.startswith("http") and value not in pdf_urls
            )

            if languages and not any(value in {"en", "eng", "english"} for value in languages):
                continue
            if resource_types and not any(value in {"book", "monograph", "textbook"} for value in resource_types):
                continue
            if not pdf_urls:
                continue

            relevance_score, matched_phrases = score_record(title, subjects, description)
            if relevance_score < self.min_score:
                continue

            self.candidates_seen += 1
            creators = texts(record, "creator")
            contributors = texts(record, "contributor")
            publishers = texts(record, "publisher")
            licenses = unique(
                record.xpath(".//*[local-name()='licenseCondition']/@uri").getall()
                + texts(record, "rights")
            )
            issued = record.xpath(
                ".//*[local-name()='date' and @type='Issued']/text()"
            ).get()
            oai_identifier = record.xpath("./*[local-name()='header']/*[local-name()='identifier']/text()").get()

            for pdf_url in pdf_urls:
                yield {
                    "source_catalog": self.catalog,
                    "oai_identifier": oai_identifier,
                    "title": title,
                    "creators": creators,
                    "contributors": contributors,
                    "subjects": subjects,
                    "description": description,
                    "language": languages,
                    "resource_types": resource_types,
                    "publishers": publishers,
                    "issued": issued,
                    "licenses": licenses,
                    "pdf_url": pdf_url,
                    "landing_urls": landing_urls,
                    "relevance_score": relevance_score,
                    "matched_phrases": matched_phrases,
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                }

            if self.max_candidates and self.candidates_seen >= self.max_candidates:
                raise CloseSpider("candidate limit reached")

        token = response.xpath("string(//*[local-name()='resumptionToken'])").get(default="").strip()
        self.next_resumption_token = token or None
        if self.max_pages and self.pages_seen >= self.max_pages:
            raise CloseSpider("page limit reached")
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
            yield response.follow(f"{self.endpoint}?{urlencode(params)}", callback=self.parse_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", choices=sorted(CATALOGS), default="doab")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-pages", type=int, default=10, help="OAI pages; normally 100 records each")
    parser.add_argument("--max-candidates", type=int, default=0, help="Stop after this many matches; 0 is unlimited")
    parser.add_argument("--min-score", type=int, default=12)
    parser.add_argument("--from-date", help="Optional OAI UTC date/datetime lower bound")
    parser.add_argument("--resume-token", help="Continue from a token saved in the previous summary")
    args = parser.parse_args()

    process = CrawlerProcess()
    process.crawl(
        OpenBooksOaiSpider,
        catalog=args.catalog,
        output_path=str(args.output),
        summary_path=str(args.summary),
        max_pages=args.max_pages,
        max_candidates=args.max_candidates,
        min_score=args.min_score,
        from_date=args.from_date,
        resume_token=args.resume_token,
    )
    process.start()
    if args.summary.exists():
        print(args.summary.read_text(encoding="utf-8"))
        print(f"Candidates: {args.output}")


if __name__ == "__main__":
    main()
