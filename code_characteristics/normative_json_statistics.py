#!/usr/bin/env python3
"""Calculate structural and word-count statistics for hierarchical regulations.

Expected hierarchy (five requested levels):
    sections
      -> subsections
        -> subsubsections
          -> subsubsubsections
            -> subsubsubsubsections

A top-level ``section`` is treated as an article for article-level statistics.
The script counts only textual values that belong to the regulation:

* hierarchy titles (``*_name``),
* ``content``,
* table captions and table text (optional, enabled by default).

It does NOT count JSON key names or metadata such as page numbers, image paths,
file names, etc. Table text is not counted twice when it is already embedded in
``content``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


# The first five levels requested by the user.
LEVEL_NAMES: tuple[str, ...] = (
    "sections",
    "subsections",
    "subsubsections",
    "subsubsubsections",
    "subsubsubsubsections",
)

# Known child-array keys. The last one is included to detect and preserve text
# from unexpectedly deeper structures, while reporting it separately.
CHILD_KEYS: tuple[str, ...] = (
    "subsections",
    "subsubsections",
    "subsubsubsections",
    "subsubsubsubsections",
    "subsubsubsubsubsections",
)

TITLE_KEYS: tuple[str, ...] = (
    "section_name",
    "subsection_name",
    "subsubsection_name",
    "subsubsubsection_name",
    "subsubsubsubsection_name",
    "subsubsubsubsubsection_name",
)

# Words can contain Spanish letters and internal hyphens/apostrophes.
# Pure numbers are counted as tokens by default, as common word processors do.
WORD_RE_WITH_NUMBERS = re.compile(
    r"[^\W_]+(?:[-'’][^\W_]+)*",
    flags=re.UNICODE,
)
WORD_RE_WITHOUT_NUMBERS = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[-'’][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*",
    flags=re.UNICODE,
)


@dataclass
class ArticleStats:
    index: int
    title: str
    word_count: int = 0
    node_count: int = 0
    counts_by_level: Counter[int] = field(default_factory=Counter)
    max_depth: int = 0
    depth_sum: int = 0

    @property
    def average_node_depth(self) -> float:
        return self.depth_sum / self.node_count if self.node_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_index": self.index,
            "article_title": self.title,
            "word_count": self.word_count,
            "node_count": self.node_count,
            "counts_by_level": {
                LEVEL_NAMES[depth - 1] if depth <= len(LEVEL_NAMES) else f"level_{depth}": count
                for depth, count in sorted(self.counts_by_level.items())
            },
            "max_hierarchy_depth": self.max_depth,
            "average_node_depth": round(self.average_node_depth, 4),
        }


def normalize_for_comparison(text: str) -> str:
    """Normalize text only for duplicate detection, not for word counting."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def find_title(node: dict[str, Any], depth: int) -> str:
    """Return the hierarchy title, tolerating mildly inconsistent schemas."""
    expected_index = min(depth - 1, len(TITLE_KEYS) - 1)
    expected_key = TITLE_KEYS[expected_index]
    value = node.get(expected_key)
    if isinstance(value, str):
        return value.strip()

    # Some generated JSON files reuse a shallower title key at deeper levels.
    for key, candidate in node.items():
        if key.endswith("_name") and isinstance(candidate, str):
            return candidate.strip()
    return ""


def child_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield all recognized hierarchical children in schema order."""
    for key in CHILD_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    yield child


def textual_fragments(
    node: dict[str, Any],
    depth: int,
    *,
    include_tables: bool,
    include_table_captions: bool,
) -> list[str]:
    """Extract only regulation text, excluding JSON metadata."""
    fragments: list[str] = []

    title = find_title(node, depth)
    if title:
        fragments.append(title)

    content = node.get("content")
    content = content if isinstance(content, str) else ""
    if content.strip():
        fragments.append(content)

    if not include_tables:
        return fragments

    # Avoid duplicate table text when an extraction pipeline has already copied
    # it into the node's content field.
    normalized_content = normalize_for_comparison(content)
    local_seen: set[str] = set()
    tables = node.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue

            candidates: list[str] = []
            if include_table_captions:
                caption = table.get("caption")
                if isinstance(caption, str) and caption.strip():
                    candidates.append(caption)

            table_text = table.get("text")
            if isinstance(table_text, str) and table_text.strip():
                candidates.append(table_text)

            for candidate in candidates:
                normalized = normalize_for_comparison(candidate)
                if not normalized or normalized in local_seen:
                    continue
                local_seen.add(normalized)

                # Exact substring detection after whitespace/case normalization.
                if normalized_content and normalized in normalized_content:
                    continue
                fragments.append(candidate)

    return fragments


def count_words(text: str, *, count_numbers: bool) -> int:
    regex = WORD_RE_WITH_NUMBERS if count_numbers else WORD_RE_WITHOUT_NUMBERS
    return len(regex.findall(text))


def node_word_count(
    node: dict[str, Any],
    depth: int,
    *,
    include_tables: bool,
    include_table_captions: bool,
    count_numbers: bool,
) -> int:
    return sum(
        count_words(fragment, count_numbers=count_numbers)
        for fragment in textual_fragments(
            node,
            depth,
            include_tables=include_tables,
            include_table_captions=include_table_captions,
        )
    )


def walk_article(
    node: dict[str, Any],
    depth: int,
    article: ArticleStats,
    overall_counts: Counter[int],
    *,
    include_tables: bool,
    include_table_captions: bool,
    count_numbers: bool,
) -> None:
    article.node_count += 1
    article.counts_by_level[depth] += 1
    article.max_depth = max(article.max_depth, depth)
    article.depth_sum += depth
    overall_counts[depth] += 1

    article.word_count += node_word_count(
        node,
        depth,
        include_tables=include_tables,
        include_table_captions=include_table_captions,
        count_numbers=count_numbers,
    )

    for child in child_nodes(node):
        walk_article(
            child,
            depth + 1,
            article,
            overall_counts,
            include_tables=include_tables,
            include_table_captions=include_table_captions,
            count_numbers=count_numbers,
        )


def descriptive_stats(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "minimum": None, "maximum": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "minimum": min(values),
        "maximum": max(values),
    }


def calculate_statistics(
    json_path: Path,
    *,
    include_tables: bool = True,
    include_table_captions: bool = True,
    count_numbers: bool = True,
) -> dict[str, Any]:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {json_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"The JSON root must be an object: {json_path}")

    sections = data.get("sections")
    if not isinstance(sections, list):
        raise ValueError(f"The JSON must contain a root 'sections' array: {json_path}")

    overall_counts: Counter[int] = Counter()
    articles: list[ArticleStats] = []

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue
        article = ArticleStats(
            index=index,
            title=find_title(section, depth=1) or f"Section {index}",
        )
        walk_article(
            section,
            depth=1,
            article=article,
            overall_counts=overall_counts,
            include_tables=include_tables,
            include_table_captions=include_table_captions,
            count_numbers=count_numbers,
        )
        articles.append(article)

    article_lengths = [article.word_count for article in articles]
    article_max_depths = [article.max_depth for article in articles]
    article_average_depths = [article.average_node_depth for article in articles]

    requested_level_counts = {
        level_name: overall_counts.get(depth, 0)
        for depth, level_name in enumerate(LEVEL_NAMES, start=1)
    }
    deeper_counts = {
        f"level_{depth}": count
        for depth, count in sorted(overall_counts.items())
        if depth > len(LEVEL_NAMES)
    }

    return {
        "source_file": json_path.name,
        "definitions": {
            "article": "Each top-level object in the root 'sections' array.",
            "article_length": (
                "Word count of the article title, its own content and all descendant hierarchy levels."
            ),
            "hierarchy_depth": (
                "Top-level section = 1, subsection = 2, subsubsection = 3, "
                "subsubsubsection = 4, subsubsubsubsection = 5."
            ),
            "average_node_depth": (
                "Arithmetic mean of the depths of all hierarchy nodes contained in an article."
            ),
        },
        "options": {
            "include_tables": include_tables,
            "include_table_captions": include_table_captions,
            "count_numbers_as_words": count_numbers,
            "table_text_duplicate_detection": True,
        },
        "hierarchy_counts": requested_level_counts,
        "nodes_beyond_requested_depth": deeper_counts,
        "total_normative_word_count": sum(article_lengths),
        "article_count": len(articles),
        "article_length_words": descriptive_stats(article_lengths),
        "article_hierarchy": {
            "average_max_depth_per_article": (
                round(statistics.fmean(article_max_depths), 4) if article_max_depths else None
            ),
            "max_depth_across_articles": max(article_max_depths) if article_max_depths else None,
            "min_depth_across_articles": min(article_max_depths) if article_max_depths else None,
            "average_of_article_average_node_depths": (
                round(statistics.fmean(article_average_depths), 4)
                if article_average_depths
                else None
            ),
        },
        "articles": [article.to_dict() for article in articles],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate hierarchy and word-count statistics for regulation JSON files."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more input JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory in which to create '<input>.statistics.json' files. "
            "Without this option, results are printed to standard output."
        ),
    )
    parser.add_argument(
        "--exclude-tables",
        action="store_true",
        help="Do not count table text or table captions.",
    )
    parser.add_argument(
        "--exclude-table-captions",
        action="store_true",
        help="Count table text but not table captions.",
    )
    parser.add_argument(
        "--exclude-numbers",
        action="store_true",
        help="Do not count pure numeric tokens as words.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for json_path in args.json_files:
        try:
            result = calculate_statistics(
                json_path,
                include_tables=not args.exclude_tables,
                include_table_captions=not args.exclude_table_captions,
                count_numbers=not args.exclude_numbers,
            )
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output_dir is None:
            if len(args.json_files) > 1:
                print(f"\n===== {json_path.name} =====")
            print(rendered)
        else:
            output_name = f"{json_path.stem}.statistics.json"
            output_path = args.output_dir / output_name
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(output_path)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
