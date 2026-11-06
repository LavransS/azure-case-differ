"""Turn manually saved TFT patch note pages into search-ready chunks.

Reads every .html file in data/raw and writes one JSON object per line to
data/processed/chunks.jsonl. Each chunk is one section of one patch, so the
designers' reasoning stays together with the changes it explains.

Usage:
    python scripts/parse_patches.py
    python scripts/parse_patches.py --input data/raw --output data/processed/chunks.jsonl
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

OVERVIEW = "OVERVIEW"
# Riot words these headings differently from patch to patch.
MID_PATCH = re.compile(r"mid-patch|(?<![a-z])b[- ]patch", re.IGNORECASE)
DATE = r"[A-Z]+ \d{1,2}(?:ST|ND|RD|TH)?"
DATE_HEADING = re.compile(rf"^{DATE}(?: AND {DATE})?$", re.IGNORECASE)
# Sections such as "LARGE CHANGES" open with a one-line joke, not information.
FLAVOUR_TEXT_MAX_CHARS = 120


@dataclass
class Group:
    """The text found under one heading while walking the page."""

    section: str
    subsection: str = ""
    lines: list[str] = field(default_factory=list)
    has_bullets: bool = False

    @property
    def label(self) -> str:
        return f"{self.section} > {self.subsection}" if self.subsection else self.section


def clean(text: str) -> str:
    """Collapse whitespace and put single spaces around the change arrow."""
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return re.sub(r"\s*⇒\s*", " ⇒ ", text)


def read_metadata(soup: BeautifulSoup, path: Path) -> dict | None:
    """Read patch details from the page's structured data."""
    article = None
    for script in soup.find_all("script", type="application/ld+json"):
        graph = json.loads(script.string).get("@graph", [])
        article = next((n for n in graph if n.get("@type") == "TechArticle"), article)

    canonical = soup.find("link", rel="canonical")
    if not article or not article.get("version") or not canonical:
        return None

    tagline = soup.select_one('[data-testid="tagline"]')
    return {
        "patch": article["version"],
        "patch_date": article["datePublished"],
        "title": article["headline"],
        "url": canonical["href"],
        "tagline": clean(tagline.get_text()) if tagline else "",
        "source_file": path.name,
    }


def render_quote(quote: Tag) -> list[str]:
    """Return a designer quote as one line per paragraph."""
    for line_break in quote.find_all("br"):
        line_break.replace_with("\n")
    return [line for line in map(clean, quote.get_text().split("\n")) if line]


def render_list(list_tag: Tag, depth: int = 0) -> list[str]:
    """Return a possibly nested list as indented bullet lines."""
    lines = []
    for item in list_tag.find_all(recursive=False):
        if item.name == "ul":
            # Some pages put a list directly inside a list, not inside an item.
            lines.extend(render_list(item, depth + 1))
            continue
        own_text = "".join(
            child.get_text() if isinstance(child, Tag) else str(child)
            for child in item.children
            if not (isinstance(child, Tag) and child.name == "ul")
        )
        if text := clean(own_text):
            lines.append(f"{'  ' * depth}- {text}")
        for nested in item.find_all("ul", recursive=False):
            lines.extend(render_list(nested, depth + 1 if text else depth))
    return lines


def walk_groups(container: Tag) -> list[Group]:
    """Group the page's text under its section and subsection headings."""
    base_section = OVERVIEW
    current = Group(OVERVIEW)
    groups = [current]

    def start_group(section: str, subsection: str = "") -> Group:
        group = Group(section, subsection)
        groups.append(group)
        return group

    for element in container.find_all(["h2", "h3", "h4", "blockquote", "ul", "p"]):
        if element.find_parent(["blockquote", "ul"]):
            continue  # Already rendered as part of its parent.

        if element.name == "h2":
            if heading := clean(element.get_text()):
                base_section = heading
                current = start_group(heading)
        elif element.name in ("h3", "h4"):
            heading = clean(element.get_text())
            if not heading or "change-detail-title" not in element.get("class", []):
                continue
            if MID_PATCH.search(base_section) and DATE_HEADING.match(heading):
                # Keep the date on every change the update made.
                if not current.lines:
                    groups.remove(current)
                current = start_group(f"{base_section} ({heading})")
            else:
                current = start_group(current.section, heading)
        elif element.name == "blockquote":
            current.lines.extend(render_quote(element))
        elif element.name == "ul":
            current.lines.extend(render_list(element))
            current.has_bullets = True
        elif text := clean(element.get_text()):
            current.lines.append(text)

    return groups


def category_of(group: Group) -> str:
    """Tag chunks so search can leave out sections that only add noise."""
    if group.section == OVERVIEW:
        return "overview"
    if "COSMETIC" in group.section:
        return "cosmetics"
    if "BUG" in group.label or "PERFORMANCE" in group.label:
        return "bug_fixes"
    return "changes"


def is_flavour_text(group: Group, sections_with_subsections: set[str]) -> bool:
    return (
        not group.subsection
        and not group.has_bullets
        and not MID_PATCH.search(group.section)
        and group.section in sections_with_subsections
        and sum(map(len, group.lines)) <= FLAVOUR_TEXT_MAX_CHARS
    )


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_chunks(meta: dict, groups: list[Group]) -> tuple[list[dict], list[str]]:
    """Convert groups to chunks and report what was left out."""
    date = meta["patch_date"][:10]
    # Riot has reused version numbers, so the year keeps these ids unique.
    id_prefix = f"{slugify(meta['patch'])}-{date[:4]}"
    sections_with_subsections = {g.section for g in groups if g.subsection}
    seen_ids: Counter[str] = Counter()
    chunks, skipped = [], []

    for group in groups:
        if group.section == OVERVIEW and meta["tagline"]:
            group.lines.insert(0, meta["tagline"])
        if not group.lines:
            if group.section not in sections_with_subsections:
                skipped.append(f"{group.label} (no text, possibly image only)")
            continue
        if is_flavour_text(group, sections_with_subsections):
            skipped.append(f"{group.label} (flavour text)")
            continue

        base_id = f"{id_prefix}-{slugify(group.label)}"
        seen_ids[base_id] += 1
        count = seen_ids[base_id]
        header = f"TFT patch {meta['patch']} ({date}) - {group.label}"
        chunks.append(
            {
                "id": base_id if count == 1 else f"{base_id}-{count}",
                "patch": meta["patch"],
                "patch_date": meta["patch_date"],
                "title": meta["title"],
                "section": group.section,
                "subsection": group.subsection,
                "category": category_of(group),
                "is_mid_patch": bool(MID_PATCH.search(group.section)),
                "content": "\n".join([header, *group.lines]),
                "url": meta["url"],
                "source_file": meta["source_file"],
            }
        )
    return chunks, skipped


def parse_file(path: Path) -> tuple[list[dict], list[str]] | None:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    meta = read_metadata(soup, path)
    container = soup.select_one("#patch-notes-container")
    if meta is None or container is None:
        return None
    return build_chunks(meta, walk_groups(container))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()

    all_chunks = []
    for path in sorted(args.input.glob("*.html")):
        result = parse_file(path)
        if result is None:
            print(f"SKIPPED {path.name}: not a patch notes page")
            continue
        chunks, skipped = result
        all_chunks.extend(chunks)
        print(f"{path.name}: {len(chunks)} chunks")
        for reason in skipped:
            print(f"  left out: {reason}")

    all_chunks.sort(key=lambda chunk: chunk["patch_date"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
