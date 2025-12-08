#!/usr/bin/env python3
"""
Analyze page break continuity in extracted JSON.
"""

import json
from pathlib import Path

if __name__ == "__main__":
    json_file = Path("phase1_output/unstructured_yolox_extraction.json")

    with open(json_file) as f:
        data = json.load(f)

    sections = data["sections"]
    prev_page = None

    print("🔍 Analyzing page break continuity...\n")
    print("="*80)

    for i, s in enumerate(sections[:60]):
        page = s["page_number"]

        # Detect page break
        if page != prev_page and prev_page is not None:
            print(f"\n{'='*80}")
            print(f"📄 PAGE BREAK: page {prev_page} -> {page}")
            print(f"{'='*80}\n")

        parent_id = s["unstructured_parent_id"]
        parent_short = parent_id[:8] + "..." if parent_id else "None"
        title = s["title"][:40] if s["title"] else "(no title)"

        print(f"[{i+1:2d}] {s['section_id']:8s} | page={page} | level={s['level']} | "
              f"parent={parent_short:12s} | {title}")

        prev_page = page

    print("\n" + "="*80)
    print("✅ Analysis complete")
