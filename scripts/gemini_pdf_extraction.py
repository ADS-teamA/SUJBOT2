#!/usr/bin/env python3
"""
Extract document hierarchy using Gemini with full PDF upload.
"""

import json
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.0-flash"  # Use 2.0 to avoid copyright blocks

genai.configure(api_key=GOOGLE_API_KEY)

EXTRACTION_PROMPT = """Extrahuj hierarchii tohoto českého zákona do JSON.

JSON FORMÁT - kompaktní:
{"sections":[{"id":"sec_1","type":"cast","num":"I","lvl":2,"path":"ČÁST I","txt":""},{"id":"sec_2","type":"hlava","num":"PÁTÁ","lvl":3,"path":"ČÁST I > HLAVA PÁTÁ","txt":"OBČANSKOPRÁVNÍ ODPOVĚDNOST ZA JADERNÉ ŠKODY"},{"id":"sec_3","type":"paragraf","num":"32","lvl":4,"path":"ČÁST I > HLAVA PÁTÁ > § 32","txt":""},{"id":"sec_4","type":"odstavec","num":"1","par":"32","lvl":5,"txt":"Pro účely...26) kterou je ČR vázána."}]}

TYPY: cast, hlava, paragraf, odstavec, pismeno, poznamka
PRAVIDLA:
- num = číslo BEZ symbolů (32 ne § 32)
- txt = zkrácený obsah (max 100 znaků)
- zachovat 26), 29) v textu
- každý (1), (2) = samostatný odstavec

Extrahuj VŠECHNY §§ z dokumentu (32-38, 45, 49, 50).
Vrať POUZE JSON."""


def extract_with_pdf_upload(pdf_path: str) -> dict:
    """Extract hierarchy by uploading full PDF to Gemini."""
    print(f"Uploading PDF: {pdf_path}")

    # Upload file
    uploaded_file = genai.upload_file(pdf_path)
    print(f"  File uploaded: {uploaded_file.name}")

    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        print("  Processing...")
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        return {"error": f"File processing failed: {uploaded_file.state.name}"}

    print(f"  File ready: {uploaded_file.uri}")

    # Generate content
    model = genai.GenerativeModel(MODEL_ID)

    try:
        response = model.generate_content(
            [uploaded_file, EXTRACTION_PROMPT],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 16384,
            }
        )

        text = response.text

        # Clean markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        return json.loads(text.strip())

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": text[:2000] if 'text' in dir() else ""}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup
        try:
            genai.delete_file(uploaded_file.name)
            print("  File deleted from API")
        except Exception:
            pass  # Cleanup failure is not critical


def main():
    import sys

    # Accept command line argument or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/Sb_1997_18_2017-01-01_IZ.pdf"
        print("Usage: python gemini_pdf_extraction.py <pdf_path>")
        print(f"Using default: {pdf_path}\n")

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 60)
    print("Gemini PDF Hierarchy Extraction (Full Document)")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")

    result = extract_with_pdf_upload(pdf_path)

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        if "raw" in result:
            print(f"Raw output:\n{result['raw']}")
        return

    # Statistics
    sections = result.get("sections", [])
    by_type = {}
    for sec in sections:
        t = sec.get("type", "unknown")  # JSON schema uses "type", not "element_type"
        by_type[t] = by_type.get(t, 0) + 1

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total sections: {len(sections)}")
    print(f"By element type:\n{json.dumps(by_type, indent=2)}")

    # Add metadata
    result["extraction_model"] = MODEL_ID
    result["source_path"] = pdf_path

    # Save
    output_path = Path("data/Sb_1997_18_gemini_full.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")

    # Show sample
    print("\n" + "=" * 60)
    print("SAMPLE SECTIONS:")
    print("=" * 60)
    for sec in sections[:15]:
        el_type = sec.get("element_type", "?")
        num = sec.get("number", "")
        path = sec.get("path", "")
        content = sec.get("content", "")[:50]
        print(f"  [{el_type} {num}] {path}")
        if content:
            print(f"      → {content}...")


if __name__ == "__main__":
    main()
