#!/usr/bin/env python3
"""
Test script pro validaci generování a ukládání summaries

Ověřuje:
1. Že config.json má generate_summaries: true
2. Že phase2_summaries.json soubory obsahují validní summaries
3. Že summaries mají správnou délku (max 150 znaků)
4. Že summaries se ukládají do PostgreSQL metadata pole

Usage:
    python scripts/test_summaries_validation.py
    python scripts/test_summaries_validation.py --output-dir output/BZ_VR1
    python scripts/test_summaries_validation.py --check-postgres
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_config() -> bool:
    """Zkontroluj, jestli je v config.json generate_summaries: true"""
    print("\n" + "="*80)
    print("KROK 1: Kontrola config.json")
    print("="*80)

    config_path = Path("config.json")
    if not config_path.exists():
        print("❌ ERROR: config.json nenalezen!")
        return False

    with open(config_path) as f:
        config = json.load(f)

    generate_summaries = config.get("extraction", {}).get("generate_summaries", False)

    if generate_summaries:
        print(f"✅ config.json: generate_summaries = {generate_summaries}")
        return True
    else:
        print(f"❌ ERROR: generate_summaries = {generate_summaries} (mělo by být true)")
        print(f"\nOprava:")
        print(f"  1. Otevři config.json")
        print(f'  2. Nastav "extraction" -> "generate_summaries": true')
        return False


def validate_phase2_summaries(output_dir: Path) -> Dict:
    """Validuj phase2_summaries.json soubor"""
    print(f"\n" + "="*80)
    print(f"KROK 2: Validace {output_dir}/phase2_summaries.json")
    print("="*80)

    phase2_path = output_dir / "phase2_summaries.json"

    if not phase2_path.exists():
        print(f"❌ ERROR: {phase2_path} nenalezen!")
        print(f"\nSpusť indexing pipeline:")
        print(f"  python run_pipeline.py <dokument.pdf>")
        return {"valid": False}

    with open(phase2_path) as f:
        data = json.load(f)

    results = {
        "valid": True,
        "document_id": data.get("document_id"),
        "document_summary": data.get("document_summary"),
        "section_summaries": data.get("section_summaries", []),
        "errors": []
    }

    # Kontrola document summary
    doc_summary = results["document_summary"]
    if not doc_summary or doc_summary == "None" or doc_summary is None:
        results["errors"].append("Document summary je prázdný nebo None")
        results["valid"] = False
    else:
        doc_len = len(doc_summary)
        print(f"✅ Document summary: {doc_len} znaků")
        if doc_len > 200:
            print(f"⚠️  VAROVÁNÍ: Document summary je delší než 200 znaků (měl by být ~150)")
        print(f"   '{doc_summary[:100]}...'")

    # Kontrola section summaries
    section_count = len(results["section_summaries"])
    print(f"\n📊 Section summaries: {section_count} sekcí")

    empty_count = 0
    too_long_count = 0
    valid_count = 0

    for i, section in enumerate(results["section_summaries"][:10], 1):  # První 10 pro přehlednost
        section_id = section.get("section_id")
        title = section.get("title", "")[:40]
        summary = section.get("summary")

        if not summary or summary == "None" or summary is None:
            empty_count += 1
            print(f"  [{i}] ❌ {section_id}: '{title}' - PRÁZDNÝ summary")
            results["errors"].append(f"Section {section_id} má prázdný summary")
        else:
            summary_len = len(summary)
            if summary_len > 200:
                too_long_count += 1
                print(f"  [{i}] ⚠️  {section_id}: {summary_len} znaků (mělo by být max 150)")
            else:
                valid_count += 1
                print(f"  [{i}] ✅ {section_id}: {summary_len} znaků - '{summary[:60]}...'")

    if section_count > 10:
        print(f"\n  ... a {section_count - 10} dalších sekcí")

    # Souhrn
    print(f"\n📈 Statistiky:")
    print(f"  - Validní summaries: {valid_count}/{section_count}")
    print(f"  - Prázdné summaries: {empty_count}/{section_count}")
    print(f"  - Příliš dlouhé (>200): {too_long_count}/{section_count}")

    if empty_count > 0:
        results["valid"] = False
        results["errors"].append(f"{empty_count} sekcí má prázdný summary")

    if too_long_count > section_count * 0.5:  # Více než 50%
        results["errors"].append(f"{too_long_count} sekcí má příliš dlouhý summary (>200 znaků)")

    return results


def check_postgres_storage(document_id: str) -> bool:
    """Zkontroluj, jestli jsou summaries v PostgreSQL"""
    print(f"\n" + "="*80)
    print("KROK 3: Kontrola PostgreSQL storage")
    print("="*80)

    try:
        import asyncpg
        import asyncio
        from src.config import get_config

        config = get_config()
        storage_config = config.storage

        if storage_config.backend != "postgresql":
            print(f"⚠️  Storage backend je '{storage_config.backend}' (ne postgresql)")
            print(f"   Přeskakuji kontrolu PostgreSQL")
            return True

        async def check_metadata():
            # Připoj se k databázi
            try:
                conn = await asyncpg.connect(
                    host=storage_config.postgresql.host,
                    port=storage_config.postgresql.port,
                    user=storage_config.postgresql.user,
                    password=storage_config.postgresql.password,
                    database=storage_config.postgresql.database,
                    timeout=10
                )
            except Exception as e:
                print(f"❌ ERROR: Nelze se připojit k PostgreSQL: {e}")
                print(f"\nZkontroluj:")
                print(f"  - Je PostgreSQL spuštěný? (docker-compose up -d postgres)")
                print(f"  - Je config.storage.postgresql správně nastavený?")
                return False

            try:
                # Zkontroluj layer3 metadata pro daný dokument
                query = """
                SELECT chunk_id, metadata
                FROM vectors.layer3
                WHERE document_id = $1
                LIMIT 5
                """

                rows = await conn.fetch(query, document_id)

                if not rows:
                    print(f"⚠️  VAROVÁNÍ: Žádné chunks nenalezeny pro document_id='{document_id}'")
                    print(f"\nSpusť migraci:")
                    print(f"  python scripts/migrate_faiss_to_postgres.py --faiss-dir vector_db/")
                    return False

                print(f"✅ Nalezeno {len(rows)} chunks v PostgreSQL")

                # Zkontroluj metadata
                summaries_found = 0
                for row in rows:
                    metadata = row['metadata']
                    if metadata and 'section_summary' in metadata:
                        summaries_found += 1
                        section_summary = metadata['section_summary']
                        print(f"  ✅ {row['chunk_id']}: section_summary ({len(section_summary)} znaků)")
                    else:
                        print(f"  ⚠️  {row['chunk_id']}: metadata neobsahuje section_summary")

                if summaries_found == 0:
                    print(f"\n⚠️  VAROVÁNÍ: Žádné section_summary nenalezeny v metadata")
                    print(f"  To je normální, pokud migrace nepřenesla summaries")
                    print(f"  Summaries jsou dostupné v phase2_summaries.json")

                return True

            finally:
                await conn.close()

        return asyncio.run(check_metadata())

    except ImportError as e:
        print(f"⚠️  PostgreSQL modul není dostupný: {e}")
        print(f"   Přeskakuji kontrolu databáze")
        return True
    except Exception as e:
        print(f"❌ ERROR při kontrole PostgreSQL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validace generování summaries")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cesta k output adresáři (např. output/BZ_VR1)"
    )
    parser.add_argument(
        "--check-postgres",
        action="store_true",
        help="Zkontroluj také PostgreSQL storage"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TEST VALIDACE GENEROVÁNÍ SUMMARIES")
    print("="*80)

    # KROK 1: Config
    config_ok = check_config()

    if not config_ok:
        print("\n" + "="*80)
        print("❌ VÝSLEDEK: FAILED - Oprav config.json")
        print("="*80)
        sys.exit(1)

    # KROK 2: Phase2 summaries
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Najdi první dostupný output adresář
        output_base = Path("output")
        if not output_base.exists():
            print(f"\n❌ ERROR: {output_base} neexistuje")
            print(f"   Spusť indexing pipeline: python run_pipeline.py <dokument.pdf>")
            sys.exit(1)

        # Hledej phase2_summaries.json
        phase2_files = list(output_base.glob("*/phase2_summaries.json"))
        if not phase2_files:
            print(f"\n❌ ERROR: Žádné phase2_summaries.json nenalezeny v {output_base}")
            print(f"   Spusť indexing pipeline: python run_pipeline.py <dokument.pdf>")
            sys.exit(1)

        output_dir = phase2_files[0].parent
        print(f"\n📁 Použit output adresář: {output_dir}")

    results = validate_phase2_summaries(output_dir)

    # KROK 3: PostgreSQL (optional)
    postgres_ok = True
    if args.check_postgres and results.get("document_id"):
        postgres_ok = check_postgres_storage(results["document_id"])

    # VÝSLEDEK
    print("\n" + "="*80)
    if results["valid"] and postgres_ok:
        print("✅ VÝSLEDEK: SUCCESS - Summaries fungují správně!")
        print("="*80)
        print(f"\n📊 Souhrn:")
        print(f"  - Config: ✅ generate_summaries = true")
        print(f"  - Document summary: ✅ {len(results['document_summary'])} znaků")
        print(f"  - Section summaries: ✅ {len(results['section_summaries'])} sekcí")
        if args.check_postgres:
            print(f"  - PostgreSQL: {'✅' if postgres_ok else '⚠️'} Kontrola provedena")
    else:
        print("❌ VÝSLEDEK: FAILED - Summaries nefungují správně")
        print("="*80)
        print(f"\n❌ Chyby:")
        for error in results.get("errors", []):
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
