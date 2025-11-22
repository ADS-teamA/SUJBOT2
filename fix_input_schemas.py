#!/usr/bin/env python3
"""Add missing Input schema classes to tool files"""

import re

# Mapping of file → Input class name
FIXES = {
    "filtered_search.py": "FilteredSearchInput",
    "expand_context.py": "ExpandContextInput",
    "similarity_search.py": "SimilaritySearchInput",
    "contextual_chunk_enricher.py": "ContextualChunkEnricherInput",
    "multi_doc_synthesizer.py": "MultiDocSynthesizerInput",
    "assess_retrieval_confidence.py": "AssessRetrievalConfidenceInput",
    "cluster_search.py": "ClusterSearchInput",
    "explain_search_results.py": "ExplainSearchResultsInput",
    "browse_entities.py": "BrowseEntitiesInput",
    # graph_search.py already fixed manually
}

# Read extracted schemas
with open("input_schemas_extracted.txt") as f:
    schemas_text = f.read()

# Extract each schema
schemas = {}
current_name = None
current_lines = []

for line in schemas_text.split('\n'):
    if line.startswith('# ') and line[2:].strip().endswith('Input'):
        if current_name:
            schemas[current_name] = '\n'.join(current_lines)
        current_name = line[2:].strip()
        current_lines = []
    elif line.startswith('class '):
        current_lines.append(line)
    elif current_name and current_lines:
        current_lines.append(line)

if current_name:
    schemas[current_name] = '\n'.join(current_lines)

# Apply fixes
for filename, schema_name in FIXES.items():
    filepath = f"src/agent/tools/{filename}"

    if schema_name not in schemas:
        print(f"⚠️  Schema {schema_name} not found in extracted schemas")
        continue

    # Read file
    with open(filepath) as f:
        content = f.read()

    # Find insertion point (after logger, before @register_tool)
    # Pattern: logger line + blank line + @register_tool
    pattern = r"(logger = logging\.getLogger\(__name__\)\n)\n(@register_tool)"

    schema_code = schemas[schema_name].strip()

    # Insert schema before @register_tool
    new_content = re.sub(
        pattern,
        r"\1\n\n" + schema_code + "\n\n\n\\2",
        content,
        count=1
    )

    if new_content != content:
        # Write back
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"✅ Fixed {filename}")
    else:
        print(f"❌ Failed to fix {filename} - pattern not matched")

print("\nDone!")
