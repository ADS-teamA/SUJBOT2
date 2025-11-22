#!/usr/bin/env python3
"""Add missing Input schemas - simple and direct approach"""

import subprocess

# Get tier2 file
result = subprocess.run(
    ["git", "show", "main:src/agent/tools/tier2_advanced.py"],
    capture_output=True, text=True, cwd="/home/prusemic/SUJBOT2"
)
tier2_content = result.stdout

# Files to fix (excluding graph_search.py which is done)
FILES = [
    ("filtered_search.py", "FilteredSearchInput"),
    ("expand_context.py", "ExpandContextInput"),
    ("similarity_search.py", "SimilaritySearchInput"),
    ("contextual_chunk_enricher.py", "ContextualChunkEnricherInput"),
    ("multi_doc_synthesizer.py", "MultiDocSynthesizerInput"),
    ("assess_retrieval_confidence.py", "AssessRetrievalConfidenceInput"),
    ("cluster_search.py", "ClusterSearchInput"),
    ("explain_search_results.py", "ExplainSearchResultsInput"),
    ("browse_entities.py", "BrowseEntitiesInput"),
]

def extract_class(content, class_name):
    """Extract a class definition from content"""
    lines = content.split('\n')
    result = []
    in_class = False

    for line in lines:
        if line.startswith(f'class {class_name}(ToolInput):'):
            in_class = True
            result.append(line)
        elif in_class:
            if line and not line[0].isspace() and not line.strip().startswith('#'):
                # End of class
                break
            result.append(line)

    return '\n'.join(result)

# Process each file
for filename, class_name in FILES:
    filepath = f"src/agent/tools/{filename}"

    # Extract schema from tier2
    schema = extract_class(tier2_content, class_name)

    if not schema:
        print(f"⚠️  Could not extract {class_name}")
        continue

    # Read target file
    with open(filepath) as f:
        content = f.read()

    # Find insert position (after logger, before @register_tool)
    insert_marker = "\n@register_tool"
    if insert_marker not in content:
        print(f"⚠️  No @register_tool found in {filename}")
        continue

    # Insert schema
    parts = content.split(insert_marker, 1)
    new_content = parts[0] + "\n\n" + schema + "\n\n" + insert_marker + parts[1]

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"✅ Fixed {filename} - added {class_name}")

print("\nDone!")
