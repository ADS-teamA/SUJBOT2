#!/usr/bin/env bash
#
# Vector Database Statistics
#
# Display statistics about the central vector database.
#
# Usage:
#   ./vector_db_stats.sh                # Show vector_db/ stats
#   ./vector_db_stats.sh <path>         # Show specific store stats
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_stat() {
    echo -e "${CYAN}  $1:${NC} $2"
}

# Get vector store path
VECTOR_STORE="${1:-vector_db}"

# Print header
echo ""
echo "========================================"
echo "  Vector Database Statistics"
echo "========================================"
echo ""

# Check if vector store exists
if [ ! -d "$VECTOR_STORE" ]; then
    print_error "Vector store not found: $VECTOR_STORE"
    echo ""
    echo "Create one by running:"
    echo "  ./add_to_vector_db.sh data/your_document.pdf"
    exit 1
fi

print_info "Vector Store: $VECTOR_STORE"
echo ""

# Check for FAISS indexes
print_info "FAISS Indexes:"
for layer in layer1 layer2 layer3; do
    INDEX_FILE="$VECTOR_STORE/${layer}.index"
    if [ -f "$INDEX_FILE" ]; then
        SIZE=$(du -h "$INDEX_FILE" | cut -f1)
        print_stat "$layer" "$SIZE"
    else
        print_warning "$layer index not found"
    fi
done

echo ""

# Check for BM25 indexes (hybrid search)
print_info "BM25 Indexes (Hybrid Search):"
BM25_FOUND=false
for layer in bm25_layer1 bm25_layer2 bm25_layer3; do
    BM25_FILE="$VECTOR_STORE/${layer}.pkl"
    if [ -f "$BM25_FILE" ]; then
        SIZE=$(du -h "$BM25_FILE" | cut -f1)
        LAYER_NUM=$(echo $layer | sed 's/bm25_layer/layer/')
        print_stat "$LAYER_NUM" "$SIZE"
        BM25_FOUND=true
    fi
done

if [ "$BM25_FOUND" = false ]; then
    print_warning "No BM25 indexes found (hybrid search disabled)"
fi

echo ""

# Check metadata
print_info "Metadata:"
METADATA_FILE="$VECTOR_STORE/metadata.pkl"
if [ -f "$METADATA_FILE" ]; then
    SIZE=$(du -h "$METADATA_FILE" | cut -f1)
    print_stat "metadata.pkl" "$SIZE"

    # Try to count chunks using Python
    if command -v python3 &> /dev/null; then
        CHUNK_COUNT=$(python3 -c "
import pickle
import sys
try:
    with open('$METADATA_FILE', 'rb') as f:
        metadata = pickle.load(f)

    # Count chunks in each layer
    layer1_count = len(metadata.get(1, []))
    layer2_count = len(metadata.get(2, []))
    layer3_count = len(metadata.get(3, []))

    print(f'{layer1_count},{layer2_count},{layer3_count}')
except Exception:
    print('N/A,N/A,N/A')
" 2>/dev/null || echo "N/A,N/A,N/A")

        IFS=',' read -r L1 L2 L3 <<< "$CHUNK_COUNT"
        if [ "$L1" != "N/A" ]; then
            echo ""
            print_stat "Layer 1 chunks (documents)" "$L1"
            print_stat "Layer 2 chunks (sections)" "$L2"
            print_stat "Layer 3 chunks (primary)" "$L3"
            TOTAL=$((L1 + L2 + L3))
            print_stat "Total chunks" "$TOTAL"
        fi
    fi
else
    print_warning "Metadata file not found"
fi

echo ""

# Check knowledge graph
print_info "Knowledge Graph:"
KG_FOUND=false
for kg_file in "$VECTOR_STORE"/*_kg.json knowledge_graph.json; do
    if [ -f "$kg_file" ]; then
        SIZE=$(du -h "$kg_file" | cut -f1)
        FILENAME=$(basename "$kg_file")
        print_stat "$FILENAME" "$SIZE"
        KG_FOUND=true

        # Count entities and relationships
        if command -v python3 &> /dev/null && command -v jq &> /dev/null; then
            ENTITY_COUNT=$(jq '.entities | length' "$kg_file" 2>/dev/null || echo "N/A")
            REL_COUNT=$(jq '.relationships | length' "$kg_file" 2>/dev/null || echo "N/A")
            if [ "$ENTITY_COUNT" != "N/A" ]; then
                print_stat "  Entities" "$ENTITY_COUNT"
                print_stat "  Relationships" "$REL_COUNT"
            fi
        fi
    fi
done

if [ "$KG_FOUND" = false ]; then
    print_warning "No knowledge graph found"
fi

echo ""

# Check configuration
print_info "Configuration:"
CONFIG_FILE="$VECTOR_STORE/hybrid_config.pkl"
if [ -f "$CONFIG_FILE" ]; then
    SIZE=$(du -h "$CONFIG_FILE" | cut -f1)
    print_stat "hybrid_config.pkl" "$SIZE"
else
    print_warning "Configuration file not found"
fi

echo ""

# Total size
print_info "Total Size:"
TOTAL_SIZE=$(du -sh "$VECTOR_STORE" | cut -f1)
print_stat "Vector store" "$TOTAL_SIZE"

echo ""
print_success "Statistics complete"
echo ""
