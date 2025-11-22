import json
import logging
import random
import argparse
from pathlib import Path
from typing import List, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Mock unstructured module to avoid dependency error if not installed
import sys
from unittest.mock import MagicMock

modules_to_mock = [
    "unstructured",
    "unstructured.partition",
    "unstructured.partition.auto",
    "unstructured.partition.pdf",
    "unstructured.partition.pptx",
    "unstructured.partition.docx",
    "unstructured.partition.image",
    "unstructured.partition.html",
    "unstructured.partition.xml",
    "unstructured.partition.md",
    "unstructured.partition.text",
    "unstructured.documents",
    "unstructured.documents.elements",
    "tiktoken",
]

for module_name in modules_to_mock:
    sys.modules[module_name] = MagicMock()

from src.utils.api_clients import APIClientFactory
from src.config import get_config, validate_config_on_startup, ModelConfig
from src.faiss_vector_store import FAISSVectorStore
from src.utils.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_question(client, model: str, provider: str, chunk_content: str) -> str:
    """Generate a question for a specific chunk using the LLM."""
    prompt = f"""You are an expert at creating evaluation benchmarks for RAG (Retrieval-Augmented Generation) systems.

Your task is to create a SPECIFIC question in CZECH language that can be answered using ONLY the provided text chunk.
The question should be challenging but unambiguous given the context.
Avoid generic questions like "What does the text say?" or "What is described?".
Focus on specific details, facts, definitions, or procedures mentioned in the text.

IMPORTANT: Output ONLY the question text. Do not include any introductory phrases, prefixes (e.g., 'Otázka:', 'Question:'), or explanations.

Chunk content:
{chunk_content}

Question (in Czech):"""

    try:
        if provider == "claude" or provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        elif provider == "openai":
            # Handle parameter differences for newer models (o1, gpt-5)
            completion_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Use max_completion_tokens for newer models, max_tokens for older ones
            # The error message suggests gpt-5-mini requires max_completion_tokens and fixed temperature
            if "o1" in model or "gpt-5" in model:
                completion_params["max_completion_tokens"] = 2000
                # Temperature is often fixed or restricted for reasoning models
                # completion_params["temperature"] = 1 
            else:
                completion_params["max_tokens"] = 150
                completion_params["temperature"] = 0.7
                
            response = client.chat.completions.create(**completion_params)
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate question: {e}")
        return ""

def is_valid_chunk(chunk: Dict, target_document_id: str = None) -> bool:
    """
    Check if a chunk is valid for benchmark generation.
    
    Criteria:
    1. Matches target_document_id (if provided)
    2. Content length > 50 characters
    3. Word count > 5 words
    """
    chunk_id = chunk.get("chunk_id", "")
    document_id = chunk.get("document_id", "")
    content = chunk.get("content", "")
    
    # 1. Document ID filter
    if target_document_id:
        # Check if target_document_id is part of the chunk's document_id
        # This allows for partial matches (e.g. "BZ_VR1" matches "BZ_VR1.pdf")
        if target_document_id not in document_id:
            return False
            
    # 2. Content quality filter
    if not content:
        return False
        
    if len(content) < 50:
        return False
        
    if len(content.split()) < 5:
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG benchmark")
    parser.add_argument("--count", type=int, default=50, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default="benchmark_dataset/synthetic_questions.json", help="Output JSON file")
    parser.add_argument("--vector_store", type=str, default="vector_db", help="Path to vector store")
    parser.add_argument("--document_id", type=str, default=None, help="Filter chunks by document ID (partial match)")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use (e.g., gpt-5-mini, claude-sonnet)")
    args = parser.parse_args()

    # Validate config
    config = validate_config_on_startup()
    model_config = ModelConfig.from_config(config)
    
    # Determine model and provider
    if args.model:
        llm_model = ModelRegistry.resolve_llm(args.model)
        llm_provider = ModelRegistry.get_provider(llm_model, "llm")
        logger.info(f"Using overridden model: {llm_model} (provider: {llm_provider})")
    else:
        llm_model = model_config.llm_model
        llm_provider = model_config.llm_provider
        logger.info(f"Using configured model: {llm_model} (provider: {llm_provider})")
    
    # Load vector store
    vector_store_path = Path(args.vector_store)
    if not vector_store_path.exists():
        logger.error(f"Vector store not found at {vector_store_path}")
        sys.exit(1)
        
    logger.info(f"Loading vector store from {vector_store_path}...")
    vector_store = FAISSVectorStore.load(vector_store_path)
    
    # Get Layer 3 chunks (primary retrieval layer)
    logger.info("Retrieving chunks from Layer 3...")
    chunks_metadata = vector_store.metadata_layer3
    
    if not chunks_metadata:
        logger.error("No chunks found in Layer 3")
        sys.exit(1)
        
    logger.info(f"Found {len(chunks_metadata)} total chunks in Layer 3")
    
    # Filter chunks
    logger.info("Filtering chunks...")
    valid_chunks = [
        chunk for chunk in chunks_metadata 
        if is_valid_chunk(chunk, args.document_id)
    ]
    
    if not valid_chunks:
        logger.error(f"No valid chunks found after filtering (Document ID: {args.document_id})")
        sys.exit(1)
        
    logger.info(f"Found {len(valid_chunks)} valid chunks after filtering")
    
    # Sample chunks
    num_samples = min(args.count, len(valid_chunks))
    sampled_chunks = random.sample(valid_chunks, num_samples)
    logger.info(f"Sampled {num_samples} chunks for benchmark generation")
    
    # Initialize LLM client
    logger.info(f"Initializing LLM client ({llm_provider}: {llm_model})...")
    if llm_provider == "claude" or llm_provider == "anthropic":
        client = APIClientFactory.create_anthropic()
    elif llm_provider == "openai":
        client = APIClientFactory.create_openai()
    else:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        sys.exit(1)
        
    # Generate questions
    benchmark_data = []
    
    logger.info("Generating questions...")
    for i, chunk in enumerate(sampled_chunks):
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        document_id = chunk.get("document_id")
        
        if not content:
            logger.warning(f"Skipping chunk {chunk_id}: No content")
            continue
            
        logger.info(f"[{i+1}/{num_samples}] Generating question for chunk {chunk_id}...")
        question = generate_question(client, llm_model, llm_provider, content)
        
        if question:
            benchmark_data.append({
                "question": question,
                "ground_truth_chunk_id": chunk_id,
                "ground_truth_document_id": document_id,
                "chunk_content_preview": content[:200] + "..."
            })
        else:
            logger.warning(f"Failed to generate question for chunk {chunk_id}")
            
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Benchmark generated with {len(benchmark_data)} questions")
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
