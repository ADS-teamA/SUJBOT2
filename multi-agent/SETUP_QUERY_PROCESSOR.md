# Query Processor Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd multi-agent

# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install/update anthropic if needed
pip install anthropic>=0.68.0

# Or install all requirements
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Create .env file if it doesn't exist
cp .env.example .env

# Edit .env and add your Claude API key
# CLAUDE_API_KEY=your-api-key-here
```

### 3. Verify Installation

```python
python3 -c "
import sys
sys.path.insert(0, 'src')
from query_processor import QueryProcessor, QueryIntent, QueryComplexity
print('✅ Query Processor installed successfully')
"
```

### 4. Run Tests

```bash
# Quick feature test
python test_query_processor.py --mode features

# Full test suite (requires API key)
python test_query_processor.py --mode full

# Interactive mode
python test_query_processor.py --mode interactive

# Unit tests with pytest
pytest tests/test_query_processor.py -v
```

### 5. Run Demo

```bash
python examples/query_processor_demo.py
```

## Usage Example

```python
import asyncio
import yaml
from src.query_processor import QueryProcessor

async def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize processor
    processor = QueryProcessor(config)

    # Process query
    query = "Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?"
    processed = await processor.process(query)

    # Print results
    print(f"Intent: {processed.intent.value}")
    print(f"Complexity: {processed.complexity.value}")
    print(f"Entities: {len(processed.entities)}")
    print(f"Sub-queries: {len(processed.sub_queries)}")

    if processed.sub_queries:
        for sq in processed.sub_queries:
            print(f"  {sq.priority}. {sq.text}")

asyncio.run(main())
```

## Configuration

Edit `config.yaml` to customize:

```yaml
query_processing:
  # Model selection
  llm_model: "claude-3-5-haiku-20241022"
  llm_temperature: 0.3
  llm_max_tokens: 1000

  # Features
  enable_decomposition: true
  max_sub_queries: 5
  enable_query_expansion: true

  # Logging
  verbose_logging: true  # Set to true for debugging
```

## Troubleshooting

### ModuleNotFoundError: No module named 'anthropic'

```bash
pip install anthropic>=0.68.0
```

### ValueError: API key required

Set `CLAUDE_API_KEY` in `.env` file or environment:

```bash
export CLAUDE_API_KEY=your-api-key-here
```

### Tests fail with API errors

- Check API key is valid
- Check internet connection
- Verify API rate limits not exceeded

## Documentation

- **Complete Guide**: `QUERY_PROCESSOR.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY_QUERY_PROCESSOR.md`
- **API Reference**: See `QUERY_PROCESSOR.md` → API Reference section

## Next Steps

1. Read `QUERY_PROCESSOR.md` for detailed documentation
2. Run `examples/query_processor_demo.py` to see all features
3. Try interactive mode: `python test_query_processor.py --mode interactive`
4. Integrate with your retrieval pipeline (see Integration section in docs)

## Support

For issues or questions:
- Check `QUERY_PROCESSOR.md` troubleshooting section
- Review test cases in `tests/test_query_processor.py`
- See examples in `examples/query_processor_demo.py`
