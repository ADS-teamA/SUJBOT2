# Setup Guide - SUJBOT2

Quick start guide to get SUJBOT2 up and running.

---

## Prerequisites

- Python 3.10 or higher
- 4GB+ RAM (8GB+ recommended)
- Claude API key from console.anthropic.com

---

## Installation Steps

### 1. Clone/Navigate to Project

```bash
cd /Users/michalprusek/PycharmProjects/SUJBOT2/advanced_sujbot2
```

### 2. Create Virtual Environment

```bash
# Using standard venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# OR using uv (faster)
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# OR using uv (faster)
uv pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes due to PyTorch and other ML dependencies.

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env  # or use your favorite editor
```

**Required in .env**:
```bash
CLAUDE_API_KEY=sk-ant-your-actual-key-here
```

### 5. Verify Installation

```bash
# Test imports (after dependencies are installed)
python3 -c "from src.config import load_config; print('✓ Configuration system working')"

python3 -c "from src.utils import count_tokens; print('✓ Utilities working')"

python3 -c "from src.logging_config import setup_logging; print('✓ Logging system working')"
```

### 6. Create Required Directories

```bash
mkdir -p logs
mkdir -p data/indices
mkdir -p uploads
```

---

## Configuration

### Quick Start Configuration

The default `config.yaml` is already set up with sensible defaults. For most use cases, you only need to set your API key in `.env`.

### Custom Configuration

Edit `config.yaml` to customize:

1. **Retrieval weights** (semantic/keyword/structural)
2. **Chunk sizes** (for document splitting)
3. **Model selection** (embedding model, LLM models)
4. **Performance** (workers, batch sizes)

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for details.

---

## Verify Setup

### Test Configuration Loading

```python
# test_config.py
from src.config import load_config
from src.logging_config import setup_logging_from_config, logger

# Load config
config = load_config("config.yaml")

# Setup logging
setup_logging_from_config(config.to_dict())

# Print configuration
logger.info(f"Main model: {config.get('llm.main_model')}")
logger.info(f"Embedding model: {config.get('embeddings.model')}")
logger.info(f"Chunk size: {config.get('chunking.target_chunk_size')}")

print("✅ Configuration system working!")
```

Run:
```bash
python test_config.py
```

### Test Utilities

```python
# test_utils.py
from src.utils import (
    normalize_text,
    count_tokens,
    extract_legal_references,
    ensure_dir
)

# Test text processing
text = "§89 odst. 2 stanoví povinnosti dodavatele"
refs = extract_legal_references(text)
print(f"Legal references: {refs}")

# Test token counting
tokens = count_tokens(text)
print(f"Tokens: {tokens}")

# Test directory creation
ensure_dir("./temp_test")
print("✅ Utilities working!")
```

Run:
```bash
python test_utils.py
```

---

## Common Issues

### Issue: ModuleNotFoundError

**Symptom**: `ModuleNotFoundError: No module named 'yaml'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: API Key Error

**Symptom**: `APIKeyError: Claude API key not found`

**Solution**: Set API key in `.env`
```bash
echo "CLAUDE_API_KEY=sk-ant-your-key" >> .env
```

### Issue: CUDA/GPU Errors

**Symptom**: Errors about CUDA or GPU

**Solution**: Use CPU mode (default) or install GPU version:
```bash
# For GPU support
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Issue: Import Errors

**Symptom**: Cannot import from src

**Solution**: Ensure you're in the project root:
```bash
cd /Users/michalprusek/PycharmProjects/SUJBOT2/advanced_sujbot2
```

---

## Development Setup

### Install Development Dependencies

Development dependencies are included in `requirements.txt`:
- pytest (testing)
- black (formatting)
- flake8 (linting)
- mypy (type checking)

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

### Format Code

```bash
# Format with black
black src/

# Sort imports
isort src/

# Lint with flake8
flake8 src/
```

### Type Checking

```bash
mypy src/
```

---

## Production Deployment

### Environment Variables

Set all required environment variables:
```bash
export CLAUDE_API_KEY=sk-ant-xxx
export LOG_LEVEL=INFO
export MAX_PARALLEL_AGENTS=5
export EMBEDDING_DEVICE=cuda  # if GPU available
```

### Performance Optimization

Edit `config.yaml`:
```yaml
performance:
  max_workers: 16  # Increase for multi-core
  enable_gpu: true  # If GPU available
  cache_size: 5000  # Larger cache

embeddings:
  device: "cuda"  # Use GPU
  batch_size: 64  # Larger batches

indexing:
  index_type: "hnsw"  # Faster approximate search
```

### Run with Gunicorn (for API)

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

---

## Directory Structure

After setup, your directory should look like:

```
advanced_sujbot2/
├── .env                      ← Your environment config (not in git)
├── .env.example              ← Template
├── config.yaml               ← Main configuration
├── requirements.txt          ← Dependencies
├── CONFIGURATION_GUIDE.md    ← Configuration docs
├── SETUP_GUIDE.md           ← This file
│
├── src/                      ← Source code
│   ├── __init__.py
│   ├── config.py            ← Configuration management
│   ├── utils.py             ← Utilities
│   ├── logging_config.py    ← Logging setup
│   ├── models.py            ← Data models
│   ├── document_reader.py   ← Document parsing
│   ├── knowledge_graph.py   ← Knowledge graph
│   ├── compliance_*.py      ← Compliance modules
│   └── ...
│
├── logs/                     ← Log files
├── data/                     ← Data storage
│   └── indices/             ← Vector indices
├── uploads/                  ← Uploaded documents
└── tests/                    ← Test files
```

---

## Next Steps

1. ✅ Complete installation
2. ✅ Configure `.env` with API key
3. ✅ Verify setup with test scripts
4. 📖 Read [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
5. 📖 Read [README.md](README.md) for system overview
6. 🚀 Start using the system!

---

## Usage Example

```python
#!/usr/bin/env python3
"""
Quick start example
"""

from src.config import load_config
from src.logging_config import setup_logging_from_config, logger
from src.document_reader import LegalDocumentReader

def main():
    # 1. Load configuration
    config = load_config("config.yaml")

    # 2. Setup logging
    setup_logging_from_config(config.to_dict())
    logger.info("System initialized")

    # 3. Create document reader
    reader = LegalDocumentReader()

    # 4. Process document
    doc = reader.read_document("path/to/document.pdf")
    logger.info(f"Read document: {doc.metadata.title}")

    # 5. Your processing logic here...

if __name__ == "__main__":
    main()
```

---

## Support & Documentation

- **Configuration**: [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
- **API Reference**: [README.md](README.md)
- **Specifications**: [specs/](specs/)
- **Examples**: [config_usage_example.py](config_usage_example.py)

---

## Troubleshooting Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with API key
- [ ] Required directories created (`logs/`, `data/indices/`)
- [ ] Can import from src (`from src.config import load_config`)
- [ ] Configuration loads without errors

If all checked and still having issues, check logs in `logs/system.log`.

---

**Version**: 1.0.0
**Last Updated**: 2025-10-08
