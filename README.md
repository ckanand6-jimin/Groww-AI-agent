# Pulse Agent

MCP host and analysis pipeline for the Groww Weekly Review Pulse.

**Status:** Phase 0 scaffold — orchestration and pipeline modules are added in later phases.

## Local development

```bash
pip install -e ".[dev]"
pytest
```

## Configuration

Product settings live in `../config/groww.yaml`. Load and validate with:

```python
from pulse.config import load_product_config

config = load_product_config()
```
