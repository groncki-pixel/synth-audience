# synth-audience

Synthetic audience simulation pipeline for predicting audience reactions to film scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Ollama running locally at `localhost:11434`.

## Pipeline

1. **Data pipeline** — `python -m agents.preprocessor` (run once, builds lookup tables)
2. **Agent sampling** — `agents/sampler.py` generates agent profiles from survey distributions
3. **Content decomposition** — `content/decomposer.py` analyzes script structure
4. **Simulation** — `python -m simulation.runner` runs agents through content
5. **Aggregation** — `aggregation/aggregate.py` produces segment breakdowns
6. **Validation** — `validation/backtest.py` scores against ground truth
