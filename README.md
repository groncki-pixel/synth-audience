# synth-audience

Synthetic audience simulation pipeline for predicting audience reactions to
film scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m agents.preprocessor   # build sampler lookup tables (one-time)
```

## Inference providers

Choose one of three backends. The provider is selected at runtime via
`--provider {ollama,gemini,claude}`.

- **Ollama** (local) — requires Ollama running at `localhost:11434`.
  Default model: `mistral`.
- **Gemini** — set `GEMINI_API_KEY` in your environment or `.env`. Get a
  key at https://aistudio.google.com/app/apikey. Default model:
  `gemini-2.5-flash` (free tier, no per-token cost).
- **Claude** — set `ANTHROPIC_API_KEY` in your environment or `.env`. Get
  a key at https://console.anthropic.com/. Default model:
  `claude-haiku-4-5`.

`.env` files are loaded automatically when `python-dotenv` is installed.

## One-command pipeline

```bash
python scripts/pipeline.py --film anora \
    --script content/scripts/anora.txt \
    --n-agents 100 \
    --provider gemini
```

Runs sample agents -> decompose -> simulate -> aggregate -> validate, and
writes `agents.json`, `decomposition.json`, `results.jsonl`,
`cost_*.json`, and `aggregation.json` to `outputs/{film}_{timestamp}/`.

Useful flags:

- `--dry-run` — sample only; no LLM calls. Verifies the pipeline wires up.
- `--skip-decomposition` — reuse an existing decomposition file.
- `--model MODEL` — override the provider default.

## Step-by-step (legacy)

If you prefer to invoke each stage manually:

1. `python -m agents.preprocessor` — build lookup tables (one-time).
2. `python -m content.decomposer --script ... --model ...` — decompose script.
3. `python -m simulation.runner --provider ... --script ... --decomposition ... --agents ...` — simulate.
4. `python -m aggregation.aggregate --results ...` — aggregate.
5. `python -m validation.backtest ...` — score against ground truth.

## Tests

```bash
pytest
```

The suite is hermetic: no network, no Ollama. Provider SDKs are stubbed
via `sys.modules` so tests run even without `google-generativeai` or
`anthropic` installed.

## Cost tracking

Every population run writes `cost_{timestamp}.json` next to the results
JSONL. The summary includes total tokens in/out, total USD cost, and a
per-provider/model breakdown. Prices live in
`simulation/cost_tracker.PRICE_TABLE`.

## Notes for contributors

The agent-construction surface (`agents/sampler.py`,
`agents/preprocessor.py`, `agents/schemas.py`, the persona prompts in
`simulation/prompts.py`, and the derivation logic in
`_derive_content_sensitivities`) was deliberately not modified by the
provider/cost/test refactor. Future migrations to dynamic configs should
preserve those interfaces.
