# PaperTrace

Data and code for **Can AI Reviewers See Your Efforts to Improve Your Paper? A Large-Scale Study of Paper Revision Trajectories**.

The AI-review release contains **1,438 papers, 1,484 revision pairs, and 69,909 reviews** from eight models under neutral, positive, and negative conditions. Reviews include overall scores, strengths, and weaknesses. Manuscript PDFs are downloaded separately.

## Repository Layout

```text
data/
  versions.jsonl.gz          # Exact arXiv versions and PDF hashes
  revision_pairs.jsonl.gz    # Sample pairs, revision labels and prediction folds
code/
  download_sample.py
  anonymize_codex.py
  analyze.py
review/
  openrouter_runner.py
  prompts/                   # Conference criteria and review conditions
  results/                   # Complete reviews, grouped by model
```

Data files use gzip-compressed JSON Lines.

## Setup

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation

Run the key analyses directly from the released review results and revision labels:

```bash
python code/analyze.py
```

This computes score responses, reviewer agreement, revision-type associations, and grouped prediction. Outputs are saved to `results/`; no API calls are needed.

## Review Pipeline

To prepare one manuscript and obtain new OpenRouter reviews:

```bash
python code/download_sample.py --paper 2305.12996
pip install -r code/requirements-anonymize.txt
python code/anonymize_codex.py downloads_papers/iclr2026/2305.12996/v1.pdf
python review/openrouter_runner.py clean_data/iclr2026/2305.12996/v1.pdf \
  --model YOUR_OPENROUTER_MODEL_ID --output results/openrouter-run --ask-api-key
```

Anonymization requires an authenticated Codex CLI and an installed PDF skill. Replace the model placeholder with the chosen OpenRouter ID. New anonymization and review runs use model quota; add `--dry-run` to preview either step.

## Citation

```bibtex

```
