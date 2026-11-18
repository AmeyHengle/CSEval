# CSEval

Dataset: [Aswini123/CSEval](https://huggingface.co/datasets/Aswini123/CSEval) on Hugging Face.

## setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AmeyHengle/CSEval.git
cd CSEval
uv sync --extra llm
```

## data

The code reads the dataset from `data/CSEval-hf/`.

```bash
git clone https://huggingface.co/datasets/Aswini123/CSEval data/CSEval-hf
uv run cseval data
```

## code

Set `OPENAI_API_KEY` or `TOGETHER_API_KEY` for the provider in the config, then:

```bash
uv run cseval calibrate configs/calibrate.toml
uv run cseval score configs/score.toml
uv run cseval evaluate runs/<score-run>
```

settings are in `configs/`. 
To score with the calibrated steps, set `kind = "auto-cseval"` and `calibration_run` in `configs/score.toml`. Results are written to `runs/`
