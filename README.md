# GBT-RAG

Implementation of **Gradient Boosted Trees for Retrieval-Augmented Generation**. GBT-RAG decomposes questions into retrieval trees, targets under-covered facets, and accepts evidence-grounded rewrites through a coverage-loss gate.

## Installation

Use Python 3.10+; local LLM serving requires Linux and a compatible CUDA GPU. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt
```

For local vLLM serving, additionally install `requirements-serving.txt`. For tests, install `requirements-test.txt` and run `python -m pytest -q`.

## Prepare resources

Download the development sets from their original releases:

| Dataset | Default file |
| --- | --- |
| [HotpotQA](https://hotpotqa.github.io/) | `data/hotpotqa/hotpot_dev_distractor_v1.json` |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | `data/musique/musique_ans_v1.0_dev.jsonl` |
| [ASQA](https://github.com/google-research/language/tree/master/language/asqa) | `data/asqa/dev.json` |

For ASQA, save the official `ASQA.json` as `data/asqa/dev.json`; the loader selects its `dev` split and preserves per-interpretation answer aliases.

Provide local Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct weights, `intfloat/e5-base-v2` at `models/e5-base-v2`, and `BAAI/bge-reranker-v2-m3` at `models/bge-reranker-v2-m3`. Model weights and datasets are not bundled. Adjust paths in `configs/default.yaml`, or supply an edited `configs/server.example.yaml` using `--server-config`.

Build the retrieval index from the Dec-2018 [DPR Wikipedia passages](https://github.com/facebookresearch/DPR) (`id`, `text`, and `title` TSV columns):

```bash
python scripts/build_index.py \
  --corpus /path/to/psgs_w100.tsv.gz \
  --embedding-model ./models/e5-base-v2 \
  --output ./corpus/dpr_wikipedia_e5
```

Use a new, empty output directory. The index encoder must match the configured retrieval encoder.

## Run

Start an OpenAI-compatible model service in a separate terminal; this example assigns GPU 0:

```bash
bash scripts/start_vllm.sh /path/to/Meta-Llama-3.1-8B-Instruct 0 8001 gbtrag-llama8b
```

Then run a small evaluation:

```bash
python experiments/run.py --dataset hotpotqa --samples 20 --seed 42 --no-cache
```

Change `--dataset` to `musique` or `asqa`. Results are written under `results/<dataset>/`. HotpotQA and MuSiQue answer F1/EM are computed automatically. ASQA requires the additional scoring step below. Omit `--samples` to use the dataset configuration; ASQA defaults to 500 draws with replacement from its 948-question development pool. Defaults use three boosting updates and coverage threshold 0.65.

## Scope and attribution

Small real-model checks and 39 unit tests have passed. These checks do not establish reproduction of all paper tables. Bundled baselines are simplified references; BGE/candidate-passage configurations are debugging alternatives to the E5/full-Wikipedia setup. The historical text-only-critique ablation is not included.

Third-party ASQA and Transformers QA utilities retain their copyright notices and licenses in `third_party/`. Follow the original licenses and access conditions for all models, datasets, and corpora.
