# HotpotQA 60/20/20 splits

These files are generated from the official HotpotQA distractor training data
with `scripts/split_hotpotqa.py`.

- `train-00000-of-00002.jsonl.gz` and `train-00001-of-00002.jsonl.gz`:
  prompt and workflow development (two shards because the combined file exceeds
  GitHub's 100 MB file limit)
- `test.jsonl.gz`: debugging and baseline comparisons
- `validation_unseen/validation.jsonl.gz`: one final evaluation only
- `split_manifest.json`: counts, seed, checksums, source, and license

The split is deterministic with seed 42 and is stratified by question type and
difficulty. Do not open or load the validation file during development.

The raw source files and the oversized combined training file are ignored by
Git. The two training shards, test file, and sealed validation file are included
in this repository.

To regenerate after downloading the two official distractor training Parquet
shards into `data/source/`:

```bash
python3 -m pip install -r requirements-data.txt
python3 scripts/split_hotpotqa.py data/source data/hotpotqa_splits
```

Official dataset: <https://hotpotqa.github.io/>

Hugging Face viewer: <https://huggingface.co/datasets/hotpotqa/hotpot_qa>
