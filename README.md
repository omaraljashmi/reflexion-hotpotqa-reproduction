# Reflexion: HotPotQA Mini Reproduction

This repository reproduces the central algorithm from **Reflexion: Language
Agents with Verbal Reinforcement Learning** (Shinn et al., NeurIPS 2023):

1. an **Actor** attempts a question;
2. an **Evaluator** returns exact-match feedback;
3. after a failure, a **Self-Reflection** component writes a verbal lesson;
4. the lesson is stored in **episodic memory** and supplied to the next trial.

The implementation uses a five-question subset of the authors' released
HotPotQA distractor sample. It reports HotPotQA-style exact match (EM), token F1,
and cumulative accuracy after each trial.


## Run a real LLM experiment

The live path uses the current OpenAI Responses API:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python run_reflexion.py --provider openai --limit 5 --trials 3
```

Use `--model MODEL_ID` or the `OPENAI_MODEL` environment variable to choose a
model available to your account. API calls may cost money. Never expose the API
key in a recording or commit it to GitHub.

Useful options:

```text
--mode compare|baseline|reflexion
--provider scripted|openai
--limit N
--trials N
--data PATH
--output PATH
```

## Extension: cross-task structured failure memory

Reflexion reuses a reflection only on the same task. The extension tests
whether lessons from past failures transfer to **different, unseen** questions.

1. **Build** (training split): each question gets one attempt. Each failure is
   labeled with a failure type (rules first, then the LLM), and the LLM writes a
   lesson in two separate parts: a task-specific *diagnosis* and a general *fix*
   that may not mention the question's entities or answer.
2. **Validate**: a fresh attempt sees only the fix. The lesson is kept only if
   that attempt now passes. This shows the fix helped, not that the diagnosis
   is true.
3. **Evaluate** (test split): before the first attempt on each new question,
   the agent retrieves up to `k` lessons covering the failure types most common
   for that question type (bridge or comparison).

The correct answer never appears in any model prompt. It is used only for
scoring and for the rule-based failure labels.

| Condition | Within-task reflection | Cross-task lessons |
|---|---|---|
| `no_memory` | no | none |
| `reflexion` (parent paper) | yes | none |
| `failure_memory` (ours) | yes | by failure type |
| `random_memory` (control) | yes | random, same count |
| `similarity_memory` (optional) | yes | by question similarity |

The main metric is **first-attempt exact match** on test questions, since
cross-task lessons are the only thing that can change a first attempt.
`random_memory` checks whether any extra advice helps, not the retrieval method.

```bash
export GROQ_API_KEY="your-key"           # console.groq.com
python run_experiment.py pilot    --provider groq --n 30     # target 40-75% first-attempt EM
python run_experiment.py build    --provider groq --n 150    # -> results/memory.json
python run_experiment.py evaluate --provider groq --n 100 --trials 2
python run_experiment.py analyze  --attempts results/eval_attempts.csv   # no API needed
```

Every model attempt is saved as a row in a CSV (`results/*_attempts.csv`): the
condition, question type, prediction, exact match, F1, failure type, lessons
used, the reflection, and token counts. `analyze` reports EM per condition,
paired fail-to-pass and pass-to-fail flips with an exact McNemar test, failure
type counts, how often each lesson helped or hurt, and repeated mistakes.

Default model: `openai/gpt-oss-20b` on Groq (reasoning effort `low`, reasoning
text hidden, temperature 0). Use `--model` to change it. Run with
`--provider scripted` to test the full pipeline offline. Scripted numbers are
not results. The sealed validation split can only be loaded with
`--final-validation`. A Colab version is in
`notebooks/failure_memory_experiment.ipynb`.

## Full dataset split

The official HotpotQA distractor training pool contains 90,447 labeled
questions. It is divided with seed 42 and stratified by question type and
difficulty:

- Training: 54,268 questions (60%) in two compressed JSONL shards
- Testing: 18,089 questions (20%)
- Validation: 18,090 questions (20%)

The files are in `data/hotpotqa_splits/`. Keep
`validation_unseen/validation.jsonl.gz` sealed until development and testing are
complete. See `split_manifest.json` for the source links, counts, and SHA-256
checksums. Run `scripts/split_hotpotqa.py` to reproduce the split from the two
official Parquet source shards.

## What matches the parent paper

- repeated trials on HotPotQA question-answering tasks;
- context-grounded chain-of-thought Actor prompts;
- exact-match environment feedback;
- language-model-generated reflection after unsuccessful trials;
- task-local verbal memory injected into later Actor prompts;
- baseline and Reflexion experimental conditions.

## What is intentionally smaller or modernized

- Five released sample items are included instead of the paper's 100-question
  experimental sample, keeping a classroom run quick and inexpensive.
- The live provider uses the supported Responses API instead of the authors'
  pinned 2023 `openai==0.27.4`/`langchain==0.0.162` stack.
- The offline provider is solely a reliable demonstration and test fixture.
- This code does not claim to reproduce the paper's exact headline scores. Model
  versions, sampling, API behavior, and the smaller sample differ.

## Repository map

```text
run_reflexion.py                         recording-friendly entry point
src/reflexion_reproduction/core.py      Actor/evaluator/reflection/memory loop
src/reflexion_reproduction/providers.py offline and live model providers
src/reflexion_reproduction/prompts.py   Actor and reflection prompts
src/reflexion_reproduction/metrics.py   exact match and token F1
data/hotpotqa_mini.json                  five released HotPotQA sample items
data/hotpotqa_splits/                    full 60/20/20 split and manifest
scripts/split_hotpotqa.py                deterministic dataset splitting tool
tests/test_reflexion.py                  automated regression tests
run_experiment.py                        failure-memory experiment entry point
src/reflexion_reproduction/data.py      split loader (validation sealed)
src/reflexion_reproduction/failure_memory.py  labels, lessons, validation, retrieval
src/reflexion_reproduction/pipeline.py  conditions, CSV logging, analysis
tests/test_failure_memory.py             tests for the extension
notebooks/failure_memory_experiment.ipynb  Colab walkthrough
results/                                 generated detailed trial records
```

## Suggested recording sequence

1. Show this public GitHub repository and its file list.
2. Open `src/reflexion_reproduction/core.py` and point out `run_trial`.
3. Open `src/reflexion_reproduction/providers.py` and identify the two providers.
4. In a terminal at the repository root, run the unit-test command above.
5. Run `python3 run_reflexion.py`.
6. Keep the final comparison and green/successful terminal exit visible.
7. Open `results/latest.json` to show that trial traces and reflections were saved.

## Sources and attribution

- Paper: [Shinn et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)
- Authors' code and released sample: [noahshinn/reflexion](https://github.com/noahshinn/reflexion)
- Dataset paper: [HotPotQA](https://arxiv.org/abs/1809.09600)
- Current live-client pattern: [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)

The reproduction code is MIT licensed. Please retain the citations above when
sharing the project.
