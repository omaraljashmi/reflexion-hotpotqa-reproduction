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

## Run this for the screen recording

No API key or package installation is needed:

```bash
python3 run_reflexion.py
```

The command runs a baseline and a Reflexion condition, saves detailed records to
`results/latest.json`, and ends with `SUCCESS: run completed without errors`.

The default `scripted` provider deliberately fails its first attempts and then
follows the stored reflections. It is a deterministic **control-flow smoke test**
that makes the Actor → Evaluator → Reflection → Memory behavior easy to inspect.
It is not presented as a scientific LLM result.

Run the automated checks before recording:

```bash
python3 -m unittest discover -s tests -v
```

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
tests/test_reflexion.py                  automated regression tests
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
