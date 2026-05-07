---
license: other
pretty_name: CALIPER artifact
tags:
- prompt-robustness
- paraphrase
- benchmark
- evaluation
- llm-as-judge
- croissant
task_categories:
- text-generation
- question-answering
language:
- en
size_categories:
- 1M<n<10M
---

# CALIPER Artifact

**Interactive CALIPER Explorer:** https://caliper-artifact.github.io/

This repository contains the anonymous artifact release for CALIPER, a prompt-robustness dataset and benchmark built from Alpaca, GSM8K, and MMLU prompts. It includes the canonical CALIPER dataset files, along with code, metadata, samples, analysis outputs, figures, and Croissant/Responsible AI metadata.

## Canonical Dataset Layout

```text
prompts_paraphrases/
  alpaca_500.json
  finetuning_50k.json
  gsm8k_500.json
  mmlu_500.json
  scores_content_preservation.json
  tags_paraphrases.json

paraphrase_answers/
  alpaca/
  gsm8k/
  mmlu/

metric_scores/
  alpaca/
  gsm8k/
  mmlu/

samples/
  prompts_paraphrases/
  paraphrase_answers/
  metric_scores/
```

## Additional Artifact Files

```text
code/
  preprocessing/
  inference_scoring/
  analysis/
  explorer/

figures/
scores/
responses/
```

The OpenReview dataset URL should point to this repository root. The OpenReview code URL can point to `code/` in this same repository.

## Main Files

- `caliper_croissant_rai.json`: Croissant and Responsible AI metadata.
- `asset_licenses.md`: license and version manifest for source datasets, evaluated models, and generated artifacts.
- `prompts_paraphrases/`: canonical prompt/paraphrase files.
- `paraphrase_answers/`: model generations for Alpaca, GSM8K, and MMLU.
- `metric_scores/`: response-quality metric score files for the evaluated model generations.
- `samples/`: sample files from the canonical release for quick inspection.
- `code/`: scripts used for preprocessing, generation, scoring, analysis, and the browser-based explorer. The hosted explorer is available at https://caliper-artifact.github.io/.
- `code.zip`: zipped copy of the submitted code tree.

## Scoring Provenance

All reported automated content-preservation and task-performance scores used `gemini-2.5-flash-preview-05-20` as the judge model. Scoring was run in September 2025 with deterministic decoding, temperature 0, and fixed system instructions.

## Code

The code is grouped by role:

- `code/preprocessing/`: source dataset conversion and paraphrase generation.
- `code/inference_scoring/`: model inference, Gemini scoring, and result-merging utilities.
- `code/analysis/`: scripts for aggregation, tables, figures, and score checks.
- `code/explorer/`: static browser interface for exploring CALIPER scores.

The Python scripts were syntax-checked after staging. Rust manifests are included for the Rust utilities; reviewers using Rust 1.88 or newer should be able to resolve the current crate ecosystem directly, while the included manifests pin `time` for compatibility with Rust 1.87.

## Terms

This is a mixed-license artifact. CALIPER-authored code, metadata, and documentation use Apache-2.0. Redistributed or prompt-derived records preserve the upstream source dataset licenses and evaluated model-provider terms. See `asset_licenses.md` for exact license and version strings.
