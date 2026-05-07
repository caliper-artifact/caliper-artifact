# Anonymous CALIPER Artifact Notes

This repository is the anonymous artifact release for the CALIPER submission.

The artifact includes:

- CALIPER prompt/paraphrase files.
- Content-preservation scores.
- Model responses for the evaluated model runs.
- Ten-metric response-quality scores.
- Style-to-tag mappings.
- Analysis scripts that reproduce the tables and figures.
- Croissant/RAI metadata in `caliper_croissant_rai.json`.
- Asset license and version manifest in `asset_licenses.md`.
- Static explorer data in `site_data/`.

The canonical dataset files are included under `prompts_paraphrases/`, `paraphrase_answers/`, `metric_scores/`, and `samples/`.

License and version details are recorded in `asset_licenses.md`. The artifact is a mixed-license release: new CALIPER code, metadata, and documentation use Apache-2.0, while redistributed or prompt-derived records preserve upstream source dataset licenses and evaluated model-provider terms.

Automated scoring provenance:

- All reported automated content-preservation and task-performance scores used `gemini-2.5-flash-preview-05-20` as the judge model.
- Scoring was run in September 2025 with deterministic decoding, temperature 0, and fixed system instructions.
