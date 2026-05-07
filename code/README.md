# CALIPER Code

**Hosted CALIPER Explorer:** https://caliper-artifact.github.io/

This folder contains the code needed to reproduce the CALIPER artifact pipeline.

## Folders

- `preprocessing/`: source dataset conversion and paraphrase generation.
- `inference_scoring/`: model inference, content-preservation scoring, task-quality scoring, and merge utilities.
- `analysis/`: aggregation scripts, table generation, figure generation, and score checks.
- `explorer/`: static browser explorer used by the hosted CALIPER Explorer.

The code expects local paths to be supplied on the command line. API credentials are read from environment variables such as `GOOGLE_API_KEY` and `HF_TOKEN`; no credentials are included in this artifact.

## Quick Checks

```bash
python -m compileall inference_scoring/src analysis/src
```

Rust utilities can be checked from their folders:

```bash
cd inference_scoring && cargo check
cd ../analysis && cargo check
cd ../preprocessing/rephras && cargo check
```

Some scripts require large model downloads or external API access and are intended as reproducibility scripts rather than lightweight unit tests.
