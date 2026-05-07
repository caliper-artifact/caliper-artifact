# CALIPER Explorer

Hosted site: https://caliper-artifact.github.io/

This folder contains the static browser explorer used by the hosted CALIPER artifact website. From the repository root, run:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/code/explorer/
```

The explorer reads the lightweight `samples/` files and `site_data/high_performing_examples.json`.
