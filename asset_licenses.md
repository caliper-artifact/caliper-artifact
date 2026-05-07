# Asset Licenses And Versions

This file records the license and version identifiers used for the anonymous CALIPER artifact. It should be included with the artifact release.

## CALIPER Release

| Asset | License / terms | Version / revision | Notes |
|---|---|---|---|
| CALIPER code, metadata, analysis scripts, and newly written documentation | Apache-2.0 (`https://www.apache.org/licenses/LICENSE-2.0`) | `1.0.0` | Intended license for the new CALIPER-authored code and documentation. |
| CALIPER generated paraphrases, generated model responses, content-preservation scores, and metric-score artifacts | Mixed release: CALIPER-authored generated artifacts are released under Apache-2.0 where separable, but redistributed prompt-derived records preserve upstream source dataset terms and model-provider terms. | `1.0.0` | The complete data package should not be described as only Apache-2.0 because Alpaca is CC-BY-NC-4.0. |

## Source Datasets

| Source | License / terms | Version / revision used as source identifier | URL |
|---|---|---|---|
| Alpaca (`tatsu-lab/alpaca`) | CC-BY-NC-4.0 | Hugging Face revision `dce01c9b08f87459cf36a430d809084718273017`; last modified 2023-05-22 | `https://huggingface.co/datasets/tatsu-lab/alpaca` |
| GSM8K (`openai/gsm8k`, config `main`) | MIT | Hugging Face revision `740312add88f781978c0658806c59bc2815b9866`; last modified 2026-03-23 | `https://huggingface.co/datasets/openai/gsm8k` |
| MMLU (`cais/mmlu`, moral-scenarios subset in CALIPER) | MIT | Hugging Face revision `c30699e8356da336a370243923dbaf21066bb9fe`; last modified 2024-03-08 | `https://huggingface.co/datasets/cais/mmlu` |

The local code contains MMLU loading fallbacks for `hendrycks_test`, `lukaemon/mmlu`, and `cais/mmlu`. The public CALIPER artifact format matches the MMLU moral-scenarios records exposed by `cais/mmlu`; if the final artifact was generated from a different mirror, replace the MMLU revision above while preserving the source MMLU citation and license.

## Core Evaluated Models

| Model | License / terms | Version / revision | URL |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | Qwen Research License Agreement (`qwen-research`) | Hugging Face revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`; last modified 2024-09-25 | `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct` |
| `google/gemma-2-2b-it` | Gemma Terms of Use (`gemma`) | Hugging Face revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`; last modified 2024-08-27 | `https://huggingface.co/google/gemma-2-2b-it` |
| `google/gemma-2-9b-it` | Gemma Terms of Use (`gemma`) | Hugging Face revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`; last modified 2024-08-27 | `https://huggingface.co/google/gemma-2-9b-it` |

## Additional Models Mentioned In The Paper

| Model | License / terms | Version / revision | URL |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | Apache-2.0 | Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca`; last modified 2025-07-26 | `https://huggingface.co/Qwen/Qwen3-0.6B` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache-2.0 | Hugging Face revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; last modified 2024-03-17 | `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `microsoft/Phi-3-mini-4k-instruct` | MIT | Hugging Face revision `f39ac1d28e925b323eae81227eaba4464caced4e`; last modified 2025-12-10 | `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct` |
| `mistralai/Mistral-7B-Instruct-v0.3` | Apache-2.0 | Hugging Face revision `c170c708c41dac9275d15a8fff4eca08d52bab71`; last modified 2025-12-03 | `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3` |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community License Agreement (`llama3.1`) | Hugging Face revision `0e9e39f249a16976918f6564b8830bc894c89659`; last modified 2024-09-25 | `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct` |
| `internlm/internlm2-chat-7b` | Hugging Face tag `other`; model card states code is Apache-2.0 and model weights are open for academic research, with commercial licensing handled through InternLM's application process. | Hugging Face revision `c2ba64483dc50b3f8eb2d8271c4b9877a79ed2e2`; last modified 2025-03-13 | `https://huggingface.co/internlm/internlm2-chat-7b` |
| `tiiuae/falcon-7b-instruct` | Apache-2.0 | Hugging Face revision `8782b5c5d8c9290412416618f36a133653e85285`; last modified 2024-10-12 | `https://huggingface.co/tiiuae/falcon-7b-instruct` |
| `01-ai/Yi-1.5-9B-Chat` | Apache-2.0 | Hugging Face revision `1a0fc698cf883c4f5c325f026ca79f0ebd9955a5`; last modified 2024-06-26 | `https://huggingface.co/01-ai/Yi-1.5-9B-Chat` |

## Generated Outputs

There is no separate upstream license string for the generated model outputs themselves in the local repository. The valid release statement is therefore:

> Generated paraphrases, generated model responses, content-preservation scores, and response-quality scores are CALIPER-generated artifacts. They are released under the CALIPER artifact terms where separable, while preserving the licenses and terms of the source prompts and evaluated model providers.

In practice, this means Alpaca-derived records should be treated as non-commercial because Alpaca is CC-BY-NC-4.0; GSM8K/MMLU-derived records preserve MIT source terms; Gemma- and Qwen-generated response artifacts should preserve the respective Gemma Terms of Use and Qwen Research License terms.
