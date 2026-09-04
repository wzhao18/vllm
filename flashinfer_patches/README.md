# FlashInfer patches

`4931-skip-trtllm-ragged-active-row-check.patch` contains the runtime changes
from [FlashInfer PR #4931](https://github.com/flashinfer-ai/flashinfer/pull/4931),
based on `1dff49bcb36299546e81bd12c6e967e2b0e3578c` and ending at
`bb0661ba6691c2e7b197c96efe86cb38857c0c59`. The upstream test-only diff is
excluded because installed wheels do not contain the FlashInfer test tree.

To patch the repository's installed FlashInfer package, run from the vLLM
repository root:

```bash
git apply --unsafe-paths \
  --directory=.venv/lib/python3.12/site-packages \
  flashinfer_patches/4931-skip-trtllm-ragged-active-row-check.patch
```

Verify that the patch is present:

```bash
git apply --reverse --check --unsafe-paths \
  --directory=.venv/lib/python3.12/site-packages \
  flashinfer_patches/4931-skip-trtllm-ragged-active-row-check.patch
```
