# Laya decision runtime for DGX Spark

This optional image serves `convaiinnovations/laya` through the official `laya==0.3.4`
SDK. The SGLang base image supplies the ARM64 PyTorch/CUDA stack; inference uses the
Laya SDK, not the SGLang server. It exposes `/health`, `/v1/models`, and `/v1/decisions`.

Build on the target Spark, from this directory:

```bash
python3 -m pip download --no-deps --only-binary=:all: laya==0.3.4
sha256sum laya-0.3.4-py3-none-any.whl
docker build -t dgx-local/laya:0.3.4-cu130 .
```

The expected wheel SHA-256 is
`8bf1c5cc5cbf6abebf63a6346b4babedd5692f88a75aa9e0cc01ed4145a9a14b`.
Model weights and the wheel are not committed. Download the full model, including
the `multilingual` and `typed-decisions` subdirectories, before creating a `laya`
deployment. Its adapter requires concurrency 1 and the 1024 context setting;
checkpoint-specific limits remain 512/1024. Sampling, MTP, and chat completions do
not apply. The SDK can truncate inputs to its checkpoint budget.

The request contract and SDK response forwarding are exercised by
`backend/tests/test_laya.py` without loading CUDA weights. A real deployment still
needs GPU validation and application-specific probability calibration. Use the
manager's legacy gateway key for `/v1/decisions`; Magpie's generic chat routing is
not an implementation of the Laya decision protocol.
