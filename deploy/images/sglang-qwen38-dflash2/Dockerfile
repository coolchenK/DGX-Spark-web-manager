ARG BASE_IMAGE=lmsysorg/sglang:qwen38-27b
FROM ${BASE_IMAGE}

ARG SGLANG_COMMIT=61fa64ae7e4975d27c275e30739e1bd9fa4372dd
ARG DFLASH_MODEL_SHA256=ad1e7957ce59751266c9359bfe65f51668fd81716744e34f55f7efd42e08d0f6
ARG DFLASH_KERNEL_SHA256=21f11619531f493bbe9f71d465912ca707f2d722991832e446ff65aecb40528c
ARG DFLASH_UTILS_SHA256=61feff7656e2e4c157e472a1140ea9525200f5003f9bf0de639b50424e776018
ARG DFLASH_WORKER_SHA256=5fd1e3f16c5725e4d56ac327515b56cfd06de52f94b069332d43faad7d771482

LABEL org.opencontainers.image.title="SGLang Qwen3.8 DFlash2 for DGX Spark" \
      org.opencontainers.image.base.name="lmsysorg/sglang:qwen38-27b" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_COMMIT}"

COPY patch_sglang.py /tmp/patch_sglang.py

RUN set -eux; \
    base_url="https://raw.githubusercontent.com/sgl-project/sglang/${SGLANG_COMMIT}/python/sglang"; \
    curl --fail --location --retry 5 --retry-all-errors \
      "${base_url}/srt/models/dflash.py" -o /tmp/dflash-model.py; \
    curl --fail --location --retry 5 --retry-all-errors \
      "${base_url}/kernels/ops/speculative/dflash.py" -o /tmp/dflash-kernel.py; \
    curl --fail --location --retry 5 --retry-all-errors \
      "${base_url}/srt/speculative/dflash_utils.py" -o /tmp/dflash-utils.py; \
    curl --fail --location --retry 5 --retry-all-errors \
      "${base_url}/srt/speculative/dflash_worker_v2.py" -o /tmp/dflash-worker.py; \
    echo "${DFLASH_MODEL_SHA256}  /tmp/dflash-model.py" | sha256sum --check --strict; \
    echo "${DFLASH_KERNEL_SHA256}  /tmp/dflash-kernel.py" | sha256sum --check --strict; \
    echo "${DFLASH_UTILS_SHA256}  /tmp/dflash-utils.py" | sha256sum --check --strict; \
    echo "${DFLASH_WORKER_SHA256}  /tmp/dflash-worker.py" | sha256sum --check --strict; \
    python3 /tmp/patch_sglang.py /tmp/dflash-utils.py /tmp/dflash-worker.py; \
    install -m 0644 /tmp/dflash-model.py \
      /sgl-workspace/sglang/python/sglang/srt/models/dflash.py; \
    install -m 0644 /tmp/dflash-kernel.py \
      /sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py; \
    install -m 0644 /tmp/dflash-utils.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py; \
    install -m 0644 /tmp/dflash-worker.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py; \
    python3 -c "from sglang.kernels.ops.speculative.dflash import selector_walk_triton; from sglang.srt.models.dflash import DFlash2DraftModel; from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2"; \
    rm -f /tmp/dflash-model.py /tmp/dflash-kernel.py \
      /tmp/dflash-utils.py /tmp/dflash-worker.py /tmp/patch_sglang.py
