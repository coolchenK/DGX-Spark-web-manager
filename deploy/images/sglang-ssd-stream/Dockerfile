ARG BASE_IMAGE=lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
FROM ${BASE_IMAGE} AS builder

ARG SSD_STREAM_COMMIT=60ce795c339f9d31a18a8057d0bbe59611e4e9c7
ARG SSD_STREAM_SOURCE_SHA256=0017a0fd8fa951e1971a83546c750551ce33118d41ee99e2f963f13b46b0821d
ARG RUST_TOOLCHAIN=1.92.0

ENV CARGO_BUILD_JOBS=2 \
    MAX_JOBS=2

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
      https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    python3 -m pip install --no-cache-dir 'maturin>=1.9,<2'; \
    mkdir -p /wheels

RUN set -eux; \
    curl --fail --location --retry 5 --retry-all-errors \
      "https://github.com/garnermccloud/sglang-ssd-stream/archive/${SSD_STREAM_COMMIT}.tar.gz" \
      -o /tmp/sglang-ssd-stream.tar.gz; \
    echo "${SSD_STREAM_SOURCE_SHA256}  /tmp/sglang-ssd-stream.tar.gz" \
      | sha256sum --check --strict; \
    mkdir -p /tmp/sglang-ssd-stream; \
    tar -xzf /tmp/sglang-ssd-stream.tar.gz \
      --strip-components=1 -C /tmp/sglang-ssd-stream; \
    cd /tmp/sglang-ssd-stream; \
    maturin build --release --locked --interpreter python3 \
      --out /wheels; \
    rm -rf /tmp/sglang-ssd-stream /tmp/sglang-ssd-stream.tar.gz

FROM ${BASE_IMAGE}

ARG SSD_STREAM_COMMIT=60ce795c339f9d31a18a8057d0bbe59611e4e9c7

LABEL org.opencontainers.image.title="SGLang SSD Stream for DGX Spark" \
      org.opencontainers.image.base.name="lmsysorg/sglang:qwen38flashnext" \
      org.opencontainers.image.source="https://github.com/garnermccloud/sglang-ssd-stream" \
      org.opencontainers.image.revision="${SSD_STREAM_COMMIT}"

COPY --from=builder /wheels /tmp/wheels
COPY patch_sm121_qsa.py /tmp/patch_sm121_qsa.py

RUN set -eux; \
    python3 -m pip install --no-cache-dir --no-deps \
      /tmp/wheels/sglang_ssd_stream-*.whl; \
    python3 /tmp/patch_sm121_qsa.py; \
    python3 -c "import importlib.metadata as m; from sglang_ssd_stream.plugin import register; assert m.version('sglang').startswith('0.0.0.dev1+gd91c3682b'); assert m.version('nvidia-modelopt') == '0.45.0'; register(); print('ssd_stream_plugin=ok')"; \
    rm -rf /tmp/wheels /tmp/patch_sm121_qsa.py
