ARG BASE_IMAGE=lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
FROM ${BASE_IMAGE} AS builder

ARG SSD_STREAM_COMMIT=5aeffa316e111d77aa6032a9956e4bcd4b752d9d
ARG SSD_STREAM_SOURCE_SHA256=a3fdfe991fbb1c1df768edc6042ff07ba2422734f77d034c7a17ac0e8d184ff4
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

ARG SSD_STREAM_COMMIT=5aeffa316e111d77aa6032a9956e4bcd4b752d9d
ARG SGLANG_COMMIT=0a79825b7baa3e2aafd54e89097a5aba83d00b4e
ARG SGLANG_SOURCE_SHA256=85bfd9f311498435c66db08ace10ec7bc7380738d689d412bfc5fb851c93a3ac
ARG SSD_STREAM_STAGING_MIB=32

LABEL org.opencontainers.image.title="SGLang SSD Stream for DGX Spark" \
      org.opencontainers.image.base.name="lmsysorg/sglang:qwen38flashnext" \
      org.opencontainers.image.source="https://github.com/garnermccloud/sglang-ssd-stream" \
      org.opencontainers.image.revision="${SSD_STREAM_COMMIT}"

COPY --from=builder /wheels /tmp/wheels

RUN set -eux; \
    curl --fail --location --retry 5 --retry-all-errors \
      "https://github.com/sgl-project/sglang/archive/${SGLANG_COMMIT}.tar.gz" \
      -o /tmp/sglang.tar.gz; \
    echo "${SGLANG_SOURCE_SHA256}  /tmp/sglang.tar.gz" \
      | sha256sum --check --strict; \
    mkdir -p /tmp/sglang; \
    tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /tmp/sglang; \
    cd /; \
    rm -rf /sgl-workspace/sglang; \
    mv /tmp/sglang /sgl-workspace/sglang; \
    SGLANG_BUILD_RUST_EXTS=none python3 -m pip install --no-cache-dir --no-deps \
      --editable /sgl-workspace/sglang/python; \
    python3 -m pip install --no-cache-dir --no-deps \
      /tmp/wheels/sglang_ssd_stream-*.whl; \
    sed -i "s/^_STAGING_MIB = 16$/_STAGING_MIB = ${SSD_STREAM_STAGING_MIB}/" \
      /usr/local/lib/python3.12/dist-packages/sglang_ssd_stream/backend.py; \
    find /usr/local/lib/python3.12/dist-packages/sglang_ssd_stream \
      -path '*/__pycache__/backend*.pyc' -delete; \
    grep -Fx "_STAGING_MIB = ${SSD_STREAM_STAGING_MIB}" \
      /usr/local/lib/python3.12/dist-packages/sglang_ssd_stream/backend.py; \
    python3 -c "import importlib.metadata as m; from sglang_ssd_stream.plugin import register; assert m.version('nvidia-modelopt') == '0.45.0'; assert m.version('sglang-ssd-stream') == '0.1.0'; register(); print('ssd_stream_plugin=ok sglang=' + m.version('sglang'))"; \
    rm -rf /tmp/wheels /tmp/sglang.tar.gz
