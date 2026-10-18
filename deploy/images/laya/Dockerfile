# Reuse the installed Spark-compatible Torch/CUDA stack; no second CUDA download.
FROM lmsysorg/sglang:v0.5.18-cu130-runtime
ENV USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
COPY laya-0.3.4-py3-none-any.whl /tmp/laya.whl
RUN mv /tmp/laya.whl /tmp/laya-0.3.4-py3-none-any.whl \
    && python3 -m pip install --no-deps /tmp/laya-0.3.4-py3-none-any.whl \
    && rm /tmp/laya-0.3.4-py3-none-any.whl
COPY server.py /opt/laya/server.py
ENTRYPOINT ["python3", "/opt/laya/server.py"]
