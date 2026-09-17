# Managed chat templates

`qwen_fixed_v22_4.jinja` is vendored from
[`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
at commit `756cfb69d742355fd310b4ba9d50815a27d9d241`.

- Upstream version: `qwen3.8-froggeric-v22.4`
- Upstream file: `chat_template.jinja`
- SHA-256: `c47c82b0544752d454f4e427228d9d9d8c3df64c9e446cbd0229362f67948009`
- Upstream license: Apache-2.0

The Manager copies this exact file into detected Qwen3.8 model directories as a
versioned sidecar. Runtime model mounts remain read-only; the sidecar is created
by the Manager before the inference container starts.
