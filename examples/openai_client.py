import os

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("DGX_OPENAI_BASE_URL", "http://dgx-spark.local:3000/v1"),
    api_key=os.environ["DGX_OPENAI_API_KEY"],
)

for chunk in client.chat.completions.create(
    model=os.getenv("DGX_MODEL", "qwen3.8-27b"),
    messages=[{"role": "user", "content": "Describe the DGX Spark in one sentence."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
