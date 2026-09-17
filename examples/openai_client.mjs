import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: process.env.DGX_OPENAI_BASE_URL ?? 'http://dgx-spark.local:3000/v1',
  apiKey: process.env.DGX_OPENAI_API_KEY,
})

const stream = await client.chat.completions.create({
  model: process.env.DGX_MODEL ?? 'qwen3.8-27b',
  messages: [{ role: 'user', content: 'Describe the DGX Spark in one sentence.' }],
  stream: true,
})

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '')
}
process.stdout.write('\n')
