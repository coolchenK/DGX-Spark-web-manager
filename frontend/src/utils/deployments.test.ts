import { describe, expect, it } from 'vitest'

import type { Deployment, ModelAsset } from '../api/types'
import { deploymentToFormValues } from './deployments'


const deployment: Deployment = {
  id: 'deployment-1',
  name: 'qwen-production',
  model_id: 'model-1',
  runtime: 'vllm',
  container_id: 'container-1',
  container_name: 'dgx-qwen-production',
  endpoint_url: 'http://127.0.0.1:8100',
  api_model_name: 'qwen-instance-a',
  status: 'running',
  health: 'healthy',
  managed: true,
  image: 'vllm/vllm-openai:v0.27.1',
  port: 8100,
  config: {
    route_alias: 'qwen-production',
    spec: {
      context_length: 8192,
      memory_fraction: 0.42,
      max_concurrency: 4,
      max_batched_tokens: 4096,
      quantization: 'fp8',
      trust_remote_code: true,
    },
  },
  capabilities: ['chat', 'completion'],
  last_checked_at: null,
}

const model: ModelAsset = {
  id: 'model-1', name: 'Qwen/Qwen2.5', alias: null, source: 'huggingface',
  repository_id: 'Qwen/Qwen2.5', revision: 'main', commit_hash: 'abc',
  local_path: '/models/qwen', format: 'safetensors', quantization: null,
  parameter_count: '0.5B', size_bytes: 1, status: 'available',
  capabilities: ['chat'], created_at: '', updated_at: '',
}


describe('deploymentToFormValues', () => {
  it('restores saved settings for editing', () => {
    expect(deploymentToFormValues(deployment, model, 'edit')).toMatchObject({
      name: 'qwen-production',
      model_path: '/models/qwen',
      context_length: 8192,
      memory_fraction: 0.42,
      max_concurrency: 4,
      max_batched_tokens: 4096,
      quantization: 'fp8',
      trust_remote_code: true,
    })
  })

  it('creates unique names and ports when cloning', () => {
    expect(deploymentToFormValues(deployment, model, 'clone')).toMatchObject({
      name: 'qwen-production-copy',
      api_model_name: 'qwen-instance-a-copy',
      route_alias: 'qwen-production',
      port: 8101,
    })
  })
})
