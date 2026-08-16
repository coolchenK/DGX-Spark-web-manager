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
      generation_defaults: {
        temperature: 0.6,
        top_p: 0.95,
        stop: ['END'],
      },
      speculative: {
        draft_model_id: 'draft-1',
        method: 'eagle3',
        num_steps: 2,
        eagle_top_k: 4,
        num_draft_tokens: 16,
        manual_review_acknowledged: true,
      },
      recommendation: {
        generated_at: '2026-08-16T00:00:00Z',
        evidence_hash: 'a'.repeat(64),
        provider_id: 'provider-1',
        resource_snapshot: {
          total_bytes: 1000,
          available_bytes: 800,
          reserved_bytes: 200,
        },
        modified_fields: ['generation_defaults.temperature'],
        sources: { 'generation_defaults.temperature': 'model_card' },
      },
      resource_warning_acknowledged: true,
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
      generation_defaults: {
        temperature: 0.6,
        top_p: 0.95,
        stop: ['END'],
      },
      speculative: {
        draft_model_id: 'draft-1',
        method: 'eagle3',
        manual_review_acknowledged: true,
      },
      resource_warning_acknowledged: true,
    })
  })

  it('creates unique names and ports when cloning', () => {
    expect(deploymentToFormValues(deployment, model, 'clone')).toMatchObject({
      name: 'qwen-production-copy',
      api_model_name: 'qwen-instance-a-copy',
      route_alias: 'qwen-production',
      port: 8101,
      speculative: {
        draft_model_id: 'draft-1',
        method: 'eagle3',
        manual_review_acknowledged: false,
      },
      resource_warning_acknowledged: false,
    })
  })

  it('restores saved recommendation provenance when editing', () => {
    expect(deploymentToFormValues(deployment, model, 'edit').recommendation).toMatchObject({
      provider_id: 'provider-1',
      modified_fields: ['generation_defaults.temperature'],
      sources: { 'generation_defaults.temperature': 'model_card' },
    })
  })

  it('omits Pydantic null defaults while preserving valid falsy values', () => {
    const pydanticDeployment: Deployment = {
      ...deployment,
      config: {
        spec: {
          ...(deployment.config.spec as Record<string, unknown>),
          generation_defaults: {
            temperature: 0,
            top_p: null,
            top_k: null,
            min_p: null,
            repetition_penalty: null,
            presence_penalty: null,
            frequency_penalty: null,
            max_tokens: null,
            stop: [],
          },
          speculative: {
            draft_model_id: 'draft-1',
            method: 'eagle3',
            num_speculative_tokens: null,
            num_steps: null,
            eagle_top_k: null,
            num_draft_tokens: null,
            manual_review_acknowledged: true,
          },
          resource_warning_acknowledged: false,
        },
      },
    }

    const edited = deploymentToFormValues(pydanticDeployment, model, 'edit')
    expect(edited.generation_defaults).toEqual({ temperature: 0, stop: [] })
    expect(edited.speculative).toEqual({
      draft_model_id: 'draft-1',
      method: 'eagle3',
      manual_review_acknowledged: true,
    })
    expect(edited.resource_warning_acknowledged).toBe(false)

    const cloned = deploymentToFormValues(pydanticDeployment, model, 'clone')
    expect(cloned.speculative).toEqual({
      draft_model_id: 'draft-1',
      method: 'eagle3',
      manual_review_acknowledged: false,
    })
  })
})
