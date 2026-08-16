import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import { createElement, StrictMode, type ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type { DeploymentRecommendation, RecommendedValue } from '../api/types'
import { useDeploymentRecommendation } from '../hooks/useDeploymentRecommendation'
import {
  flattenChangedFields,
  valuesFromRecommendation,
} from './deploymentRecommendations'


function recommended<T>(
  value: T,
  source: RecommendedValue<T>['source'] = 'model_card',
): RecommendedValue<T> {
  return {
    value,
    source,
    confidence: 'high',
    reason: 'fixture recommendation',
    warning: null,
  }
}


function recommendationFixture(
  overrides: Partial<DeploymentRecommendation> = {},
): DeploymentRecommendation {
  return {
    status: 'complete',
    generated_at: '2026-08-16T00:00:00Z',
    model_id: 'model-1',
    runtime: 'vllm',
    image_digest: 'sha256:test',
    evidence_hash: 'a'.repeat(64),
    fields: {},
    generation_defaults: {},
    resource_snapshot: {
      total_bytes: 128,
      available_bytes: 96,
      reserved_bytes: 16,
    },
    resource_estimate: {
      total_bytes: 128,
      available_bytes: 96,
      reserved_bytes: 16,
      weight_bytes: 16,
      draft_weight_bytes: 0,
      kv_cache_bytes: 8,
      runtime_overhead_bytes: 4,
      required_bytes: 28,
      decision: 'ok',
      confidence: 'high',
      reasons: [],
    },
    runtime_capabilities: {
      runtime: 'vllm',
      image: 'vllm:test',
      image_digest: 'sha256:test',
      source: 'probe',
      generation_defaults: ['temperature'],
      quantization_methods: ['auto', 'modelopt_fp4'],
      quantization_mapping: { nvfp4: 'modelopt_fp4' },
      speculative_methods: ['draft_model'],
      method_mapping: { draft_model: 'draft_model' },
      speculative_transport: 'json',
      warnings: [],
    },
    draft_candidates: [],
    warnings: [],
    ...overrides,
  }
}


function wrapper(queryClient: QueryClient) {
  return ({ children }: { children: ReactNode }) => createElement(
    StrictMode,
    null,
    createElement(QueryClientProvider, { client: queryClient }, children),
  )
}


describe('deployment recommendation helpers', () => {
  it('applies only untouched recommendations unless force is true', () => {
    const recommendation = recommendationFixture({
      fields: {
        context_length: recommended(16_384, 'device_rule'),
        max_concurrency: recommended(4),
      },
      generation_defaults: {
        temperature: recommended(0.6),
        stop: recommended(['END']),
      },
    })

    expect(valuesFromRecommendation(
      recommendation,
      new Set(['context_length', 'generation_defaults.stop']),
      false,
    )).toEqual({
      max_concurrency: 4,
      generation_defaults: { temperature: 0.6 },
    })
    expect(valuesFromRecommendation(
      recommendation,
      new Set(['context_length']),
      true,
    )).toMatchObject({
      context_length: 16_384,
      generation_defaults: { temperature: 0.6, stop: ['END'] },
    })
  })

  it('recursively flattens sorted scalar and array leaves', () => {
    expect(flattenChangedFields({
      speculative: { method: 'eagle3', reasons: ['compatible'] },
      generation_defaults: { temperature: 0 },
      resource_warning_acknowledged: false,
      ignored_empty_group: {},
    })).toEqual([
      'generation_defaults.temperature',
      'resource_warning_acknowledged',
      'speculative.method',
      'speculative.reasons',
    ])
  })
})


describe('recommendation client and hook', () => {
  it('passes RequestInit and AbortSignal through POST without breaking JSON bodies', async () => {
    const signal = new AbortController().signal
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({ ok: true }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    ))

    await api.post('/api/example', { value: 0 }, { signal, headers: { 'X-Test': 'yes' } })

    const [, options] = fetchSpy.mock.calls[0]
    expect(options).toMatchObject({ method: 'POST', body: JSON.stringify({ value: 0 }), signal })
    expect(new Headers(options?.headers).get('X-Test')).toBe('yes')
  })

  it('debounces a normalized tuple and passes the query cancellation signal', async () => {
    const response = recommendationFixture()
    const post = vi.spyOn(api, 'post').mockResolvedValue(response)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1',
        runtime: 'vllm',
        image: 'vllm:test',
        providerId: undefined,
        enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    expect(post).not.toHaveBeenCalled()
    await waitFor(() => expect(result.current.data).toEqual(response))

    expect(post).toHaveBeenCalledOnce()
    expect(post.mock.calls[0][0]).toBe('/api/deployments/recommendations')
    expect(post.mock.calls[0][1]).toEqual({
      model_id: 'model-1',
      runtime: 'vllm',
      image: 'vllm:test',
      provider_id: null,
    })
    expect(post.mock.calls[0][2]?.signal).toBeInstanceOf(AbortSignal)
  })

  it('cancels an old tuple immediately and ignores its late response', async () => {
    const pending: Array<{
      signal: AbortSignal
      resolve: (value: DeploymentRecommendation) => void
    }> = []
    vi.spyOn(api, 'post').mockImplementation((_path, _body, options) => new Promise((resolve) => {
      pending.push({ signal: options?.signal as AbortSignal, resolve })
    }))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let input = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', enabled: true,
    }
    const { rerender, result } = renderHook(
      () => useDeploymentRecommendation(input),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(pending).toHaveLength(1))
    input = { ...input, modelId: 'model-2' }
    rerender()
    expect(pending[0].signal.aborted).toBe(true)

    await act(async () => {
      pending[0].resolve(recommendationFixture({ model_id: 'model-1' }))
    })
    await waitFor(() => expect(pending).toHaveLength(2))
    await act(async () => {
      pending[1].resolve(recommendationFixture({ model_id: 'model-2' }))
    })
    await waitFor(() => expect(result.current.data?.model_id).toBe('model-2'))
  })

  it('cancels an in-flight query when recommendations are disabled', async () => {
    let requestSignal: AbortSignal | null | undefined
    vi.spyOn(api, 'post').mockImplementation((_path, _body, options) => {
      requestSignal = options?.signal
      return new Promise(() => {})
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let enabled = true
    const { rerender } = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(requestSignal).toBeInstanceOf(AbortSignal))
    enabled = false
    rerender()

    expect(requestSignal?.aborted).toBe(true)
  })

  it('refreshes AI once and stores the result under the ordinary base key', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const refreshed = recommendationFixture({ status: 'complete' })
    const post = vi.spyOn(api, 'post').mockImplementation((path) => Promise.resolve(
      path.includes('refresh_ai=true') ? refreshed : ordinary,
    ))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const input = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', enabled: true,
    }
    const { result, unmount } = renderHook(
      () => useDeploymentRecommendation(input),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(ordinary))
    await act(async () => { await result.current.refreshAI() })
    await waitFor(() => expect(result.current.data).toEqual(refreshed))
    expect(post.mock.calls.map(([path]) => path)).toEqual([
      '/api/deployments/recommendations',
      '/api/deployments/recommendations?refresh_ai=true',
    ])

    unmount()
    queryClient.removeQueries({ queryKey: ['deployment-recommendation'] })
    renderHook(
      () => useDeploymentRecommendation(input),
      { wrapper: wrapper(queryClient) },
    )
    await waitFor(() => expect(post).toHaveBeenCalledTimes(3))
    expect(post.mock.calls[2][0]).toBe('/api/deployments/recommendations')
  })
})
