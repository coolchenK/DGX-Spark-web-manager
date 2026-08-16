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
      deployments: [{
        id: 'deployment-1',
        runtime: 'vllm',
        status: 'running',
        memory_bytes: 24,
      }],
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


function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
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

  it('copies recommended stop arrays before handing values to the form', () => {
    const cachedStop = ['END']
    const recommendation = recommendationFixture({
      generation_defaults: { stop: recommended(cachedStop) },
    })

    const values = valuesFromRecommendation(recommendation, new Set(), false)
    const formStop = values.generation_defaults?.stop as string[]
    formStop.push('FORM-ONLY')

    expect(cachedStop).toEqual(['END'])
    expect(recommendation.generation_defaults.stop.value).toEqual(['END'])
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

  it('does not restart the debounce when provider id changes from undefined to null', async () => {
    vi.useFakeTimers()
    try {
      const post = vi.spyOn(api, 'post').mockResolvedValue(recommendationFixture())
      const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      let providerId: string | null | undefined
      const rendered = renderHook(
        () => useDeploymentRecommendation({
          modelId: 'model-1',
          runtime: 'vllm',
          image: 'vllm:test',
          providerId,
          enabled: true,
        }),
        { wrapper: wrapper(queryClient) },
      )

      await act(async () => { await vi.advanceTimersByTimeAsync(151) })
      providerId = null
      rendered.rerender()
      await act(async () => { await vi.advanceTimersByTimeAsync(150) })

      expect(post).toHaveBeenCalledOnce()
      expect(post.mock.calls[0][1]).toMatchObject({ provider_id: null })
      rendered.unmount()
    } finally {
      vi.useRealTimers()
    }
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
    expect(result.current.data).toBeUndefined()

    await act(async () => {
      pending[0].resolve(recommendationFixture({ model_id: 'model-1' }))
    })
    await waitFor(() => expect(pending).toHaveLength(2))
    await act(async () => {
      pending[1].resolve(recommendationFixture({ model_id: 'model-2' }))
    })
    await waitFor(() => expect(result.current.data?.model_id).toBe('model-2'))
  })

  it('debounces every A-B-A activation without exposing cached A data', async () => {
    const responseA = recommendationFixture({ model_id: 'model-a' })
    const responseB = recommendationFixture({ model_id: 'model-b' })
    const post = vi.spyOn(api, 'post').mockImplementation((_path, body) => Promise.resolve(
      (body as { model_id: string }).model_id === 'model-a' ? responseA : responseB,
    ))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let modelId = 'model-a'
    const { rerender, result } = renderHook(
      () => useDeploymentRecommendation({
        modelId, runtime: 'vllm', image: 'vllm:test', enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(responseA))
    modelId = 'model-b'
    rerender()
    expect(result.current.data).toBeUndefined()
    modelId = 'model-a'
    rerender()

    expect(result.current.data).toBeUndefined()
    expect(post).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(result.current.data).toEqual(responseA))
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

  it('waits for a new debounce interval when the same tuple is re-enabled', async () => {
    const response = recommendationFixture()
    vi.spyOn(api, 'post').mockResolvedValue(response)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let enabled = true
    const { rerender, result } = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(response))
    const callsBeforeDisable = vi.mocked(api.post).mock.calls.length
    enabled = false
    rerender()
    expect(result.current.data).toBeUndefined()
    await expect(result.current.refreshAI()).rejects.toThrow('Recommendation tuple is not stable')
    expect(api.post).toHaveBeenCalledTimes(callsBeforeDisable)
    enabled = true
    rerender()

    expect(result.current.data).toBeUndefined()
    await waitFor(() => expect(result.current.data).toEqual(response))
  })

  it('aborts on unmount and does not cache a late response', async () => {
    let signal: AbortSignal | null | undefined
    let resolveRequest: ((value: DeploymentRecommendation) => void) | undefined
    vi.spyOn(api, 'post').mockImplementation((_path, _body, options) => {
      signal = options?.signal
      return new Promise((resolve) => { resolveRequest = resolve })
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const tuple = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', providerId: null,
    }
    const { unmount } = renderHook(
      () => useDeploymentRecommendation({ ...tuple, enabled: true }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(signal).toBeInstanceOf(AbortSignal))
    unmount()
    expect(signal?.aborted).toBe(true)
    await act(async () => { resolveRequest?.(recommendationFixture()) })

    expect(queryClient.getQueryData(['deployment-recommendation', tuple])).toBeUndefined()
  })

  it('does not abort a shared exact query while another consumer remains active', async () => {
    let signal: AbortSignal | null | undefined
    vi.spyOn(api, 'post').mockImplementation((_path, _body, options) => {
      signal = options?.signal
      return new Promise(() => {})
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let firstEnabled = true
    const first = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled: firstEnabled,
      }),
      { wrapper: wrapper(queryClient) },
    )
    const second = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(signal).toBeInstanceOf(AbortSignal))
    firstEnabled = false
    first.rerender()
    expect(signal?.aborted).toBe(false)

    second.unmount()
    expect(signal?.aborted).toBe(true)
  })

  it('rejects AI refresh until an enabled tuple has completed its debounce', async () => {
    const post = vi.spyOn(api, 'post').mockResolvedValue(recommendationFixture())
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await expect(result.current.refreshAI()).rejects.toThrow('Recommendation tuple is not stable')
    expect(post).not.toHaveBeenCalled()
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

  it('lets only the latest concurrent AI refresh update the base query', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const firstResult = recommendationFixture({ generated_at: '2026-08-16T01:00:00Z' })
    const secondResult = recommendationFixture({ generated_at: '2026-08-16T02:00:00Z' })
    const first = deferred<DeploymentRecommendation>()
    const second = deferred<DeploymentRecommendation>()
    const refreshSignals: Array<AbortSignal | null | undefined> = []
    let refreshCalls = 0
    vi.spyOn(api, 'post').mockImplementation((path, _body, options) => {
      if (!path.includes('refresh_ai=true')) return Promise.resolve(ordinary)
      refreshSignals.push(options?.signal)
      return (++refreshCalls === 1 ? first : second).promise
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(
      () => useDeploymentRecommendation({
        modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(ordinary))
    const firstRefresh = result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(1))
    const secondRefresh = result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(2))
    expect(refreshSignals[0]?.aborted).toBe(true)

    second.resolve(secondResult)
    await expect(secondRefresh).resolves.toEqual(secondResult)
    first.resolve(firstResult)
    await expect(firstRefresh).rejects.toMatchObject({ name: 'AbortError' })
    await waitFor(() => expect(result.current.data).toEqual(secondResult))
  })

  it('shares refresh ownership across hooks without old-hook cleanup aborting the winner', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const firstResult = recommendationFixture({ generated_at: '2026-08-16T06:00:00Z' })
    const secondResult = recommendationFixture({ generated_at: '2026-08-16T07:00:00Z' })
    const first = deferred<DeploymentRecommendation>()
    const second = deferred<DeploymentRecommendation>()
    const refreshSignals: Array<AbortSignal | null | undefined> = []
    let refreshCalls = 0
    vi.spyOn(api, 'post').mockImplementation((path, _body, options) => {
      if (!path.includes('refresh_ai=true')) return Promise.resolve(ordinary)
      refreshSignals.push(options?.signal)
      return (++refreshCalls === 1 ? first : second).promise
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const options = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', enabled: true,
    }
    const hook1 = renderHook(
      () => useDeploymentRecommendation(options),
      { wrapper: wrapper(queryClient) },
    )
    const hook2 = renderHook(
      () => useDeploymentRecommendation(options),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(hook1.result.current.data).toEqual(ordinary))
    await waitFor(() => expect(hook2.result.current.data).toEqual(ordinary))
    const firstRefresh = hook1.result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(1))
    const secondRefresh = hook2.result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(2))
    expect(refreshSignals[0]?.aborted).toBe(true)

    hook1.unmount()
    expect(refreshSignals[1]?.aborted).toBe(false)
    second.resolve(secondResult)
    await expect(secondRefresh).resolves.toEqual(secondResult)
    first.resolve(firstResult)
    await expect(firstRefresh).rejects.toMatchObject({ name: 'AbortError' })
    expect(queryClient.getQueryData([
      'deployment-recommendation',
      { modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', providerId: null },
    ])).toEqual(secondResult)
  })

  it('allows either hook to become the latest shared refresh owner', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const first = deferred<DeploymentRecommendation>()
    const second = deferred<DeploymentRecommendation>()
    const third = deferred<DeploymentRecommendation>()
    const thirdResult = recommendationFixture({ generated_at: '2026-08-16T08:00:00Z' })
    const pending = [first, second, third]
    const refreshSignals: Array<AbortSignal | null | undefined> = []
    let refreshCalls = 0
    vi.spyOn(api, 'post').mockImplementation((path, _body, options) => {
      if (!path.includes('refresh_ai=true')) return Promise.resolve(ordinary)
      refreshSignals.push(options?.signal)
      const current = pending[refreshCalls]
      refreshCalls += 1
      return current.promise
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const options = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', enabled: true,
    }
    const hook1 = renderHook(
      () => useDeploymentRecommendation(options),
      { wrapper: wrapper(queryClient) },
    )
    const hook2 = renderHook(
      () => useDeploymentRecommendation(options),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(hook1.result.current.data).toEqual(ordinary))
    const refresh1 = hook1.result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(1))
    const refresh2 = hook2.result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(2))
    const refresh3 = hook1.result.current.refreshAI()
    await waitFor(() => expect(refreshCalls).toBe(3))
    expect(refreshSignals[0]?.aborted).toBe(true)
    expect(refreshSignals[1]?.aborted).toBe(true)
    expect(refreshSignals[2]?.aborted).toBe(false)

    hook2.unmount()
    expect(refreshSignals[2]?.aborted).toBe(false)
    third.resolve(thirdResult)
    await expect(refresh3).resolves.toEqual(thirdResult)
    second.resolve(recommendationFixture({ generated_at: '2026-08-16T09:00:00Z' }))
    first.resolve(recommendationFixture({ generated_at: '2026-08-16T10:00:00Z' }))
    await expect(refresh2).rejects.toMatchObject({ name: 'AbortError' })
    await expect(refresh1).rejects.toMatchObject({ name: 'AbortError' })
    expect(queryClient.getQueryData([
      'deployment-recommendation',
      { modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', providerId: null },
    ])).toEqual(thirdResult)
  })

  it.each(['disable', 'unmount', 'tuple-change'] as const)(
    'invalidates an AI refresh on %s even when transport ignores abort',
    async (transition) => {
      const ordinary = recommendationFixture({ status: 'partial' })
      const late = recommendationFixture({ generated_at: '2026-08-16T03:00:00Z' })
      const refresh = deferred<DeploymentRecommendation>()
      let refreshSignal: AbortSignal | null | undefined
      vi.spyOn(api, 'post').mockImplementation((path, _body, options) => {
        if (!path.includes('refresh_ai=true')) return Promise.resolve(ordinary)
        refreshSignal = options?.signal
        return refresh.promise
      })
      const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      let enabled = true
      let modelId = 'model-1'
      const rendered = renderHook(
        () => useDeploymentRecommendation({
          modelId, runtime: 'vllm', image: 'vllm:test', enabled,
        }),
        { wrapper: wrapper(queryClient) },
      )

      await waitFor(() => expect(rendered.result.current.data).toEqual(ordinary))
      const refreshPromise = rendered.result.current.refreshAI()
      await waitFor(() => expect(refreshSignal).toBeInstanceOf(AbortSignal))
      if (transition === 'disable') {
        enabled = false
        rendered.rerender()
      } else if (transition === 'tuple-change') {
        modelId = 'model-2'
        rendered.rerender()
      } else {
        rendered.unmount()
      }
      expect(refreshSignal?.aborted).toBe(true)

      refresh.resolve(late)
      await expect(refreshPromise).rejects.toMatchObject({ name: 'AbortError' })
      expect(queryClient.getQueryData([
        'deployment-recommendation',
        { modelId: 'model-1', runtime: 'vllm', image: 'vllm:test', providerId: null },
      ])).toEqual(ordinary)
    },
  )

  it('rejects a saved refresh callback after its tuple is no longer current', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const post = vi.spyOn(api, 'post').mockResolvedValue(ordinary)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    let modelId = 'model-1'
    const { rerender, result } = renderHook(
      () => useDeploymentRecommendation({
        modelId, runtime: 'vllm', image: 'vllm:test', enabled: true,
      }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(ordinary))
    const staleRefresh = result.current.refreshAI
    modelId = 'model-2'
    rerender()

    await expect(staleRefresh()).rejects.toThrow('Recommendation tuple is not stable')
    expect(post.mock.calls.filter(([path]) => path.includes('refresh_ai=true'))).toHaveLength(0)
  })

  it('cancels an ordinary refetch before committing a refresh result', async () => {
    const ordinary = recommendationFixture({ status: 'partial' })
    const refreshed = recommendationFixture({ generated_at: '2026-08-16T04:00:00Z' })
    const lateOrdinary = recommendationFixture({ generated_at: '2026-08-16T05:00:00Z' })
    const refresh = deferred<DeploymentRecommendation>()
    const refetch = deferred<DeploymentRecommendation>()
    let ordinaryCalls = 0
    let refetchSignal: AbortSignal | null | undefined
    vi.spyOn(api, 'post').mockImplementation((path, _body, options) => {
      if (path.includes('refresh_ai=true')) return refresh.promise
      ordinaryCalls += 1
      if (ordinaryCalls === 1) return Promise.resolve(ordinary)
      refetchSignal = options?.signal
      return refetch.promise
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const tuple = {
      modelId: 'model-1', runtime: 'vllm' as const, image: 'vllm:test', providerId: null,
    }
    const { result } = renderHook(
      () => useDeploymentRecommendation({ ...tuple, enabled: true }),
      { wrapper: wrapper(queryClient) },
    )

    await waitFor(() => expect(result.current.data).toEqual(ordinary))
    const refreshPromise = result.current.refreshAI()
    await waitFor(() => expect(api.post).toHaveBeenCalledWith(
      '/api/deployments/recommendations?refresh_ai=true',
      expect.anything(),
      expect.anything(),
    ))
    const refetchPromise = queryClient.refetchQueries({
      queryKey: ['deployment-recommendation', tuple], exact: true,
    })
    await waitFor(() => expect(refetchSignal).toBeInstanceOf(AbortSignal))

    refresh.resolve(refreshed)
    await expect(refreshPromise).resolves.toEqual(refreshed)
    expect(refetchSignal?.aborted).toBe(true)
    refetch.resolve(lateOrdinary)
    await refetchPromise
    expect(queryClient.getQueryData(['deployment-recommendation', tuple])).toEqual(refreshed)
  })
})
