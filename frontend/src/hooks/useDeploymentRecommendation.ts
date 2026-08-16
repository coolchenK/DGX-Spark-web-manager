import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../api/client'
import type { DeploymentRecommendation } from '../api/types'


interface DeploymentRecommendationOptions {
  modelId?: string
  runtime: 'vllm' | 'sglang'
  image?: string
  providerId?: string | null
  enabled: boolean
}

interface RecommendationTuple {
  modelId: string
  runtime: 'vllm' | 'sglang'
  image: string
  providerId: string | null
}

interface StableRecommendationTuple {
  key: string
  tuple: RecommendationTuple
}

interface RefreshOwnership {
  sequence: number
  controller: AbortController | null
  tupleKey: string | null
}


function refreshAbortError(controller: AbortController): Error {
  return controller.signal.reason instanceof Error
    ? controller.signal.reason
    : new DOMException('AI recommendation refresh was cancelled', 'AbortError')
}


export function useDeploymentRecommendation({
  modelId = '',
  runtime,
  image = '',
  providerId,
  enabled,
}: DeploymentRecommendationOptions) {
  const queryClient = useQueryClient()
  const refreshOwnership = useRef<RefreshOwnership>({
    sequence: 0,
    controller: null,
    tupleKey: null,
  })
  const invalidateRefresh = useCallback(() => {
    const ownership = refreshOwnership.current
    ownership.sequence += 1
    ownership.controller?.abort()
    ownership.controller = null
    ownership.tupleKey = null
  }, [])
  const normalizedProviderId = providerId ?? null
  const tuple = useMemo<RecommendationTuple>(() => ({
    modelId,
    runtime,
    image,
    providerId: normalizedProviderId,
  }), [image, modelId, normalizedProviderId, runtime])
  const tupleKey = useMemo(
    () => JSON.stringify([modelId, runtime, image, normalizedProviderId]),
    [image, modelId, normalizedProviderId, runtime],
  )
  const committedInput = useRef<{ enabled: boolean; tupleKey: string | null }>({
    enabled: false,
    tupleKey: null,
  })
  const eligible = Boolean(enabled && tuple.modelId && tuple.image)
  const [stable, setStable] = useState<StableRecommendationTuple | null>(null)

  useEffect(() => {
    setStable(null)
    if (!enabled || !modelId || !image) return
    const timeout = window.setTimeout(() => {
      setStable({
        key: tupleKey,
        tuple: { modelId, runtime, image, providerId: normalizedProviderId },
      })
    }, 300)
    return () => window.clearTimeout(timeout)
  }, [enabled, image, modelId, normalizedProviderId, runtime, tupleKey])

  useEffect(() => {
    committedInput.current = { enabled, tupleKey }
    return () => {
      committedInput.current = { enabled: false, tupleKey: null }
      invalidateRefresh()
    }
  }, [enabled, image, invalidateRefresh, modelId, normalizedProviderId, runtime, tupleKey])

  const activeTuple = eligible && stable?.key === tupleKey ? stable.tuple : null
  const body = useMemo(() => activeTuple ? {
    model_id: activeTuple.modelId,
    runtime: activeTuple.runtime,
    image: activeTuple.image,
    provider_id: activeTuple.providerId,
  } : null, [activeTuple])
  const queryKey = useMemo(
    () => ['deployment-recommendation', activeTuple] as const,
    [activeTuple],
  )
  const query = useQuery({
    queryKey,
    enabled: activeTuple !== null,
    queryFn: ({ signal }) => {
      if (!body) throw new Error('Recommendation tuple is not stable')
      return api.post<DeploymentRecommendation>(
        '/api/deployments/recommendations',
        body,
        { signal },
      )
    },
    staleTime: 5 * 60_000,
  })

  const refreshAI = useCallback(async () => {
    const input = committedInput.current
    if (
      !activeTuple
      || !body
      || !stable
      || !input.enabled
      || input.tupleKey !== stable.key
    ) {
      throw new Error('Recommendation tuple is not stable')
    }
    const ownership = refreshOwnership.current
    ownership.controller?.abort()
    const controller = new AbortController()
    const sequence = ownership.sequence + 1
    const ownedTupleKey = stable.key
    ownership.sequence = sequence
    ownership.controller = controller
    ownership.tupleKey = ownedTupleKey

    const assertOwnership = () => {
      const current = refreshOwnership.current
      if (
        controller.signal.aborted
        || current.sequence !== sequence
        || current.controller !== controller
        || current.tupleKey !== ownedTupleKey
        || !committedInput.current.enabled
        || committedInput.current.tupleKey !== ownedTupleKey
      ) {
        throw refreshAbortError(controller)
      }
    }

    try {
      await queryClient.cancelQueries({ queryKey, exact: true })
      assertOwnership()
      const result = await api.post<DeploymentRecommendation>(
        '/api/deployments/recommendations?refresh_ai=true',
        body,
        { signal: controller.signal },
      )
      assertOwnership()
      await queryClient.cancelQueries({ queryKey, exact: true })
      assertOwnership()
      queryClient.setQueryData(queryKey, result)
      return result
    } finally {
      const current = refreshOwnership.current
      if (current.sequence === sequence && current.controller === controller) {
        current.controller = null
      }
    }
  }, [activeTuple, body, queryClient, queryKey, stable])

  return { ...query, refreshAI }
}
