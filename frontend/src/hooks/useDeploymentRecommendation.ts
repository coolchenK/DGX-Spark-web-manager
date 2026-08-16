import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
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

interface RefreshToken {
  sequence: number
  owner: symbol
  controller: AbortController
  tupleKey: string
}

const refreshRegistries = new WeakMap<QueryClient, Map<string, RefreshToken>>()
let refreshSequence = 0


function refreshAbortError(controller: AbortController): Error {
  return controller.signal.reason instanceof Error
    ? controller.signal.reason
    : new DOMException('AI recommendation refresh was cancelled', 'AbortError')
}


function refreshRegistry(queryClient: QueryClient): Map<string, RefreshToken> {
  let registry = refreshRegistries.get(queryClient)
  if (!registry) {
    registry = new Map()
    refreshRegistries.set(queryClient, registry)
  }
  return registry
}


function acquireRefreshToken(
  queryClient: QueryClient,
  tupleKey: string,
  owner: symbol,
): RefreshToken {
  const registry = refreshRegistry(queryClient)
  const previous = registry.get(tupleKey)
  const token: RefreshToken = {
    sequence: ++refreshSequence,
    owner,
    controller: new AbortController(),
    tupleKey,
  }
  registry.set(tupleKey, token)
  previous?.controller.abort()
  return token
}


function ownsRefreshToken(queryClient: QueryClient, token: RefreshToken): boolean {
  const current = refreshRegistries.get(queryClient)?.get(token.tupleKey)
  return current === token && current.sequence === token.sequence
}


function releaseRefreshToken(
  queryClient: QueryClient,
  token: RefreshToken,
  abort: boolean,
): boolean {
  const registry = refreshRegistries.get(queryClient)
  if (!registry || registry.get(token.tupleKey) !== token) return false
  if (abort) token.controller.abort()
  registry.delete(token.tupleKey)
  if (registry.size === 0) refreshRegistries.delete(queryClient)
  return true
}


export function useDeploymentRecommendation({
  modelId = '',
  runtime,
  image = '',
  providerId,
  enabled,
}: DeploymentRecommendationOptions) {
  const queryClient = useQueryClient()
  const [refreshOwner] = useState(() => Symbol('deployment-recommendation-refresh'))
  const ownedRefresh = useRef<RefreshToken | null>(null)
  const releaseOwnedRefresh = useCallback(() => {
    const token = ownedRefresh.current
    if (!token || token.owner !== refreshOwner) return
    releaseRefreshToken(queryClient, token, true)
    if (ownedRefresh.current === token) ownedRefresh.current = null
  }, [queryClient, refreshOwner])
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
      releaseOwnedRefresh()
    }
  }, [enabled, image, modelId, normalizedProviderId, releaseOwnedRefresh, runtime, tupleKey])

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
    const ownedTupleKey = stable.key
    const token = acquireRefreshToken(queryClient, ownedTupleKey, refreshOwner)
    ownedRefresh.current = token

    const assertOwnership = () => {
      if (
        token.controller.signal.aborted
        || !ownsRefreshToken(queryClient, token)
        || !committedInput.current.enabled
        || committedInput.current.tupleKey !== ownedTupleKey
      ) {
        throw refreshAbortError(token.controller)
      }
    }

    try {
      await queryClient.cancelQueries({ queryKey, exact: true })
      assertOwnership()
      const result = await api.post<DeploymentRecommendation>(
        '/api/deployments/recommendations?refresh_ai=true',
        body,
        { signal: token.controller.signal },
      )
      assertOwnership()
      await queryClient.cancelQueries({ queryKey, exact: true })
      assertOwnership()
      queryClient.setQueryData(queryKey, result)
      return result
    } finally {
      releaseRefreshToken(queryClient, token, false)
      if (ownedRefresh.current === token) ownedRefresh.current = null
    }
  }, [activeTuple, body, queryClient, queryKey, refreshOwner, stable])

  return {
    ...query,
    refreshAI,
    activeTupleKey: activeTuple ? tupleKey : null,
  }
}
