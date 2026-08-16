import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'

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


export function useDeploymentRecommendation({
  modelId = '',
  runtime,
  image = '',
  providerId,
  enabled,
}: DeploymentRecommendationOptions) {
  const queryClient = useQueryClient()
  const tuple = useMemo<RecommendationTuple>(() => ({
    modelId,
    runtime,
    image,
    providerId: providerId ?? null,
  }), [image, modelId, providerId, runtime])
  const tupleKey = useMemo(
    () => JSON.stringify([tuple.modelId, tuple.runtime, tuple.image, tuple.providerId]),
    [tuple],
  )
  const [readyKey, setReadyKey] = useState<string | null>(null)

  useEffect(() => {
    const timeout = window.setTimeout(() => setReadyKey(tupleKey), 300)
    return () => window.clearTimeout(timeout)
  }, [tupleKey])

  const body = useMemo(() => ({
    model_id: tuple.modelId,
    runtime: tuple.runtime,
    image: tuple.image,
    provider_id: tuple.providerId,
  }), [tuple])
  const queryKey = useMemo(
    () => ['deployment-recommendation', tuple] as const,
    [tuple],
  )
  const query = useQuery({
    queryKey,
    enabled: Boolean(
      enabled
      && tuple.modelId
      && tuple.image
      && readyKey === tupleKey
    ),
    queryFn: ({ signal }) => api.post<DeploymentRecommendation>(
      '/api/deployments/recommendations',
      body,
      { signal },
    ),
    staleTime: 5 * 60_000,
  })

  useEffect(() => {
    if (!enabled || !tuple.modelId || !tuple.image) {
      void queryClient.cancelQueries({ queryKey, exact: true })
    }
  }, [enabled, queryClient, queryKey, tuple.image, tuple.modelId])

  const refreshAI = useCallback(async () => {
    const result = await api.post<DeploymentRecommendation>(
      '/api/deployments/recommendations?refresh_ai=true',
      body,
    )
    queryClient.setQueryData(queryKey, result)
    return result
  }, [body, queryClient, queryKey])

  return { ...query, refreshAI }
}
