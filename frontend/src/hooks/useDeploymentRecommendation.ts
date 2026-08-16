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
  epoch: number
  tuple: RecommendationTuple
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
  const activationSignature = `${enabled ? 'enabled' : 'disabled'}:${tupleKey}`
  const activation = useRef({ signature: '', epoch: 0 })
  if (activation.current.signature !== activationSignature) {
    activation.current = {
      signature: activationSignature,
      epoch: activation.current.epoch + 1,
    }
  }
  const activationEpoch = activation.current.epoch
  const eligible = Boolean(enabled && tuple.modelId && tuple.image)
  const [stable, setStable] = useState<StableRecommendationTuple | null>(null)

  useEffect(() => {
    if (!eligible) return
    const timeout = window.setTimeout(() => {
      setStable({ epoch: activationEpoch, tuple })
    }, 300)
    return () => window.clearTimeout(timeout)
  }, [activationEpoch, eligible, tuple])

  const activeTuple = eligible && stable?.epoch === activationEpoch ? stable.tuple : null
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
    if (!activeTuple || !body) {
      throw new Error('Recommendation tuple is not stable')
    }
    const result = await api.post<DeploymentRecommendation>(
      '/api/deployments/recommendations?refresh_ai=true',
      body,
    )
    queryClient.setQueryData(queryKey, result)
    return result
  }, [activeTuple, body, queryClient, queryKey])

  return { ...query, refreshAI }
}
