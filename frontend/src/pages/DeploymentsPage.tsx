import {
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  FileTextOutlined,
  LeftOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  RightOutlined,
  RocketOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Descriptions,
  Drawer,
  Flex,
  Form,
  Grid,
  Input,
  Modal,
  Popconfirm,
  Space,
  Steps,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { api } from '../api/client'
import type {
  Deployment,
  DeploymentRecommendation,
  DraftCandidate,
  ModelAsset,
  Provider,
  RecommendationProvenance,
  RecommendationSource,
  ResourceEstimate,
  RuntimeName,
  TaskRecord,
} from '../api/types'
import { DeploymentBasicsStep } from '../components/deployments/DeploymentBasicsStep'
import {
  DeploymentPreviewStep,
  type DeploymentPreview,
} from '../components/deployments/DeploymentPreviewStep'
import { DraftModelStep } from '../components/deployments/DraftModelStep'
import { LlamaCppStep } from '../components/deployments/LlamaCppStep'
import { RecommendationStep } from '../components/deployments/RecommendationStep'
import { LogViewer } from '../components/LogViewer'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { useDeploymentRecommendation } from '../hooks/useDeploymentRecommendation'
import {
  flattenChangedFields,
  valuesFromRecommendation,
} from '../utils/deploymentRecommendations'
import {
  deploymentToFormValues,
  type DeploymentFormValues,
  type SpeculativeSettings,
} from '../utils/deployments'


interface DeploymentWizardValues extends DeploymentFormValues {
  provider_id?: string
}

const defaultValues: Partial<DeploymentWizardValues> = {
  runtime: 'vllm',
  image: 'vllm/vllm-openai:v0.27.1',
  port: 8100,
  context_length: 32768,
  memory_fraction: 0.8,
  max_concurrency: 8,
  trust_remote_code: false,
  generation_defaults: {},
  speculative: null,
  llama_cpp: null,
  recommendation: null,
  resource_warning_acknowledged: false,
}

const recommendationFieldPaths = new Set([
  'context_length',
  'memory_fraction',
  'max_concurrency',
  'max_batched_tokens',
  'quantization',
  'generation_defaults.temperature',
  'generation_defaults.top_p',
  'generation_defaults.top_k',
  'generation_defaults.min_p',
  'generation_defaults.repetition_penalty',
  'generation_defaults.presence_penalty',
  'generation_defaults.frequency_penalty',
  'generation_defaults.max_tokens',
  'generation_defaults.stop',
])

const basicFields: Array<keyof DeploymentWizardValues> = [
  'model_id',
  'name',
  'model_path',
  'api_model_name',
  'runtime',
  'image',
  'port',
]

const recommendationValidationFields: Array<string | string[]> = [
  'context_length',
  'memory_fraction',
  'max_concurrency',
  'max_batched_tokens',
  ['generation_defaults', 'temperature'],
  ['generation_defaults', 'top_p'],
  ['generation_defaults', 'top_k'],
  ['generation_defaults', 'min_p'],
  ['generation_defaults', 'repetition_penalty'],
  ['generation_defaults', 'presence_penalty'],
  ['generation_defaults', 'frequency_penalty'],
  ['generation_defaults', 'max_tokens'],
]

const recommendationDefaults: Record<string, unknown> = {
  context_length: 32768,
  memory_fraction: 0.8,
  max_concurrency: 8,
  max_batched_tokens: undefined,
  quantization: 'auto',
  'generation_defaults.temperature': undefined,
  'generation_defaults.top_p': undefined,
  'generation_defaults.top_k': undefined,
  'generation_defaults.min_p': undefined,
  'generation_defaults.repetition_penalty': undefined,
  'generation_defaults.presence_penalty': undefined,
  'generation_defaults.frequency_penalty': undefined,
  'generation_defaults.max_tokens': undefined,
  'generation_defaults.stop': undefined,
}


function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError'
}


function recommendationTupleKey(
  modelId: string | undefined,
  runtime: RuntimeName,
  image: string | undefined,
  providerId: string | null | undefined,
): string {
  return JSON.stringify([modelId ?? '', runtime, image ?? '', providerId || null])
}


function recommendationPaths(recommendation: DeploymentRecommendation): Set<string> {
  return new Set([
    ...Object.keys(recommendation.fields),
    ...Object.keys(recommendation.generation_defaults).map((field) => `generation_defaults.${field}`),
  ])
}


function restoredRecommendationFields(values: DeploymentFormValues): Set<string> {
  const restored = new Set<string>()
  for (const path of recommendationFieldPaths) {
    const [root, nested] = path.split('.')
    const value = nested
      ? (values[root as 'generation_defaults'] as Record<string, unknown> | undefined)?.[nested]
      : values[root as keyof DeploymentFormValues]
    if (value !== undefined && value !== null) restored.add(path)
  }
  return restored
}


function hasPathValue(values: DeploymentFormValues, path: string): boolean {
  const [root, nested] = path.split('.')
  const value = nested
    ? (values[root as 'generation_defaults'] as Record<string, unknown> | undefined)?.[nested]
    : values[root as keyof DeploymentFormValues]
  return value !== undefined && value !== null
}


function sourceSnapshot(
  recommendation: DeploymentRecommendation,
  values: DeploymentFormValues,
): Record<string, RecommendationSource> {
  const sources: Record<string, RecommendationSource> = {}
  for (const [field, value] of Object.entries(recommendation.fields)) {
    if (recommendationFieldPaths.has(field) && hasPathValue(values, field)) {
      sources[field] = value.source
    }
  }
  for (const [field, value] of Object.entries(recommendation.generation_defaults)) {
    const path = `generation_defaults.${field}`
    if (recommendationFieldPaths.has(path) && hasPathValue(values, path)) {
      sources[path] = value.source
    }
  }
  return sources
}


function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}


function combinedResourceEstimate(
  base: Partial<ResourceEstimate> | undefined,
  candidate: DraftCandidate | undefined,
): { estimate: Partial<ResourceEstimate> | undefined; unverified: boolean } {
  if (!candidate) return { estimate: base, unverified: false }
  const required = base?.required_bytes
  const existingDraft = base?.draft_weight_bytes ?? 0
  const draftWeight = Math.ceil(Math.max(0, candidate.size_bytes) * 1.15)
  if (typeof required !== 'number') {
    return { estimate: base, unverified: true }
  }
  const calculatedRequired = Math.max(0, required - existingDraft) + draftWeight
  const candidateTotal = candidate.estimated_total_bytes
  const combinedRequired = typeof candidateTotal === 'number'
    ? Math.max(calculatedRequired, candidateTotal)
    : calculatedRequired
  const total = base?.total_bytes
  const available = base?.available_bytes
  const reserved = base?.reserved_bytes
  let decision = base?.decision ?? 'warning'
  if (typeof total === 'number' && combinedRequired > total) {
    decision = 'blocked'
  } else if (
    typeof available === 'number'
    && typeof reserved === 'number'
    && combinedRequired > Math.max(0, available - reserved)
  ) {
    decision = 'warning'
  } else if (decision !== 'blocked' && decision !== 'warning') {
    decision = 'ok'
  }
  const unverified = candidateTotal === null
  const resourceReason = decision === 'blocked'
    ? '基础模型与 Draft Model 的组合资源超过物理统一内存'
    : decision === 'warning'
      ? '基础模型与 Draft Model 的组合资源超过当前可用统一内存余量'
      : '基础模型与 Draft Model 的组合资源在当前统一内存余量内'
  return {
    estimate: {
      ...base,
      draft_weight_bytes: draftWeight,
      required_bytes: combinedRequired,
      decision,
      confidence: unverified ? 'low' : base?.confidence,
      reasons: [...(base?.reasons ?? []), resourceReason],
    },
    unverified,
  }
}

type DeploymentActionName = 'start' | 'stop' | 'restart' | 'delete'


export function DeploymentsPage() {
  const [form] = Form.useForm<DeploymentWizardValues>()
  const [searchParams, setSearchParams] = useSearchParams()
  const deploymentTargetId = searchParams.get('deployment')
  const screens = Grid.useBreakpoint()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [editingDeployment, setEditingDeployment] = useState<Deployment | null>(null)
  const [step, setStep] = useState(0)
  const [editedFields, setEditedFields] = useState<Set<string>>(new Set())
  const editedFieldsRef = useRef<Set<string>>(new Set())
  const [advancedDrafts, setAdvancedDrafts] = useState(false)
  const [draftValidationError, setDraftValidationError] = useState<string | null>(null)
  const [preview, setPreview] = useState<DeploymentPreview | null>(null)
  const [previewedPayload, setPreviewedPayload] = useState<DeploymentFormValues | null>(null)
  const [logsFor, setLogsFor] = useState<Deployment | null>(null)
  const [uninstallTarget, setUninstallTarget] = useState<Deployment | null>(null)
  const [uninstallConfirmation, setUninstallConfirmation] = useState('')
  const [pendingDeploymentActions, setPendingDeploymentActions] = useState<
    Map<string, DeploymentActionName>
  >(new Map())
  const applyingRecommendation = useRef(false)
  const drawerOpenRef = useRef(false)
  const uninstallTargetRef = useRef<Deployment | null>(null)
  const pendingDeploymentIdsRef = useRef<Set<string>>(new Set())
  const previewSequence = useRef(0)
  const previewController = useRef<AbortController | null>(null)
  const lastAppliedRecommendation = useRef<DeploymentRecommendation | null>(null)
  const restoredTupleKey = useRef<string | null>(null)
  const queryClient = useQueryClient()

  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api.get<Deployment[]>('/api/deployments'),
    refetchInterval: 8_000,
    refetchOnMount: deploymentTargetId ? 'always' : undefined,
  })
  const locatedDeployment = deployments.data?.find((item) => item.id === deploymentTargetId)
  const visibleDeployments = locatedDeployment ? [locatedDeployment] : deployments.data ?? []
  const locatorValidated = deployments.isFetchedAfterMount && !deployments.error
  const clearDeploymentLocator = () => {
    const next = new URLSearchParams(searchParams)
    next.delete('deployment')
    setSearchParams(next, { replace: true })
  }
  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.get<ModelAsset[]>('/api/models'),
  })
  const providers = useQuery({
    queryKey: ['providers'],
    queryFn: () => api.get<Provider[]>('/api/providers'),
  })
  const logs = useQuery({
    queryKey: ['deployment-logs', logsFor?.id],
    queryFn: () => api.get<{ logs: string }>(`/api/deployments/${logsFor?.id}/logs?tail=1000`),
    enabled: Boolean(logsFor),
    refetchInterval: logsFor ? 5_000 : false,
  })

  const watchOptions = useMemo(() => ({ form, preserve: true }), [form])
  const modelId = Form.useWatch('model_id', watchOptions)
  const runtime = Form.useWatch('runtime', watchOptions) ?? 'vllm'
  const image = Form.useWatch('image', watchOptions)
  const providerId = Form.useWatch('provider_id', watchOptions)
  const selectedDraftId = Form.useWatch(['speculative', 'draft_model_id'], watchOptions)
  const currentTupleKey = useMemo(
    () => recommendationTupleKey(modelId, runtime, image, providerId),
    [image, modelId, providerId, runtime],
  )

  const recommendation = useDeploymentRecommendation({
    modelId,
    runtime,
    image,
    providerId: providerId || null,
    enabled: drawerOpen,
  })
  const activeRecommendation = useMemo(() => {
    const data = recommendation.data
    if (
      !data
      || data.model_id !== modelId
      || data.runtime !== runtime
      || recommendation.activeTupleKey !== currentTupleKey
    ) return undefined
    return data
  }, [currentTupleKey, modelId, recommendation.activeTupleKey, recommendation.data, runtime])

  const replaceEditedFields = useCallback((next: Set<string>) => {
    editedFieldsRef.current = next
    setEditedFields(next)
  }, [])

  const invalidatePreview = useCallback((fallbackStep?: 0 | 1 | 2) => {
    previewSequence.current += 1
    previewController.current?.abort()
    previewController.current = null
    setPreview(null)
    setPreviewedPayload(null)
    if (fallbackStep !== undefined) {
      setStep((current) => current === 3 ? fallbackStep : current)
    }
  }, [])

  const clearRecommendationValues = useCallback((
    paths: Iterable<string>,
    forcedPaths: ReadonlySet<string> = new Set(),
  ) => {
    const updates: Record<string, unknown> = {}
    const generationUpdates: Record<string, unknown> = {}
    const cleared = new Set<string>()
    for (const path of paths) {
      if (!recommendationFieldPaths.has(path)) continue
      if (editedFieldsRef.current.has(path) && !forcedPaths.has(path)) continue
      const [root, nested] = path.split('.')
      if (nested) generationUpdates[nested] = recommendationDefaults[path]
      else updates[root] = recommendationDefaults[path]
      cleared.add(path)
    }
    if (Object.keys(generationUpdates).length) updates.generation_defaults = generationUpdates
    if (Object.keys(updates).length) {
      applyingRecommendation.current = true
      form.setFieldsValue(updates)
      applyingRecommendation.current = false
    }
    if (cleared.size) {
      replaceEditedFields(new Set(
        [...editedFieldsRef.current].filter((path) => !cleared.has(path)),
      ))
    }
    return cleared
  }, [form, replaceEditedFields])

  const priorRecommendationPaths = useCallback(() => new Set([
    ...(lastAppliedRecommendation.current
      ? recommendationPaths(lastAppliedRecommendation.current)
      : []),
    ...Object.keys(form.getFieldValue('recommendation')?.sources ?? {}),
  ]), [form])

  const applyRecommendation = useCallback((result: DeploymentRecommendation, force: boolean) => {
    const currentEdited = editedFieldsRef.current
    const nextPaths = recommendationPaths(result)
    const previousPaths = lastAppliedRecommendation.current
      ? recommendationPaths(lastAppliedRecommendation.current)
      : new Set<string>()
    const stalePaths = new Set([...previousPaths].filter((path) => !nextPaths.has(path)))
    clearRecommendationValues(stalePaths, force ? stalePaths : new Set())
    invalidatePreview(1)
    applyingRecommendation.current = true
    form.setFieldsValue({
      ...valuesFromRecommendation(result, currentEdited, force),
      resource_warning_acknowledged: false,
    })
    applyingRecommendation.current = false
    lastAppliedRecommendation.current = result
    replaceEditedFields(new Set([...editedFieldsRef.current].filter((path) => (
      path !== 'resource_warning_acknowledged'
      && (!force || !nextPaths.has(path))
    ))))
  }, [clearRecommendationValues, form, invalidatePreview, replaceEditedFields])

  useEffect(() => {
    if (activeRecommendation) applyRecommendation(activeRecommendation, false)
  }, [activeRecommendation, applyRecommendation])

  useEffect(() => {
    if (!drawerOpen || form.getFieldValue('provider_id') !== undefined || !providers.data) return
    const enabledProviders = providers.data.filter((provider) => provider.enabled)
    const saved = window.localStorage.getItem('dgx-deployment-recommendation-provider')
    const selected = enabledProviders.length === 1
      ? enabledProviders[0].id
      : enabledProviders.some((provider) => provider.id === saved) ? saved ?? '' : ''
    form.setFieldValue('provider_id', selected)
  }, [drawerOpen, form, providers.data])

  const resetWizardState = useCallback(() => {
    invalidatePreview()
    setStep(0)
    replaceEditedFields(new Set())
    setAdvancedDrafts(false)
    setDraftValidationError(null)
    lastAppliedRecommendation.current = null
    restoredTupleKey.current = null
  }, [invalidatePreview, replaceEditedFields])

  const selectModel = useCallback((selectedModelId: string) => {
    const model = models.data?.find((item) => item.id === selectedModelId)
    if (!model) return
    const shortName = model.name.split('/').pop() ?? model.name
    applyingRecommendation.current = true
    form.setFieldsValue({
      name: shortName,
      model_path: model.local_path,
      api_model_name: model.alias ?? shortName.toLowerCase(),
      model_id: model.id,
      context_length: 32768,
      memory_fraction: 0.8,
      max_concurrency: 8,
      max_batched_tokens: undefined,
      quantization: runtime === 'llama_cpp' ? 'gguf' : 'auto',
      generation_defaults: Object.fromEntries(
        [...recommendationFieldPaths]
          .filter((path) => path.startsWith('generation_defaults.'))
          .map((path) => [path.split('.')[1], undefined]),
      ),
      speculative: null,
      llama_cpp: runtime === 'llama_cpp' ? {
        gpu_layers: 'all',
        jinja: true,
        continuous_batching: true,
        mtp_enabled: false,
        mtp_tokens: 3,
      } : null,
      resource_warning_acknowledged: false,
      recommendation: null,
    })
    applyingRecommendation.current = false
    replaceEditedFields(new Set())
    lastAppliedRecommendation.current = null
    restoredTupleKey.current = null
    setAdvancedDrafts(false)
    setDraftValidationError(null)
    invalidatePreview(0)
  }, [form, invalidatePreview, models.data, replaceEditedFields, runtime])

  const openCreate = useCallback(() => {
    setEditingDeployment(null)
    resetWizardState()
    form.resetFields()
    applyingRecommendation.current = true
    form.setFieldsValue(defaultValues)
    applyingRecommendation.current = false
    drawerOpenRef.current = true
    setDrawerOpen(true)
  }, [form, resetWizardState])

  const openFromDeployment = (deployment: Deployment, mode: 'edit' | 'clone') => {
    const model = models.data?.find((item) => item.id === deployment.model_id)
    if (!model) {
      message.error('该实例未关联可用的本地模型，无法编辑或克隆')
      return
    }
    const restored = deploymentToFormValues(deployment, model, mode)
    resetWizardState()
    setEditingDeployment(mode === 'edit' ? deployment : null)
    applyingRecommendation.current = true
    form.resetFields()
    form.setFieldsValue({
      ...restored,
      provider_id: restored.recommendation?.provider_id ?? '',
    })
    applyingRecommendation.current = false
    restoredTupleKey.current = JSON.stringify([
      restored.model_id,
      restored.runtime,
      restored.image,
      restored.recommendation?.provider_id ?? null,
    ])
    if (mode === 'edit') replaceEditedFields(restoredRecommendationFields(restored))
    setAdvancedDrafts(Boolean(restored.speculative))
    drawerOpenRef.current = true
    setDrawerOpen(true)
  }

  useEffect(() => {
    const selectedModelId = searchParams.get('model')
    if (selectedModelId && models.data?.some((item) => item.id === selectedModelId)) {
      openCreate()
      selectModel(selectedModelId)
    }
  }, [models.data, openCreate, searchParams, selectModel])

  const closeDrawer = () => {
    drawerOpenRef.current = false
    invalidatePreview()
    setDrawerOpen(false)
    setEditingDeployment(null)
    resetWizardState()
    form.resetFields()
  }

  const payloadFromForm = useCallback((): DeploymentFormValues => {
    const values = form.getFieldsValue(true)
    const { provider_id: _, ...deploymentValues } = values
    const data = activeRecommendation
    let provenance: RecommendationProvenance | null = (
      currentTupleKey === restoredTupleKey.current
      ? deploymentValues.recommendation ?? null
      : null
    )
    const snapshot = data?.resource_snapshot
    if (
      data?.evidence_hash
      && typeof snapshot?.total_bytes === 'number'
      && typeof snapshot.available_bytes === 'number'
      && typeof snapshot.reserved_bytes === 'number'
    ) {
      const sources = sourceSnapshot(data, deploymentValues)
      provenance = {
        generated_at: data.generated_at,
        evidence_hash: data.evidence_hash,
        provider_id: providerId || null,
        resource_snapshot: {
          total_bytes: snapshot.total_bytes,
          available_bytes: snapshot.available_bytes,
          reserved_bytes: snapshot.reserved_bytes,
        },
        modified_fields: [...editedFieldsRef.current]
          .filter((path) => path in sources && hasPathValue(deploymentValues, path))
          .sort(),
        sources,
      }
    }
    const savedSpeculative = deploymentValues.speculative
    let speculative: SpeculativeSettings | null = null
    if (savedSpeculative?.draft_model_id) {
      speculative = {
        draft_model_id: savedSpeculative.draft_model_id,
        method: savedSpeculative.method,
        manual_review_acknowledged: savedSpeculative.manual_review_acknowledged === true,
        ...(runtime === 'vllm' && savedSpeculative.num_speculative_tokens != null
          ? { num_speculative_tokens: savedSpeculative.num_speculative_tokens }
          : {}),
        ...(runtime === 'sglang'
          && savedSpeculative.num_steps != null
          && savedSpeculative.eagle_top_k != null
          && savedSpeculative.num_draft_tokens != null
          ? {
              num_steps: savedSpeculative.num_steps,
              eagle_top_k: savedSpeculative.eagle_top_k,
              num_draft_tokens: savedSpeculative.num_draft_tokens,
            }
          : {}),
      }
    }
    const result: DeploymentFormValues = {
      ...deploymentValues,
      route_alias: deploymentValues.route_alias || undefined,
      generation_defaults: Object.fromEntries(
        Object.entries(deploymentValues.generation_defaults ?? {})
          .filter(([, value]) => value !== undefined && value !== null),
      ),
      speculative: runtime === 'llama_cpp' ? null : speculative,
      llama_cpp: runtime === 'llama_cpp' ? {
        model_file: deploymentValues.llama_cpp?.model_file || undefined,
        mmproj_file: deploymentValues.llama_cpp?.mmproj_file || undefined,
        gpu_layers: deploymentValues.llama_cpp?.gpu_layers ?? 'all',
        jinja: deploymentValues.llama_cpp?.jinja !== false,
        continuous_batching: deploymentValues.llama_cpp?.continuous_batching !== false,
        mtp_enabled: deploymentValues.llama_cpp?.mtp_enabled === true,
        mtp_tokens: deploymentValues.llama_cpp?.mtp_tokens ?? 3,
      } : null,
      recommendation: provenance,
      resource_warning_acknowledged: deploymentValues.resource_warning_acknowledged === true,
    }
    if (runtime !== 'vllm' || result.max_batched_tokens === undefined) {
      delete result.max_batched_tokens
    }
    return cloneJson(result)
  }, [activeRecommendation, currentTupleKey, form, providerId, runtime])

  const previewMutation = useMutation({
    mutationFn: async ({
      payload,
      sequence,
      controller,
    }: {
      payload: DeploymentFormValues
      sequence: number
      controller: AbortController
    }) => {
      const suffix = editingDeployment ? `?deployment_id=${encodeURIComponent(editingDeployment.id)}` : ''
      const result = await api.post<DeploymentPreview>(
        `/api/deployments/preview${suffix}`,
        payload,
        { signal: controller.signal },
      )
      return { result, payload, sequence, controller }
    },
    onSuccess: ({ result, payload, sequence, controller }) => {
      if (
        sequence !== previewSequence.current
        || controller.signal.aborted
        || !drawerOpenRef.current
      ) return
      previewController.current = null
      setPreview(cloneJson(result))
      setPreviewedPayload(cloneJson(payload))
      setStep(3)
    },
    onError: (error: Error, variables) => {
      if (variables.sequence !== previewSequence.current || isAbortError(error)) return
      previewController.current = null
      message.error(error.message)
    },
  })
  const saveMutation = useMutation({
    mutationFn: (values: DeploymentFormValues) => editingDeployment
      ? api.patch<TaskRecord>(`/api/deployments/${editingDeployment.id}`, values)
      : api.post<TaskRecord>('/api/deployments', values),
    onSuccess: () => {
      message.success(editingDeployment ? '部署更新任务已创建' : '部署任务已创建')
      closeDrawer()
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error: Error) => {
      if (!isAbortError(error)) message.error(error.message)
    },
  })
  const closeDiscoveredUninstall = (targetId?: string) => {
    if (targetId && uninstallTargetRef.current?.id !== targetId) return
    uninstallTargetRef.current = null
    setUninstallTarget(null)
    setUninstallConfirmation('')
  }
  const openDiscoveredUninstall = (deployment: Deployment) => {
    if (!deployment.container_name) return
    uninstallTargetRef.current = deployment
    setUninstallConfirmation('')
    setUninstallTarget(deployment)
  }
  const clearPendingDeploymentAction = (deploymentId: string) => {
    if (!pendingDeploymentIdsRef.current.delete(deploymentId)) return
    setPendingDeploymentActions((current) => {
      const next = new Map(current)
      next.delete(deploymentId)
      return next
    })
  }
  const action = useMutation({
    mutationFn: async ({
      deployment,
      actionName,
    }: {
      deployment: Deployment
      actionName: DeploymentActionName
    }) => {
      const path = `/api/deployments/${deployment.id}/${actionName}`
      try {
        if (actionName === 'delete' && !deployment.managed) {
          return await api.post<TaskRecord>(path, {
            confirm_container_name: deployment.container_name,
          })
        }
        return await api.post<TaskRecord>(path)
      } finally {
        clearPendingDeploymentAction(deployment.id)
      }
    },
    onSuccess: (_task, { deployment, actionName }) => {
      if (actionName === 'delete' && !deployment.managed) {
        closeDiscoveredUninstall(deployment.id)
      }
      const successMessages = {
        start: '启动实例任务已创建',
        stop: '停止实例任务已创建',
        restart: '重启实例任务已创建',
        delete: '卸载服务任务已创建',
      }
      message.success(successMessages[actionName])
      queryClient.invalidateQueries({ queryKey: ['deployments'] })
      queryClient.invalidateQueries({ queryKey: ['gateway-stats'] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error: Error, { deployment, actionName }) => {
      if (
        actionName === 'delete'
        && !deployment.managed
        && uninstallTargetRef.current?.id !== deployment.id
      ) return
      message.error(error.message)
    },
  })
  const runDeploymentAction = (
    deployment: Deployment,
    actionName: DeploymentActionName,
  ) => {
    if (pendingDeploymentIdsRef.current.has(deployment.id)) return
    pendingDeploymentIdsRef.current.add(deployment.id)
    setPendingDeploymentActions((current) => new Map(current).set(deployment.id, actionName))
    action.mutate({ deployment, actionName })
  }
  const uninstallTargetPendingAction = uninstallTarget
    ? pendingDeploymentActions.get(uninstallTarget.id)
    : undefined
  const discoveredUninstallPending = uninstallTargetPendingAction === 'delete'

  const handleRetryAI = async () => {
    const requestedTupleKey = recommendation.activeTupleKey
    if (!requestedTupleKey || requestedTupleKey !== currentTupleKey) return
    try {
      const refreshed = await recommendation.refreshAI()
      const currentModelId = form.getFieldValue('model_id') as string | undefined
      const currentRuntime = form.getFieldValue('runtime') as RuntimeName
      const currentImage = form.getFieldValue('image') as string | undefined
      const currentProviderId = form.getFieldValue('provider_id') as string | null | undefined
      const formTupleKey = recommendationTupleKey(
        currentModelId,
        currentRuntime,
        currentImage,
        currentProviderId,
      )
      if (
        requestedTupleKey === formTupleKey
        && refreshed.model_id === currentModelId
        && refreshed.runtime === currentRuntime
        && (
          refreshed.runtime_capabilities.image == null
          || refreshed.runtime_capabilities.image === currentImage
        )
      ) {
        applyRecommendation(refreshed, false)
      }
    } catch (error) {
      if (!isAbortError(error)) message.error(error instanceof Error ? error.message : '推荐分析失败')
    }
  }

  const handleDraftSelect = (candidate?: DraftCandidate) => {
    invalidatePreview(2)
    applyingRecommendation.current = true
    if (!candidate) {
      form.setFieldValue('speculative', null)
    } else {
      const restored = form.getFieldValue('speculative')
      const keepRestored = restored?.draft_model_id === candidate.model_id
      const next: SpeculativeSettings = {
        draft_model_id: candidate.model_id,
        method: candidate.method ?? 'draft_model',
        num_speculative_tokens: runtime === 'vllm'
          ? (keepRestored ? restored.num_speculative_tokens ?? 5 : 5)
          : undefined,
        num_steps: runtime === 'sglang' && keepRestored ? restored.num_steps : undefined,
        eagle_top_k: runtime === 'sglang' && keepRestored ? restored.eagle_top_k : undefined,
        num_draft_tokens: runtime === 'sglang' && keepRestored ? restored.num_draft_tokens : undefined,
        manual_review_acknowledged: keepRestored
          ? restored.manual_review_acknowledged
          : false,
      }
      form.setFieldValue('speculative', next)
    }
    form.setFieldValue('resource_warning_acknowledged', false)
    applyingRecommendation.current = false
    replaceEditedFields(new Set([
      ...[...editedFieldsRef.current].filter((path) => path !== 'resource_warning_acknowledged'),
      'speculative.draft_model_id',
    ]))
    setDraftValidationError(null)
  }

  const selectedDraft = activeRecommendation?.draft_candidates.find(
    (item) => item.model_id === selectedDraftId,
  )
  const draftResources = useMemo(
    () => combinedResourceEstimate(activeRecommendation?.resource_estimate, selectedDraft),
    [activeRecommendation?.resource_estimate, selectedDraft],
  )

  const goForward = async () => {
    if (step === 0) {
      try {
        await form.validateFields(basicFields)
      } catch {
        return
      }
      setStep(1)
      return
    }
    if (step === 1) {
      try {
        await form.validateFields(recommendationValidationFields)
      } catch {
        return
      }
      setStep(2)
      return
    }
    if (step === 2) {
      if (runtime === 'llama_cpp') {
        try {
          await form.validateFields([
            ['llama_cpp', 'model_file'],
            ['llama_cpp', 'mmproj_file'],
            ['llama_cpp', 'mtp_tokens'],
          ])
        } catch {
          return
        }
        const resourceDecision = draftResources.estimate?.decision
        if (resourceDecision === 'warning' && !form.getFieldValue('resource_warning_acknowledged')) {
          setDraftValidationError('请确认统一内存资源警告')
          return
        }
        if (resourceDecision === 'blocked') {
          message.error('资源估算超过 DGX Spark 硬上限，无法生成部署任务')
          return
        }
        const payload = payloadFromForm()
        previewController.current?.abort()
        const controller = new AbortController()
        const sequence = previewSequence.current + 1
        previewSequence.current = sequence
        previewController.current = controller
        previewMutation.mutate({ payload: cloneJson(payload), sequence, controller })
        return
      }
      if (activeRecommendation && selectedDraftId && !selectedDraft) {
        setDraftValidationError('当前 Draft Model 不在最新候选列表中')
        return
      }
      if (selectedDraft?.status === 'incompatible') {
        setDraftValidationError('当前 Draft Model 与基础模型不兼容')
        return
      }
      const speculative = form.getFieldValue('speculative') as SpeculativeSettings | null
      try {
        await form.validateFields(runtime === 'vllm'
          ? [['speculative', 'num_speculative_tokens']]
          : [
              ['speculative', 'num_steps'],
              ['speculative', 'eagle_top_k'],
              ['speculative', 'num_draft_tokens'],
            ])
      } catch {
        return
      }
      if (speculative) {
        const grouped = [speculative.num_steps, speculative.eagle_top_k, speculative.num_draft_tokens]
        if (runtime === 'vllm' && grouped.some((value) => value != null)) {
          setDraftValidationError('vLLM 不支持 SGLang 分组推测解码参数')
          return
        }
        if (runtime === 'sglang' && speculative.num_speculative_tokens != null) {
          setDraftValidationError('SGLang 不支持每轮推测 Token 参数')
          return
        }
        if (runtime === 'sglang' && grouped.some((value) => value != null) && !grouped.every((value) => value != null)) {
          setDraftValidationError('SGLang 的三个推测解码参数必须全部填写或全部留空')
          return
        }
      }
      if (selectedDraft?.status === 'review' && !form.getFieldValue(['speculative', 'manual_review_acknowledged'])) {
        setDraftValidationError('请确认 Draft Model 配对风险')
        return
      }
      const resourceDecision = draftResources.estimate?.decision
      if (selectedDraft && draftResources.unverified && !form.getFieldValue('resource_warning_acknowledged')) {
        setDraftValidationError('请确认 Draft Model 资源未验证')
        return
      }
      if (resourceDecision === 'warning' && !form.getFieldValue('resource_warning_acknowledged')) {
        setDraftValidationError('请确认统一内存资源警告')
        return
      }
      if (resourceDecision === 'blocked') {
        message.error('资源估算超过 DGX Spark 硬上限，无法生成部署任务')
        return
      }
      const payload = payloadFromForm()
      previewController.current?.abort()
      const controller = new AbortController()
      const sequence = previewSequence.current + 1
      previewSequence.current = sequence
      previewController.current = controller
      previewMutation.mutate({ payload: cloneJson(payload), sequence, controller })
    }
  }

  const goBack = () => {
    if (step === 3) {
      invalidatePreview()
      setStep(2)
      return
    }
    if (step === 2) {
      invalidatePreview()
      setStep(1)
      return
    }
    setStep((current) => Math.max(0, current - 1))
  }

  const onValuesChange = (changed: Record<string, unknown>) => {
    if (applyingRecommendation.current) return
    invalidatePreview(step === 0 ? 0 : step === 1 ? 1 : 2)
    setDraftValidationError(null)
    const changedPaths = flattenChangedFields(changed)
    replaceEditedFields(new Set([...editedFieldsRef.current, ...changedPaths]))
    if ('runtime' in changed) {
      const previousPaths = priorRecommendationPaths()
      const forced = new Set(['max_batched_tokens', 'quantization'])
      clearRecommendationValues(new Set([...previousPaths, ...forced]), forced)
      lastAppliedRecommendation.current = null
      restoredTupleKey.current = null
      const nextRuntime = changed.runtime as RuntimeName
      applyingRecommendation.current = true
      form.setFieldsValue({
        image: nextRuntime === 'vllm'
          ? 'vllm/vllm-openai:v0.27.1'
          : nextRuntime === 'sglang'
            ? 'sglang-inkling:specforge'
            : 'nvidia/cuda:12.9.0-devel-ubuntu24.04',
        speculative: null,
        llama_cpp: nextRuntime === 'llama_cpp' ? {
          gpu_layers: 'all',
          jinja: true,
          continuous_batching: true,
          mtp_enabled: false,
          mtp_tokens: 3,
        } : null,
        resource_warning_acknowledged: false,
        recommendation: null,
        max_batched_tokens: undefined,
        quantization: nextRuntime === 'llama_cpp' ? 'gguf' : 'auto',
      })
      applyingRecommendation.current = false
      replaceEditedFields(new Set([...editedFieldsRef.current].filter((path) => (
        !path.startsWith('speculative.')
        && path !== 'resource_warning_acknowledged'
        && path !== 'max_batched_tokens'
        && path !== 'quantization'
      ))))
      return
    }
    if ('image' in changed) {
      const previousPaths = priorRecommendationPaths()
      const forced = new Set(['max_batched_tokens', 'quantization'])
      clearRecommendationValues(new Set([...previousPaths, ...forced]), forced)
      lastAppliedRecommendation.current = null
      restoredTupleKey.current = null
      applyingRecommendation.current = true
      form.setFieldsValue({
        speculative: null,
        resource_warning_acknowledged: false,
        recommendation: null,
        max_batched_tokens: undefined,
        quantization: runtime === 'llama_cpp' ? 'gguf' : 'auto',
      })
      applyingRecommendation.current = false
      replaceEditedFields(new Set([...editedFieldsRef.current].filter((path) => (
        !path.startsWith('speculative.')
        && path !== 'resource_warning_acknowledged'
        && path !== 'max_batched_tokens'
        && path !== 'quantization'
      ))))
      return
    }
    if ('provider_id' in changed) {
      const previousPaths = priorRecommendationPaths()
      clearRecommendationValues(previousPaths)
      lastAppliedRecommendation.current = null
      restoredTupleKey.current = null
      applyingRecommendation.current = true
      form.setFieldsValue({
        speculative: null,
        resource_warning_acknowledged: false,
        recommendation: null,
      })
      applyingRecommendation.current = false
      replaceEditedFields(new Set([...editedFieldsRef.current].filter((path) => (
        !path.startsWith('speculative.')
        && path !== 'resource_warning_acknowledged'
      ))))
    }
  }

  const operationButtons = (item: Deployment, mobile = false) => {
    const isRunning = item.status === 'running'
    const primaryAction = isRunning ? 'stop' : 'start'
    const pendingAction = pendingDeploymentActions.get(item.id)
    const rowPending = Boolean(pendingAction)
    const primaryPending = pendingAction === primaryAction
    const deletePending = pendingAction === 'delete'
    const uninstallButton = (
      <Tooltip title={item.managed || item.container_name
        ? `卸载服务 ${item.name}`
        : '缺少容器名称，无法安全确认卸载。'}>
        <span>
          <Button
            size={mobile ? 'middle' : 'small'}
            danger
            icon={<DeleteOutlined />}
            aria-label={`卸载服务 ${item.name}`}
            disabled={rowPending || (!item.managed && !item.container_name)}
            loading={deletePending}
            onClick={item.managed ? undefined : () => openDiscoveredUninstall(item)}
          >
            卸载服务
          </Button>
        </span>
      </Tooltip>
    )
    return <Space wrap>
      <Button size={mobile ? 'middle' : 'small'} icon={<FileTextOutlined />} onClick={() => setLogsFor(item)}>
        {mobile ? '日志' : null}
      </Button>
      <Tooltip title={isRunning
        ? '释放 GPU/统一内存，但保留容器配置、网关别名和模型文件。'
        : `启动实例 ${item.name}`}>
        <Button
          size={mobile ? 'middle' : 'small'}
          loading={primaryPending}
          disabled={rowPending}
          icon={isRunning ? <StopOutlined /> : <PlayCircleOutlined />}
          onClick={() => runDeploymentAction(item, primaryAction)}
          aria-label={`${isRunning ? '停止实例' : '启动实例'} ${item.name}`}
        >
          {isRunning ? '停止实例' : '启动'}
        </Button>
      </Tooltip>
      <Tooltip title="重启实例">
        <Button
          size={mobile ? 'middle' : 'small'}
          icon={<ReloadOutlined />}
          loading={pendingAction === 'restart'}
          disabled={rowPending}
          onClick={() => runDeploymentAction(item, 'restart')}
          aria-label="重启实例"
        />
      </Tooltip>
      {item.managed && <>
        <Tooltip title="编辑部署参数">
          <Button size={mobile ? 'middle' : 'small'} icon={<EditOutlined />} onClick={() => openFromDeployment(item, 'edit')} aria-label="编辑部署参数" />
        </Tooltip>
        <Tooltip title="克隆部署">
          <Button size={mobile ? 'middle' : 'small'} icon={<CopyOutlined />} onClick={() => openFromDeployment(item, 'clone')} aria-label="克隆部署" />
        </Tooltip>
        <Popconfirm
          title={`卸载服务 ${item.name}`}
          description="将删除服务容器、部署记录和网关路由，但保留模型文件。"
          okText="确认卸载"
          cancelText="取消"
          okButtonProps={{ danger: true }}
          onConfirm={() => runDeploymentAction(item, 'delete')}
        >
          {uninstallButton}
        </Popconfirm>
      </>}
      {!item.managed && uninstallButton}
    </Space>
  }

  const stepContent = [
    <DeploymentBasicsStep
      key="basics"
      models={models.data ?? []}
      providers={providers.data ?? []}
      runtime={runtime}
      loading={models.isLoading}
      providersLoading={providers.isLoading}
      onModelChange={selectModel}
      onProviderChange={(selectedProviderId) => {
        if (selectedProviderId) {
          window.localStorage.setItem('dgx-deployment-recommendation-provider', selectedProviderId)
        } else {
          window.localStorage.removeItem('dgx-deployment-recommendation-provider')
        }
        invalidatePreview(0)
      }}
    />,
    <RecommendationStep
      key="recommendation"
      recommendation={activeRecommendation}
      editedFields={editedFields}
      loading={recommendation.isLoading || recommendation.isFetching}
      refreshing={recommendation.isFetching}
      error={recommendation.error}
      runtime={runtime}
      editing={Boolean(editingDeployment)}
      onReapplyAll={() => activeRecommendation && applyRecommendation(activeRecommendation, true)}
      onRetryAI={handleRetryAI}
    />,
    runtime === 'llama_cpp'
      ? <LlamaCppStep key="llama-cpp" />
      : <DraftModelStep
          key="draft"
          candidates={activeRecommendation?.draft_candidates ?? []}
          selectedId={selectedDraftId}
          runtime={runtime}
          advanced={advancedDrafts}
          resourceEstimate={draftResources.estimate}
          resourceUnverified={draftResources.unverified}
          validationError={draftValidationError}
          onAdvancedChange={setAdvancedDrafts}
          onSelect={handleDraftSelect}
        />,
    preview
      ? <DeploymentPreviewStep
          key="preview"
          preview={preview}
          editing={Boolean(editingDeployment)}
          fallbackRoute={form.getFieldValue('api_model_name')}
        />
      : <div key="preview-empty" />,
  ]

  return (
    <div className="page-stack">
      <PageHeader
        title="部署实例"
        description="管理已发现容器，并通过受控适配器创建新服务"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>新建部署</Button>}
      />
      {deploymentTargetId && !deployments.error && (locatedDeployment || locatorValidated) && (
        <Alert
          type={locatedDeployment ? 'info' : 'warning'}
          showIcon
          message={locatedDeployment ? `正在定位部署 ${locatedDeployment.name}` : '未找到指定部署'}
          description={locatedDeployment
            ? '当前仅显示该部署实例。'
            : `未找到 ID 为 ${deploymentTargetId} 的部署，当前显示全部部署。`}
          action={<Button size="small" onClick={clearDeploymentLocator}>显示全部部署</Button>}
        />
      )}
      <QueryState loading={deployments.isLoading} error={deployments.error} empty={!visibleDeployments.length}>
        <ResponsiveDataView data={visibleDeployments} rowKey="id" columns={[
          { title: '实例', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.api_model_name}</small></div> },
          { title: '运行时', dataIndex: 'runtime', width: 100, render: (value) => <Tag>{value}</Tag> },
          { title: '端点', dataIndex: 'endpoint_url' },
          { title: '所有权', dataIndex: 'managed', width: 90, render: (value) => value ? '管理器' : '已发现' },
          { title: '状态', dataIndex: 'health', width: 100, render: (value) => <StatusBadge status={value} /> },
          { title: '操作', width: 330, render: (_, item) => operationButtons(item) },
        ]} renderMobile={(item) => (
          <div className="mobile-record">
            <Flex justify="space-between"><strong>{item.name}</strong><StatusBadge status={item.health} /></Flex>
            <Typography.Text type="secondary">{item.api_model_name}</Typography.Text>
            <dl><div><dt>运行时</dt><dd>{item.runtime}</dd></div><div><dt>端点</dt><dd>{item.endpoint_url}</dd></div></dl>
            {operationButtons(item, true)}
          </div>
        )} />
      </QueryState>
      <Drawer
        title={editingDeployment ? `编辑 ${editingDeployment.name}` : '新建模型部署'}
        width="min(900px, 100vw)"
        open={drawerOpen}
        onClose={closeDrawer}
        destroyOnHidden
      >
        <Steps
          className="deployment-wizard-steps"
          size="small"
          current={step}
          direction={screens.md ? 'horizontal' : 'vertical'}
          items={[
            { title: '基础模型' },
            { title: '推荐配置' },
            { title: runtime === 'llama_cpp' ? 'llama.cpp' : 'Draft Model' },
            { title: '部署预览' },
          ]}
        />
        <Form
          form={form}
          layout="vertical"
          initialValues={defaultValues}
          onValuesChange={onValuesChange}
        >
          {stepContent[step]}
          <div className="deployment-wizard-actions">
            <Button aria-label="上一步" icon={<LeftOutlined />} disabled={step === 0} onClick={goBack}>上一步</Button>
            {step < 3 ? (
              <Button
                type="primary"
                iconPosition="end"
                icon={<RightOutlined />}
                loading={previewMutation.isPending}
                onClick={goForward}
                aria-label={step === 2 ? '生成部署预览' : '下一步'}
              >
                {step === 2 ? '生成部署预览' : '下一步'}
              </Button>
            ) : (
              <Button
                type="primary"
                icon={<RocketOutlined />}
                loading={saveMutation.isPending}
                disabled={!previewedPayload}
                onClick={() => previewedPayload && saveMutation.mutate(cloneJson(previewedPayload))}
                aria-label={editingDeployment ? '确认并创建更新任务' : '确认并创建任务'}
              >
                {editingDeployment ? '确认并创建更新任务' : '确认并创建任务'}
              </Button>
            )}
          </div>
        </Form>
      </Drawer>
      <Drawer title={`${logsFor?.name ?? ''} 日志`} width={760} open={Boolean(logsFor)} onClose={() => setLogsFor(null)}>
        <QueryState loading={logs.isLoading} error={logs.error}><LogViewer value={logs.data?.logs ?? ''} filename={`${logsFor?.name}.log`} /></QueryState>
      </Drawer>
      <Modal
        title={`卸载服务 ${uninstallTarget?.name ?? ''}`}
        open={Boolean(uninstallTarget)}
        onCancel={() => {
          if (!discoveredUninstallPending) closeDiscoveredUninstall(uninstallTarget?.id)
        }}
        onOk={() => uninstallTarget && runDeploymentAction(uninstallTarget, 'delete')}
        okText="确认卸载"
        cancelText="取消"
        confirmLoading={discoveredUninstallPending}
        okButtonProps={{
          danger: true,
          disabled: Boolean(uninstallTargetPendingAction)
            || !uninstallTarget?.container_name
            || uninstallConfirmation !== uninstallTarget.container_name,
        }}
        cancelButtonProps={{ disabled: discoveredUninstallPending }}
        destroyOnHidden
        keyboard={!discoveredUninstallPending}
        closable={!discoveredUninstallPending}
        maskClosable={!discoveredUninstallPending}
      >
        {uninstallTarget && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Alert
              type="error"
              showIcon
              message="高风险操作"
              description="将删除此已发现服务的容器、部署记录和网关路由；保存的部署参数无法重建此服务。模型文件仍会保留。"
            />
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="容器名称">
                {uninstallTarget.container_name}
              </Descriptions.Item>
              <Descriptions.Item label="镜像">
                {uninstallTarget.image ?? '未知'}
              </Descriptions.Item>
              <Descriptions.Item label="端点">
                {uninstallTarget.endpoint_url}
              </Descriptions.Item>
            </Descriptions>
            <label htmlFor="deployment-uninstall-confirmation">输入容器名称确认</label>
            <Input
              id="deployment-uninstall-confirmation"
              value={uninstallConfirmation}
              autoComplete="off"
              placeholder={uninstallTarget.container_name ?? ''}
              onChange={(event) => setUninstallConfirmation(event.target.value)}
            />
          </Space>
        )}
      </Modal>
    </div>
  )
}
