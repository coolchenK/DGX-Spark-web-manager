import type { Deployment, ModelAsset, RecommendationProvenance, RuntimeName } from '../api/types'


export interface GenerationDefaults {
  temperature?: number
  top_p?: number
  top_k?: number
  min_p?: number
  repetition_penalty?: number
  presence_penalty?: number
  frequency_penalty?: number
  max_tokens?: number
  stop?: string | string[]
}

export type QuantizationMethod =
  | 'auto'
  | 'awq'
  | 'gptq'
  | 'fp8'
  | 'bitsandbytes'
  | 'marlin'
  | 'gguf'
  | 'modelopt'
  | 'modelopt_fp4'
  | 'nvfp4_online'
  | 'compressed-tensors'

export interface SpeculativeSettings {
  draft_model_id: string
  method: 'draft_model' | 'dflash' | 'dspark' | 'eagle' | 'eagle3' | 'mtp'
  num_speculative_tokens?: number
  num_steps?: number
  eagle_top_k?: number
  num_draft_tokens?: number
  manual_review_acknowledged: boolean
}

export interface LlamaCppSettings {
  model_file?: string
  mmproj_file?: string
  gpu_layers: number | 'all'
  jinja: boolean
  continuous_batching: boolean
  mtp_enabled: boolean
  mtp_tokens: number
}


export interface DeploymentFormValues {
  name: string
  model_id: string
  model_path: string
  api_model_name: string
  route_alias?: string
  runtime: RuntimeName
  image: string
  port?: number
  context_length: number
  memory_fraction: number
  max_concurrency: number
  max_batched_tokens?: number
  quantization?: QuantizationMethod
  trust_remote_code: boolean
  generation_defaults: GenerationDefaults
  chat_template_kwargs?: Record<string, string | number | boolean>
  speculative?: SpeculativeSettings | null
  llama_cpp?: LlamaCppSettings | null
  recommendation?: RecommendationProvenance | null
  resource_warning_acknowledged: boolean
}


function commandValue(command: string[], name: string): string | undefined {
  const index = command.indexOf(name)
  return index >= 0 ? command[index + 1] : undefined
}


function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}


function objectValue(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}


function normalizeGenerationDefaults(value: unknown): GenerationDefaults {
  const saved = objectValue(value)
  if (!saved) return {}
  const normalized: GenerationDefaults = {}
  if (saved.temperature != null) normalized.temperature = saved.temperature as number
  if (saved.top_p != null) normalized.top_p = saved.top_p as number
  if (saved.top_k != null) normalized.top_k = saved.top_k as number
  if (saved.min_p != null) normalized.min_p = saved.min_p as number
  if (saved.repetition_penalty != null) {
    normalized.repetition_penalty = saved.repetition_penalty as number
  }
  if (saved.presence_penalty != null) {
    normalized.presence_penalty = saved.presence_penalty as number
  }
  if (saved.frequency_penalty != null) {
    normalized.frequency_penalty = saved.frequency_penalty as number
  }
  if (saved.max_tokens != null) normalized.max_tokens = saved.max_tokens as number
  if (saved.stop != null) {
    normalized.stop = Array.isArray(saved.stop)
      ? [...saved.stop] as string[]
      : saved.stop as string
  }
  return normalized
}


function normalizeChatTemplateKwargs(
  value: unknown,
): Record<string, string | number | boolean> | undefined {
  const saved = objectValue(value)
  if (!saved) return undefined
  const normalized = Object.fromEntries(
    Object.entries(saved).filter(([, item]) => (
      typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean'
    )),
  ) as Record<string, string | number | boolean>
  return Object.keys(normalized).length ? normalized : undefined
}


function normalizeSpeculativeSettings(value: unknown): SpeculativeSettings | null {
  const saved = objectValue(value)
  if (!saved || typeof saved.draft_model_id !== 'string') return null
  if (!['draft_model', 'dflash', 'dspark', 'eagle', 'eagle3', 'mtp'].includes(String(saved.method))) {
    return null
  }
  const normalized: SpeculativeSettings = {
    draft_model_id: saved.draft_model_id,
    method: saved.method as SpeculativeSettings['method'],
    manual_review_acknowledged: saved.manual_review_acknowledged === true,
  }
  if (saved.num_speculative_tokens != null) {
    normalized.num_speculative_tokens = saved.num_speculative_tokens as number
  }
  if (saved.num_steps != null) normalized.num_steps = saved.num_steps as number
  if (saved.eagle_top_k != null) normalized.eagle_top_k = saved.eagle_top_k as number
  if (saved.num_draft_tokens != null) {
    normalized.num_draft_tokens = saved.num_draft_tokens as number
  }
  return normalized
}


function normalizeLlamaCppSettings(value: unknown): LlamaCppSettings | null {
  const saved = objectValue(value)
  if (!saved) return null
  return {
    ...(typeof saved.model_file === 'string' ? { model_file: saved.model_file } : {}),
    ...(typeof saved.mmproj_file === 'string' ? { mmproj_file: saved.mmproj_file } : {}),
    gpu_layers: saved.gpu_layers === 'all' ? 'all' : finiteNumber(saved.gpu_layers, 0),
    jinja: saved.jinja !== false,
    continuous_batching: saved.continuous_batching !== false,
    mtp_enabled: saved.mtp_enabled === true,
    mtp_tokens: finiteNumber(saved.mtp_tokens, 3),
  }
}


function normalizeRecommendationProvenance(value: unknown): RecommendationProvenance | null {
  const saved = objectValue(value)
  const resourceSnapshot = objectValue(saved?.resource_snapshot)
  const sources = objectValue(saved?.sources)
  if (!saved || !resourceSnapshot || !sources) return null
  return {
    generated_at: saved.generated_at as string,
    evidence_hash: saved.evidence_hash as string,
    provider_id: saved.provider_id as string | null,
    resource_snapshot: {
      total_bytes: resourceSnapshot.total_bytes as number,
      available_bytes: resourceSnapshot.available_bytes as number,
      reserved_bytes: resourceSnapshot.reserved_bytes as number,
    },
    modified_fields: Array.isArray(saved.modified_fields)
      ? [...saved.modified_fields] as string[]
      : [],
    sources: { ...sources } as RecommendationProvenance['sources'],
  }
}


export function deploymentToFormValues(
  deployment: Deployment,
  model: ModelAsset,
  mode: 'edit' | 'clone',
): DeploymentFormValues {
  const saved = (deployment.config.spec ?? {}) as Partial<DeploymentFormValues>
  const command = Array.isArray(deployment.config.command)
    ? deployment.config.command.map(String)
    : []
  const contextValue = commandValue(
    command,
    deployment.runtime === 'sglang'
      ? '--context-length'
      : deployment.runtime === 'llama_cpp' ? '--ctx-size' : '--max-model-len',
  )
  const memoryValue = commandValue(
    command,
    deployment.runtime === 'sglang' ? '--mem-fraction-static' : '--gpu-memory-utilization',
  )
  const concurrencyValue = commandValue(
    command,
    deployment.runtime === 'sglang'
      ? '--max-running-requests'
      : deployment.runtime === 'llama_cpp' ? '--parallel' : '--max-num-seqs',
  )
  const routeAlias = String(deployment.config.route_alias ?? saved.route_alias ?? '') || undefined
  const values: DeploymentFormValues = {
    name: deployment.name,
    model_id: deployment.model_id ?? model.id,
    model_path: model.local_path,
    api_model_name: deployment.api_model_name,
    route_alias: routeAlias,
    runtime: deployment.runtime as RuntimeName,
    image: deployment.image ?? '',
    port: deployment.port ?? undefined,
    context_length: finiteNumber(saved.context_length ?? contextValue, 32768),
    memory_fraction: finiteNumber(saved.memory_fraction ?? memoryValue, 0.8),
    max_concurrency: finiteNumber(saved.max_concurrency ?? concurrencyValue, 8),
    max_batched_tokens: (
      saved.max_batched_tokens
      ?? finiteNumber(commandValue(command, '--max-num-batched-tokens'), 0)
    ) || undefined,
    quantization: saved.quantization
      ?? commandValue(command, '--quantization') as DeploymentFormValues['quantization'],
    trust_remote_code: saved.trust_remote_code ?? command.includes('--trust-remote-code'),
    generation_defaults: normalizeGenerationDefaults(saved.generation_defaults),
    chat_template_kwargs: normalizeChatTemplateKwargs(saved.chat_template_kwargs),
    speculative: normalizeSpeculativeSettings(saved.speculative),
    llama_cpp: normalizeLlamaCppSettings(saved.llama_cpp),
    recommendation: normalizeRecommendationProvenance(saved.recommendation),
    resource_warning_acknowledged: saved.resource_warning_acknowledged ?? false,
  }
  if (mode === 'clone') {
    values.name = `${values.name}-copy`
    values.api_model_name = `${values.api_model_name}-copy`
    values.port = undefined
    values.resource_warning_acknowledged = false
    if (values.speculative) {
      values.speculative = {
        ...values.speculative,
        manual_review_acknowledged: false,
      }
    }
  }
  return values
}
