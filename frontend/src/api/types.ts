export interface User {
  username: string
  role: string
}

export interface GpuMetric {
  name: string
  driver_version: string
  temperature_c: number | null
  power_w: number | null
  memory_used_bytes: number | null
  utilization_percent: number | null
}

export interface SystemSnapshot {
  hostname: string
  architecture: string
  os: string
  kernel: string
  cpu: { percent: number; cores: number }
  memory: { total_bytes: number; used_bytes: number; available_bytes: number }
  disk: { total_bytes: number; used_bytes: number; free_bytes: number }
  gpus: GpuMetric[]
  uptime_seconds: number
}

export interface ModelAsset {
  id: string
  name: string
  alias: string | null
  source: string
  repository_id: string | null
  revision: string | null
  commit_hash: string | null
  local_path: string
  format: string | null
  quantization: string | null
  parameter_count: string | null
  size_bytes: number
  status: string
  capabilities: string[]
  created_at: string
  updated_at: string
}

export interface ModelReference {
  deployment_id: string
  deployment_name: string
  usage: 'base' | 'draft' | 'legacy_path'
}

export interface ModelInUseDetail {
  code: 'model_in_use'
  references: ModelReference[]
}

export interface Deployment {
  id: string
  name: string
  model_id: string | null
  runtime: string
  container_id: string | null
  container_name: string | null
  endpoint_url: string
  api_model_name: string
  status: string
  health: string
  managed: boolean
  image: string | null
  port: number | null
  config: Record<string, unknown>
  capabilities: string[]
  last_checked_at: string | null
}

export interface TaskRecord {
  id: string
  type: string
  status: string
  title: string
  progress: number
  completed_bytes: number
  total_bytes: number | null
  speed_bytes_per_second: number | null
  eta_seconds: number | null
  result: Record<string, unknown>
  error: string | null
  log: string
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
}

export interface GatewayStats {
  total_requests: number
  failed_requests: number
  error_rate: number
  average_latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  requests_last_minute: number
  tokens_per_second: number
  active_requests: number
}

export interface ApiKeyRecord {
  id: string
  name: string
  prefix: string
  key?: string
  created_at: string
  last_used_at: string | null
  revoked_at: string | null
}

export interface Provider {
  id: string
  name: string
  base_url: string
  default_model: string
  api_key_masked: string
  timeout_seconds: number
  headers: Record<string, string>
  enabled: boolean
  last_test_status: string | null
  last_tested_at: string | null
  created_at: string
  updated_at: string
}

export interface OperationStep {
  id?: string
  operation: string
  command?: string
  cwd?: string
  timeout?: number
  deployment_id: string | null
  reason: string
  impact: string
  rollback: string
  executable: boolean
}

export interface OperationPlan {
  id: string
  provider_id: string | null
  deployment_id: string | null
  summary: string
  diagnosis: string
  risk: string
  steps: OperationStep[]
  status: string
  requested_by: string
  approved_by: string | null
  approved_at: string | null
  result: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface OpsSessionSummary {
  id: string
  title: string
  provider_id: string | null
  provider_name: string | null
  deployment_id: string | null
  deployment_name: string | null
  status: string
  requested_by: string
  created_at: string
  updated_at: string
}

export interface OpsMessage {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  metadata: Record<string, unknown>
  operation_plan_id: string | null
  created_at: string
  updated_at: string
}

export interface OpsToolRun {
  id: string
  session_id: string
  tool_name: string
  risk: string
  status: string
  arguments: Record<string, unknown>
  result: Record<string, unknown>
  agent_job_id: string | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
  updated_at: string
}

export interface OpsSession extends OpsSessionSummary {
  messages: OpsMessage[]
  tool_runs: OpsToolRun[]
  plans: OperationPlan[]
}

export interface AuditEvent {
  id: string
  actor: string
  action: string
  resource_type: string
  resource_id: string | null
  outcome: string
  source_ip: string | null
  details: Record<string, unknown>
  created_at: string
}

export interface HuggingFaceModel {
  id: string
  downloads: number
  likes: number
  pipeline_tag: string | null
  private: boolean
  gated: boolean
  last_modified: string | null
  tags: string[]
  spark_compatibility: {
    level: 'recommended' | 'compatible' | 'review'
    score: number
    reasons: string[]
  }
}

export interface HuggingFaceModelInfo {
  id: string
  sha: string
  pipeline_tag: string | null
  private: boolean
  gated: boolean
  tags: string[]
  siblings: Array<{ name: string; size: number | null }>
  total_size: number
  card_data: Record<string, unknown>
}

export interface ManagerSettings {
  huggingface: { token_configured: boolean; cache_dir: string }
  models: { roots: string[] }
  runtimes: { vllm: string[]; sglang: string[] }
}

export type RecommendationSource =
  | 'model_card'
  | 'local_config'
  | 'runtime_default'
  | 'device_rule'
  | 'ai'

export type RecommendationConfidence = 'high' | 'medium' | 'low'

export interface RecommendedValue<T = unknown> {
  value: T
  source: RecommendationSource
  confidence: RecommendationConfidence
  reason: string
  warning: string | null
}

export interface ResourceSnapshot {
  total_bytes: number
  available_bytes: number
  reserved_bytes: number
  deployments?: DeploymentResourceSnapshot[]
}

export interface DeploymentResourceSnapshot {
  id?: string
  runtime?: string
  status?: string
  health?: string
  total_bytes?: number
  available_bytes?: number
  reserved_bytes?: number
  weight_bytes?: number
  draft_weight_bytes?: number
  kv_cache_bytes?: number
  runtime_overhead_bytes?: number
  required_bytes?: number
  memory_bytes?: number
  size_bytes?: number
}

export interface ResourceEstimate {
  total_bytes: number
  available_bytes: number
  reserved_bytes: number
  weight_bytes: number
  draft_weight_bytes: number
  kv_cache_bytes: number
  runtime_overhead_bytes: number
  required_bytes: number
  decision: 'ok' | 'warning' | 'blocked'
  confidence: 'high' | 'low'
  reasons: string[]
}

export interface DraftCandidate {
  model_id: string
  name: string
  repository_id: string | null
  method: 'draft_model' | 'eagle' | 'eagle3' | 'mtp' | null
  status: 'compatible' | 'review' | 'incompatible'
  reasons: string[]
  size_bytes: number
  estimated_total_bytes: number | null
}

export interface RuntimeCapabilities {
  runtime: 'vllm' | 'sglang'
  image: string
  image_digest: string
  source: 'probe' | 'manifest'
  generation_defaults: string[]
  quantization_methods: string[]
  quantization_mapping: Record<string, string>
  speculative_methods: string[]
  method_mapping: Record<string, string>
  speculative_transport: 'json' | 'flags' | 'none'
  warnings: string[]
}

export interface DeploymentRecommendation {
  status: 'complete' | 'partial' | 'unavailable'
  generated_at: string
  model_id: string
  runtime: 'vllm' | 'sglang'
  image_digest: string | null
  evidence_hash: string | null
  fields: Record<string, RecommendedValue>
  generation_defaults: Record<string, RecommendedValue>
  resource_snapshot: Partial<ResourceSnapshot>
  resource_estimate: Partial<ResourceEstimate>
  runtime_capabilities: Partial<RuntimeCapabilities>
  draft_candidates: DraftCandidate[]
  warnings: string[]
}

export interface RecommendationProvenance {
  generated_at: string
  evidence_hash: string
  provider_id: string | null
  resource_snapshot: Omit<ResourceSnapshot, 'deployments'>
  modified_fields: string[]
  sources: Record<string, RecommendationSource>
}
