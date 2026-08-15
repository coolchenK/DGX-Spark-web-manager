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
  operation: string
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
}

export interface ManagerSettings {
  huggingface: { token_configured: boolean; cache_dir: string }
  models: { roots: string[] }
  runtimes: { vllm: string[]; sglang: string[] }
}
