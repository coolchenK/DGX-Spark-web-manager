import type { Deployment, ModelAsset } from '../api/types'


export interface DeploymentFormValues {
  name: string
  model_id: string
  model_path: string
  api_model_name: string
  route_alias?: string
  runtime: 'vllm' | 'sglang'
  image: string
  port: number
  context_length: number
  memory_fraction: number
  max_concurrency: number
  max_batched_tokens?: number
  quantization?: 'auto' | 'awq' | 'gptq' | 'fp8' | 'bitsandbytes' | 'marlin'
  trust_remote_code: boolean
}


function commandValue(command: string[], name: string): string | undefined {
  const index = command.indexOf(name)
  return index >= 0 ? command[index + 1] : undefined
}


function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
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
    deployment.runtime === 'sglang' ? '--context-length' : '--max-model-len',
  )
  const memoryValue = commandValue(
    command,
    deployment.runtime === 'sglang' ? '--mem-fraction-static' : '--gpu-memory-utilization',
  )
  const concurrencyValue = commandValue(
    command,
    deployment.runtime === 'sglang' ? '--max-running-requests' : '--max-num-seqs',
  )
  const routeAlias = String(deployment.config.route_alias ?? saved.route_alias ?? '') || undefined
  const values: DeploymentFormValues = {
    name: deployment.name,
    model_id: deployment.model_id ?? model.id,
    model_path: model.local_path,
    api_model_name: deployment.api_model_name,
    route_alias: routeAlias,
    runtime: deployment.runtime as 'vllm' | 'sglang',
    image: deployment.image ?? '',
    port: deployment.port ?? 8100,
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
  }
  if (mode === 'clone') {
    values.name = `${values.name}-copy`
    values.api_model_name = `${values.api_model_name}-copy`
    values.port = Math.min(values.port + 1, 65535)
  }
  return values
}
