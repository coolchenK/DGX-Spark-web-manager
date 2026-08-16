import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type {
  Deployment,
  DeploymentRecommendation,
  ModelAsset,
  Provider,
  TaskRecord,
} from '../api/types'
import { DeploymentsPage } from './DeploymentsPage'


const GiB = 1024 ** 3

const models: ModelAsset[] = [
  {
    id: 'model-1',
    name: 'Qwen/Qwen-Test',
    alias: 'qwen-test',
    source: 'huggingface',
    repository_id: 'Qwen/Qwen-Test',
    revision: 'main',
    commit_hash: 'base-commit',
    local_path: '/models/qwen-test',
    format: 'safetensors',
    quantization: 'modelopt_fp4',
    parameter_count: '7B',
    size_bytes: 7 * GiB,
    status: 'available',
    capabilities: ['chat'],
    created_at: '2026-08-16T00:00:00Z',
    updated_at: '2026-08-16T00:00:00Z',
  },
  {
    id: 'model-2',
    name: 'Qwen/Qwen-Second',
    alias: null,
    source: 'huggingface',
    repository_id: 'Qwen/Qwen-Second',
    revision: 'main',
    commit_hash: 'second-commit',
    local_path: '/models/qwen-second',
    format: 'safetensors',
    quantization: null,
    parameter_count: '3B',
    size_bytes: 3 * GiB,
    status: 'available',
    capabilities: ['chat'],
    created_at: '2026-08-16T00:00:00Z',
    updated_at: '2026-08-16T00:00:00Z',
  },
]

const providers: Provider[] = [
  {
    id: 'provider-1',
    name: 'Online advisor',
    base_url: 'https://provider.invalid/v1',
    default_model: 'advisor',
    api_key_masked: 'sk-***',
    timeout_seconds: 60,
    headers: {},
    enabled: true,
    last_test_status: 'healthy',
    last_tested_at: '2026-08-16T00:00:00Z',
    created_at: '2026-08-16T00:00:00Z',
    updated_at: '2026-08-16T00:00:00Z',
  },
]

function recommendationFixture(
  overrides: Partial<DeploymentRecommendation> = {},
): DeploymentRecommendation {
  return {
    status: 'complete',
    generated_at: '2026-08-16T12:00:00Z',
    model_id: 'model-1',
    runtime: 'vllm',
    image_digest: `sha256:${'b'.repeat(64)}`,
    evidence_hash: 'a'.repeat(64),
    fields: {
      context_length: {
        value: 16_384,
        source: 'device_rule',
        confidence: 'high',
        reason: '模型卡建议 32768，按 DGX Spark 当前统一内存调整为 16384',
        warning: null,
      },
      memory_fraction: {
        value: 0.72,
        source: 'device_rule',
        confidence: 'high',
        reason: '为操作系统和管理器保留统一内存',
        warning: null,
      },
      max_concurrency: {
        value: 4,
        source: 'runtime_default',
        confidence: 'medium',
        reason: '运行时保守并发值',
        warning: null,
      },
      max_batched_tokens: {
        value: 8192,
        source: 'local_config',
        confidence: 'medium',
        reason: '由本地模型配置计算',
        warning: null,
      },
      quantization: {
        value: 'modelopt_fp4',
        source: 'local_config',
        confidence: 'high',
        reason: '本地权重为 ModelOpt FP4',
        warning: null,
      },
    },
    generation_defaults: {
      temperature: {
        value: 0.6,
        source: 'model_card',
        confidence: 'high',
        reason: '模型卡明确推荐采样温度',
        warning: null,
      },
      top_p: {
        value: 0.95,
        source: 'ai',
        confidence: 'medium',
        reason: 'AI 根据模型卡未明确部分补充',
        warning: null,
      },
    },
    resource_snapshot: {
      total_bytes: 128 * GiB,
      available_bytes: 96 * GiB,
      reserved_bytes: 16 * GiB,
    },
    resource_estimate: {
      total_bytes: 128 * GiB,
      available_bytes: 96 * GiB,
      reserved_bytes: 16 * GiB,
      weight_bytes: 7 * GiB,
      draft_weight_bytes: 2 * GiB,
      kv_cache_bytes: 12 * GiB,
      runtime_overhead_bytes: 5 * GiB,
      required_bytes: 26 * GiB,
      decision: 'ok',
      confidence: 'high',
      reasons: ['预计保留 70 GiB 可用统一内存'],
    },
    runtime_capabilities: {
      runtime: 'vllm',
      image: 'vllm/vllm-openai:v0.27.1',
      image_digest: `sha256:${'b'.repeat(64)}`,
      source: 'probe',
      generation_defaults: ['temperature', 'top_p'],
      quantization_methods: ['auto', 'modelopt_fp4', 'fp8'],
      quantization_mapping: {},
      speculative_methods: ['draft_model', 'eagle3'],
      method_mapping: { draft_model: 'draft_model', eagle3: 'eagle3' },
      speculative_transport: 'json',
      warnings: [],
    },
    draft_candidates: [
      {
        model_id: 'draft-compatible',
        name: 'Target-EAGLE3',
        repository_id: 'org/Target-EAGLE3',
        method: 'eagle3',
        status: 'compatible',
        reasons: ['模型配置明确声明目标为 Qwen/Qwen-Test'],
        size_bytes: 2 * GiB,
        estimated_total_bytes: 28 * GiB,
      },
      {
        model_id: 'draft-review',
        name: 'Review-Draft',
        repository_id: 'org/Review-Draft',
        method: 'draft_model',
        status: 'review',
        reasons: ['Tokenizer 未发现冲突，但缺少明确配对声明'],
        size_bytes: GiB,
        estimated_total_bytes: 27 * GiB,
      },
      {
        model_id: 'draft-incompatible',
        name: 'Wrong-Tokenizer',
        repository_id: 'org/Wrong-Tokenizer',
        method: 'draft_model',
        status: 'incompatible',
        reasons: ['Tokenizer 词表哈希不一致'],
        size_bytes: GiB,
        estimated_total_bytes: null,
      },
    ],
    warnings: [],
    ...overrides,
  }
}

function previewFixture() {
  return {
    runtime: 'vllm',
    image: 'vllm/vllm-openai:v0.27.1',
    container_name: 'dgx-qwen-test',
    port: 8100,
    route_alias: 'qwen-test',
    compatibility: {
      compatible: true,
      architectures: ['Qwen2ForCausalLM'],
      reasons: [],
    },
    estimated_disk_bytes: 7 * GiB,
    estimated_memory_bytes: 26 * GiB,
    resource_estimate: recommendationFixture().resource_estimate,
    generation_defaults: { temperature: 0.6, top_p: 0.95 },
    mounts: {
      base: { host_path: '/models/qwen-test', container_path: '/models/base' },
      draft: { host_path: '/models/draft', container_path: '/models/draft' },
    },
    command: ['python', '-m', 'vllm.entrypoints.openai.api_server'],
    operations: ['创建受管理容器', '检查 /v1/models 并注册网关路由'],
    api_example: 'client.chat.completions.create(model="qwen-test")',
    rollback: '失败时删除新容器并保留模型文件',
  }
}

const task: TaskRecord = {
  id: 'task-1',
  type: 'deployment.create',
  status: 'queued',
  title: '部署 qwen-test',
  progress: 0,
  completed_bytes: 0,
  total_bytes: null,
  speed_bytes_per_second: null,
  eta_seconds: null,
  result: {},
  error: null,
  log: '',
  created_at: '2026-08-16T00:00:00Z',
  updated_at: '2026-08-16T00:00:00Z',
  started_at: null,
  finished_at: null,
}

const existingDeployment: Deployment = {
  id: 'deployment-1',
  name: 'qwen-production',
  model_id: 'model-1',
  runtime: 'vllm',
  container_id: 'container-1',
  container_name: 'dgx-qwen-production',
  endpoint_url: 'http://127.0.0.1:8100',
  api_model_name: 'qwen-production',
  status: 'running',
  health: 'healthy',
  managed: true,
  image: 'vllm/vllm-openai:v0.27.1',
  port: 8100,
  config: {
    route_alias: 'qwen-shared',
    spec: {
      name: 'qwen-production',
      model_id: 'model-1',
      model_path: '/models/qwen-test',
      api_model_name: 'qwen-production',
      route_alias: 'qwen-shared',
      runtime: 'vllm',
      image: 'vllm/vllm-openai:v0.27.1',
      port: 8100,
      context_length: 8192,
      memory_fraction: 0.68,
      max_concurrency: 2,
      max_batched_tokens: 4096,
      quantization: 'modelopt_fp4',
      trust_remote_code: false,
      generation_defaults: { temperature: 0.2, top_p: 0.8 },
      speculative: {
        draft_model_id: 'draft-review',
        method: 'draft_model',
        num_speculative_tokens: 5,
        manual_review_acknowledged: true,
      },
      recommendation: {
        generated_at: '2026-08-15T12:00:00Z',
        evidence_hash: 'c'.repeat(64),
        provider_id: 'provider-1',
        resource_snapshot: {
          total_bytes: 128 * GiB,
          available_bytes: 90 * GiB,
          reserved_bytes: 16 * GiB,
        },
        modified_fields: ['context_length'],
        sources: { context_length: 'device_rule' },
      },
      resource_warning_acknowledged: true,
    },
  },
  capabilities: ['chat'],
  last_checked_at: '2026-08-16T00:00:00Z',
}

interface ApiFixtureOptions {
  deployments?: Deployment[]
  recommendations?: (
    path: string,
    body: Record<string, unknown>,
  ) => Promise<DeploymentRecommendation>
}

function renderDeploymentsPage(options: ApiFixtureOptions = {}) {
  const user = userEvent.setup()
  const getSpy = vi.spyOn(api, 'get').mockImplementation(async <T,>(path: string): Promise<T> => {
    if (path === '/api/deployments') return (options.deployments ?? []) as T
    if (path === '/api/models') return models as T
    if (path === '/api/providers') return providers as T
    if (path.includes('/logs')) return { logs: '' } as T
    throw new Error(`Unexpected GET ${path}`)
  })
  const postSpy = vi.spyOn(api, 'post').mockImplementation(async <T,>(
    path: string,
    body?: unknown,
  ): Promise<T> => {
    if (path.startsWith('/api/deployments/recommendations')) {
      const recommendation = options.recommendations
        ? await options.recommendations(path, body as Record<string, unknown>)
        : recommendationFixture({
            model_id: String((body as Record<string, unknown>).model_id),
            runtime: String((body as Record<string, unknown>).runtime) as 'vllm' | 'sglang',
            runtime_capabilities: {
              ...recommendationFixture().runtime_capabilities,
              runtime: String((body as Record<string, unknown>).runtime) as 'vllm' | 'sglang',
              image: String((body as Record<string, unknown>).image),
            },
          })
      return recommendation as T
    }
    if (path.startsWith('/api/deployments/preview')) return previewFixture() as T
    if (path === '/api/deployments') return task as T
    if (/\/api\/deployments\/[^/]+\/(start|stop|restart|delete)$/.test(path)) return task as T
    throw new Error(`Unexpected POST ${path}`)
  })
  const patchSpy = vi.spyOn(api, 'patch').mockResolvedValue(task)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/deployments']}>
        <DeploymentsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { user, getSpy, postSpy, patchSpy, queryClient }
}

async function openCreateAndSelectModel(
  user: ReturnType<typeof userEvent.setup>,
  modelName = 'Qwen/Qwen-Test',
) {
  await user.click(await screen.findByRole('button', { name: /新建部署/ }))
  await user.click(screen.getByLabelText('模型'))
  await user.click(await screen.findByText(modelName))
}

async function goToRecommendationStep(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '下一步' }))
  expect(await screen.findByRole('heading', { name: '推荐配置' })).toBeInTheDocument()
}

async function goToDraftStep(user: ReturnType<typeof userEvent.setup>) {
  await goToRecommendationStep(user)
  await screen.findByDisplayValue('16384')
  await user.click(screen.getByRole('button', { name: '下一步' }))
  expect(await screen.findByRole('heading', { name: 'Draft Model' })).toBeInTheDocument()
}


describe('DeploymentsPage assisted deployment wizard', () => {
  it('auto-prefills recommendations, shows their source, and only exposes probed quantization methods', async () => {
    const { user, postSpy } = renderDeploymentsPage()
    await openCreateAndSelectModel(user)
    await goToRecommendationStep(user)

    await waitFor(() => expect(postSpy.mock.calls.some(([path]) => String(path).startsWith('/api/deployments/recommendations'))).toBe(true))
    expect(await screen.findByDisplayValue('16384')).toBeInTheDocument()
    expect(screen.getAllByText('DGX Spark 资源调整').length).toBeGreaterThan(0)
    expect(await screen.findByLabelText('Temperature')).toHaveValue('0.6')
    await user.click(screen.getByLabelText('量化加载方式'))
    expect((await screen.findAllByText('NVFP4 / ModelOpt FP4')).length).toBeGreaterThan(0)
    expect(screen.queryByText('awq')).not.toBeInTheDocument()
  })

  it('preserves a manually edited field across refresh and force reapply restores the recommendation', async () => {
    let refreshCount = 0
    const { user } = renderDeploymentsPage({
      recommendations: async (path) => {
        if (path.includes('refresh_ai=true')) {
          refreshCount += 1
          return recommendationFixture({
            fields: {
              ...recommendationFixture().fields,
              context_length: {
                ...recommendationFixture().fields.context_length,
                value: 8192,
              },
            },
          })
        }
        return recommendationFixture()
      },
    })
    await openCreateAndSelectModel(user)
    await goToRecommendationStep(user)

    const context = await screen.findByLabelText('上下文长度')
    await user.clear(context)
    await user.type(context, '12288')
    await user.click(screen.getByRole('button', { name: '重新分析' }))

    await waitFor(() => expect(refreshCount).toBe(1))
    expect(context).toHaveValue('12288')
    expect(screen.getByText('已手动修改')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '重新应用全部建议' }))
    expect(context).toHaveValue('8192')
    expect(screen.queryByText('已手动修改')).not.toBeInTheDocument()
  })

  it('keeps deterministic values visible when AI is partial and retries AI analysis', async () => {
    let refreshCount = 0
    const partial = recommendationFixture({
      status: 'partial',
      warnings: ['AI provider timeout; deterministic recommendations remain available'],
    })
    const { user } = renderDeploymentsPage({
      recommendations: async (path) => {
        if (path.includes('refresh_ai=true')) {
          refreshCount += 1
          return recommendationFixture()
        }
        return partial
      },
    })
    await openCreateAndSelectModel(user)
    await goToRecommendationStep(user)

    expect(await screen.findByText('AI 补充不可用，已使用确定性建议')).toBeInTheDocument()
    expect(screen.getByDisplayValue('16384')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '重新分析' }))
    await waitFor(() => expect(refreshCount).toBe(1))
    await waitFor(() => expect(screen.queryByText('AI 补充不可用，已使用确定性建议')).not.toBeInTheDocument())
  })

  it('shows compatible drafts by default and disables incompatible drafts in advanced mode', async () => {
    const { user } = renderDeploymentsPage()
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)

    expect(screen.getByText('Target-EAGLE3')).toBeInTheDocument()
    expect(screen.queryByText('Wrong-Tokenizer')).not.toBeInTheDocument()
    await user.click(screen.getByLabelText('显示待确认及不兼容模型'))
    expect(screen.getByText('Wrong-Tokenizer')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Wrong-Tokenizer/ })).toBeDisabled()
  })

  it('requires review acknowledgement before previewing a review draft', async () => {
    const { user, postSpy } = renderDeploymentsPage()
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)
    await user.click(screen.getByLabelText('显示待确认及不兼容模型'))
    await user.click(screen.getByRole('radio', { name: /Review-Draft/ }))
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))

    expect(await screen.findByText('请确认 Draft Model 配对风险')).toBeInTheDocument()
    expect(postSpy.mock.calls.some(([path]) => String(path).startsWith('/api/deployments/preview'))).toBe(false)

    await user.click(screen.getByLabelText('我已核对该 Draft Model 的兼容性风险'))
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))
    expect(await screen.findByRole('heading', { name: '部署预览' })).toBeInTheDocument()
  })

  it('requires resource warning acknowledgement before preview', async () => {
    const warning = recommendationFixture({
      resource_estimate: {
        ...recommendationFixture().resource_estimate,
        decision: 'warning',
        reasons: ['预计占用超过当前可用统一内存'],
      },
    })
    const { user, postSpy } = renderDeploymentsPage({ recommendations: async () => warning })
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))

    expect(await screen.findByText('请确认统一内存资源警告')).toBeInTheDocument()
    expect(postSpy.mock.calls.some(([path]) => String(path).startsWith('/api/deployments/preview'))).toBe(false)

    await user.click(screen.getByLabelText('我了解资源不足可能导致部署失败'))
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))
    expect(await screen.findByRole('heading', { name: '部署预览' })).toBeInTheDocument()
  })

  it('clears draft selection after a runtime change while retaining generic manual values', async () => {
    const { user } = renderDeploymentsPage()
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)
    await user.click(screen.getByRole('radio', { name: /Target-EAGLE3/ }))
    await user.click(screen.getByRole('button', { name: '上一步' }))
    const context = screen.getByLabelText('上下文长度')
    await user.clear(context)
    await user.type(context, '12288')
    await user.click(screen.getByRole('button', { name: '上一步' }))
    await user.click(screen.getByText('SGLang'))
    await user.click(screen.getByRole('button', { name: '下一步' }))

    expect(await screen.findByLabelText('上下文长度')).toHaveValue('12288')
    await user.click(screen.getByRole('button', { name: '下一步' }))
    expect(screen.getByRole('radio', { name: /Target-EAGLE3/ })).not.toBeChecked()
  })

  it('clears edits, draft and acknowledgements after changing the base model', async () => {
    const warning = recommendationFixture({
      resource_estimate: {
        ...recommendationFixture().resource_estimate,
        decision: 'warning',
      },
    })
    const { user } = renderDeploymentsPage({
      recommendations: async (_path, body) => recommendationFixture({
        ...warning,
        model_id: String(body.model_id),
        fields: {
          ...warning.fields,
          context_length: {
            ...warning.fields.context_length,
            value: body.model_id === 'model-2' ? 8192 : 16_384,
          },
        },
      }),
    })
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)
    await user.click(screen.getByRole('radio', { name: /Target-EAGLE3/ }))
    await user.click(screen.getByLabelText('我了解资源不足可能导致部署失败'))
    await user.click(screen.getByRole('button', { name: '上一步' }))
    const context = screen.getByLabelText('上下文长度')
    await user.clear(context)
    await user.type(context, '12288')
    await user.click(screen.getByRole('button', { name: '上一步' }))
    await user.click(screen.getByLabelText('模型'))
    await user.click(await screen.findByText('Qwen/Qwen-Second'))
    await user.click(screen.getByRole('button', { name: '下一步' }))

    await waitFor(() => expect(screen.getByLabelText('上下文长度')).toHaveValue('8192'))
    expect(screen.queryByText('已手动修改')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '下一步' }))
    expect(screen.getByRole('radio', { name: /Target-EAGLE3/ })).not.toBeChecked()
    expect(screen.getByLabelText('我了解资源不足可能导致部署失败')).not.toBeChecked()
  })

  it('ignores a late recommendation from a previously selected model', async () => {
    let resolveFirst: ((value: DeploymentRecommendation) => void) | undefined
    const first = new Promise<DeploymentRecommendation>((resolve) => { resolveFirst = resolve })
    const { user } = renderDeploymentsPage({
      recommendations: async (_path, body) => {
        if (body.model_id === 'model-1') return first
        return recommendationFixture({
          model_id: 'model-2',
          fields: {
            ...recommendationFixture().fields,
            context_length: {
              ...recommendationFixture().fields.context_length,
              value: 8192,
            },
          },
        })
      },
    })
    await openCreateAndSelectModel(user)
    await waitFor(() => expect(resolveFirst).toBeDefined())
    await user.click(screen.getByLabelText('模型'))
    await user.click(await screen.findByText('Qwen/Qwen-Second'))
    await goToRecommendationStep(user)
    await waitFor(() => expect(screen.getByLabelText('上下文长度')).toHaveValue('8192'))

    resolveFirst?.(recommendationFixture())
    await waitFor(() => expect(screen.getByLabelText('上下文长度')).toHaveValue('8192'))
  })

  it('restores edit values as protected edits and includes deployment id in preview and patch payload', async () => {
    const { user, postSpy, patchSpy } = renderDeploymentsPage({ deployments: [existingDeployment] })
    await user.click(await screen.findByRole('button', { name: '编辑部署参数' }))
    await goToRecommendationStep(user)

    expect(await screen.findByLabelText('上下文长度')).toHaveValue('8192')
    expect(screen.getAllByText('已手动修改').length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: '下一步' }))
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))
    await screen.findByRole('heading', { name: '部署预览' })

    const previewCall = postSpy.mock.calls.find(([path]) => String(path).startsWith('/api/deployments/preview'))
    expect(previewCall?.[0]).toBe('/api/deployments/preview?deployment_id=deployment-1')

    await user.click(screen.getByRole('button', { name: '确认并创建更新任务' }))
    await waitFor(() => expect(patchSpy).toHaveBeenCalledTimes(1))
    expect(patchSpy.mock.calls[0][0]).toBe('/api/deployments/deployment-1')
    expect(patchSpy.mock.calls[0][1]).toMatchObject({
      context_length: 8192,
      generation_defaults: { temperature: 0.2, top_p: 0.8 },
      speculative: {
        draft_model_id: 'draft-review',
        manual_review_acknowledged: true,
      },
      recommendation: {
        provider_id: 'provider-1',
        modified_fields: expect.arrayContaining(['context_length']),
      },
    })
  })

  it('restores clone identity fields without entering edit mode', async () => {
    const { user } = renderDeploymentsPage({ deployments: [existingDeployment] })
    await user.click(await screen.findByRole('button', { name: '克隆部署' }))

    expect(screen.getByLabelText('部署名称')).toHaveValue('qwen-production-copy')
    expect(screen.getByLabelText('实例模型名称')).toHaveValue('qwen-production-copy')
    expect(screen.getByLabelText('主机端口')).toHaveValue('8101')
    expect(screen.getByText('新建模型部署')).toBeInTheDocument()
  })

  it('submits the final create payload with recommendation provenance and selected draft settings', async () => {
    const { user, postSpy } = renderDeploymentsPage()
    await openCreateAndSelectModel(user)
    await goToDraftStep(user)
    await user.click(screen.getByRole('radio', { name: /Target-EAGLE3/ }))
    await user.click(screen.getByRole('button', { name: '生成部署预览' }))
    await screen.findByRole('heading', { name: '部署预览' })
    await user.click(screen.getByRole('button', { name: '确认并创建任务' }))

    await waitFor(() => {
      expect(postSpy.mock.calls.some(([path]) => path === '/api/deployments')).toBe(true)
    })
    const createCall = postSpy.mock.calls.find(([path]) => path === '/api/deployments')
    expect(createCall?.[1]).toMatchObject({
      model_id: 'model-1',
      context_length: 16_384,
      quantization: 'modelopt_fp4',
      generation_defaults: { temperature: 0.6, top_p: 0.95 },
      speculative: {
        draft_model_id: 'draft-compatible',
        method: 'eagle3',
        manual_review_acknowledged: false,
      },
      recommendation: {
        generated_at: '2026-08-16T12:00:00Z',
        evidence_hash: 'a'.repeat(64),
        provider_id: 'provider-1',
        resource_snapshot: {
          total_bytes: 128 * GiB,
          available_bytes: 96 * GiB,
          reserved_bytes: 16 * GiB,
        },
        sources: expect.objectContaining({
          context_length: 'device_rule',
          'generation_defaults.temperature': 'model_card',
        }),
      },
      resource_warning_acknowledged: false,
    })
  })
})
