import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Grid } from 'antd'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type { Deployment, OpsSession, OpsSessionSummary, Provider, TaskRecord } from '../api/types'
import { DiagnosticsPage } from './DiagnosticsPage'


const provider: Provider = {
  id: 'provider-1',
  name: 'Operations AI',
  base_url: 'https://provider.example/v1',
  default_model: 'ops-model',
  api_key_masked: 'sk-***',
  timeout_seconds: 60,
  headers: {},
  enabled: true,
  last_test_status: 'healthy',
  last_test_result: {},
  last_tested_at: '2026-08-18T00:00:00Z',
  created_at: '2026-08-18T00:00:00Z',
  updated_at: '2026-08-18T00:00:00Z',
}

const deployment: Deployment = {
  id: 'deployment-1',
  name: 'Gateway',
  model_id: null,
  runtime: 'vllm',
  container_id: null,
  container_name: null,
  endpoint_url: 'http://127.0.0.1:3000/v1',
  api_model_name: 'gateway-model',
  status: 'running',
  health: 'healthy',
  managed: true,
  image: null,
  port: 3000,
  config: {},
  capabilities: [],
  last_checked_at: null,
}

const summary: OpsSessionSummary = {
  id: 'session-1',
  title: '修复网关模型列表',
  provider_id: provider.id,
  provider_name: provider.name,
  deployment_id: deployment.id,
  deployment_name: deployment.name,
  status: 'approval_required',
  requested_by: 'admin',
  created_at: '2026-08-18T00:00:00Z',
  updated_at: '2026-08-18T00:02:00Z',
}

const task: TaskRecord = {
  id: 'task-1',
  type: 'ops.respond',
  status: 'queued',
  title: 'AI 运维响应',
  progress: 0,
  completed_bytes: 0,
  total_bytes: null,
  speed_bytes_per_second: null,
  eta_seconds: null,
  result: {},
  error: null,
  log: '',
  created_at: '2026-08-18T00:00:00Z',
  updated_at: '2026-08-18T00:00:00Z',
  started_at: null,
  finished_at: null,
}

function renderPage(getImpl: (path: string) => unknown) {
  vi.spyOn(Grid, 'useBreakpoint').mockReturnValue({ md: true })
  vi.spyOn(api, 'get').mockImplementation(async (path) => getImpl(path) as never)
  const postSpy = vi.spyOn(api, 'post')
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/diagnostics']}>
        <DiagnosticsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { postSpy, user: userEvent.setup() }
}


describe('DiagnosticsPage operations sessions', () => {
  it('creates a session before queuing its first message', async () => {
    const { postSpy, user } = renderPage((path) => {
      if (path === '/api/providers') return [provider]
      if (path === '/api/deployments') return [deployment]
      if (path === '/api/diagnostics/sessions') return []
      throw new Error(`Unexpected GET ${path}`)
    })
    postSpy
      .mockResolvedValueOnce({ ...summary, status: 'active' })
      .mockResolvedValueOnce(task)

    await user.click(await screen.findByLabelText('在线 AI 服务'))
    await user.click(await screen.findByText('Operations AI · ops-model'))
    await user.type(screen.getByLabelText('运维请求'), '检查网关无法获取模型列表的问题')
    await user.click(screen.getByRole('button', { name: '发送请求' }))

    await waitFor(() => expect(postSpy).toHaveBeenCalledTimes(2))
    expect(postSpy.mock.calls[0]).toEqual([
      '/api/diagnostics/sessions',
      {
        provider_id: provider.id,
        deployment_id: undefined,
        title: '检查网关无法获取模型列表的问题',
      },
    ])
    expect(postSpy.mock.calls[1]).toEqual([
      '/api/diagnostics/sessions/session-1/messages',
      { content: '检查网关无法获取模型列表的问题' },
    ])
  })

  it('shows read-only tool evidence and exact shell approval commands', async () => {
    const session: OpsSession = {
      ...summary,
      messages: [
        {
          id: 'message-1',
          session_id: summary.id,
          role: 'user',
          content: '检查并修复网关',
          metadata: {},
          operation_plan_id: null,
          created_at: '2026-08-18T00:00:00Z',
          updated_at: '2026-08-18T00:00:00Z',
        },
        {
          id: 'message-2',
          session_id: summary.id,
          role: 'assistant',
          content: '需要批准后执行修复。',
          metadata: { action: 'plan' },
          operation_plan_id: 'plan-1',
          created_at: '2026-08-18T00:02:00Z',
          updated_at: '2026-08-18T00:02:00Z',
        },
      ],
      tool_runs: [
        {
          id: 'tool-1',
          session_id: summary.id,
          tool_name: 'systemd.status',
          risk: 'read_only',
          status: 'succeeded',
          arguments: { unit: 'dgx-manager.service' },
          result: { active: false },
          agent_job_id: null,
          error: null,
          started_at: '2026-08-18T00:01:00Z',
          finished_at: '2026-08-18T00:01:01Z',
          created_at: '2026-08-18T00:01:00Z',
          updated_at: '2026-08-18T00:01:01Z',
        },
      ],
      plans: [
        {
          id: 'plan-1',
          provider_id: provider.id,
          deployment_id: deployment.id,
          summary: '重启管理服务',
          diagnosis: '服务状态异常。',
          risk: 'high',
          steps: [{
            id: 'step-1',
            operation: 'shell',
            command: 'sudo systemctl restart dgx-manager',
            cwd: '/',
            timeout: 60,
            deployment_id: deployment.id,
            reason: '恢复模型列表接口',
            impact: '管理面板短暂不可用',
            rollback: 'sudo systemctl start dgx-manager',
            executable: true,
          }],
          status: 'pending',
          requested_by: 'admin',
          approved_by: null,
          approved_at: null,
          result: {},
          created_at: '2026-08-18T00:02:00Z',
          updated_at: '2026-08-18T00:02:00Z',
        },
      ],
    }
    renderPage((path) => {
      if (path === '/api/providers') return [provider]
      if (path === '/api/deployments') return [deployment]
      if (path === '/api/diagnostics/sessions') return [summary]
      if (path === '/api/diagnostics/sessions/session-1') return session
      throw new Error(`Unexpected GET ${path}`)
    })

    await userEvent.click(await screen.findByRole('button', { name: /修复网关模型列表/ }))

    expect(await screen.findByText('自动执行 · 只读')).toBeInTheDocument()
    expect(screen.getByText('systemd.status')).toBeInTheDocument()
    const approval = screen.getByRole('article', { name: '操作审批：重启管理服务' })
    expect(within(approval).getByText('sudo systemctl restart dgx-manager')).toBeInTheDocument()
    expect(within(approval).getByRole('button', { name: '批准执行' })).toBeInTheDocument()
  })
})
