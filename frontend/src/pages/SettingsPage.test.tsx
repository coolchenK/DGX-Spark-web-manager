import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { ApiError, api } from '../api/client'
import type { HistoryClearResult, ManagerSettings, SystemSnapshot } from '../api/types'
import { SettingsPage } from './SettingsPage'


const settings: ManagerSettings = {
  huggingface: { token_configured: false, cache_dir: '/models/.cache' },
  models: { roots: ['/models'] },
  runtimes: { vllm: ['vllm:test'], sglang: ['sglang:test'], llama_cpp: ['llama:test'] },
}

const system: SystemSnapshot = {
  hostname: 'spark',
  architecture: 'aarch64',
  os: 'Ubuntu',
  kernel: '6.14',
  cpu: { percent: 10, cores: 20 },
  memory: { total_bytes: 128, used_bytes: 64, available_bytes: 64 },
  disk: { total_bytes: 256, used_bytes: 64, free_bytes: 192 },
  gpus: [],
  uptime_seconds: 100,
}

const result: HistoryClearResult = {
  status: 'cleared',
  deleted: {
    failed_tasks: 2,
    operation_plans: 3,
    ops_sessions: 1,
    ops_messages: 4,
    ops_tool_runs: 5,
    audit_events: 6,
  },
}

function renderPage() {
  vi.spyOn(api, 'get').mockImplementation(async (path) => {
    if (path === '/api/settings') return settings as never
    if (path === '/api/system') return system as never
    throw new Error(`Unexpected GET ${path}`)
  })
  const deleteSpy = vi.spyOn(api, 'delete').mockResolvedValue(result)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  })
  const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/settings']}>
        <SettingsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { deleteSpy, invalidateSpy, user: userEvent.setup() }
}


describe('SettingsPage physical history clear', () => {
  it('requires the exact phrase and refreshes every affected view', async () => {
    const { deleteSpy, invalidateSpy, user } = renderPage()
    await user.click(await screen.findByRole('button', { name: '清除告警与诊断信息' }))
    const dialog = screen.getByRole('dialog', { name: '清除告警与诊断信息' })

    expect(within(dialog).getByText(/失败任务、诊断方案和 AI 运维会话/)).toBeInTheDocument()
    expect(within(dialog).getByText(/模型、部署、Provider、API Key/)).toBeInTheDocument()
    const confirm = within(dialog).getByLabelText('输入确认短语')
    const submit = within(dialog).getByRole('button', { name: '永久清除' })
    expect(submit).toBeDisabled()

    await user.type(confirm, '清除历史记')
    expect(submit).toBeDisabled()
    await user.type(confirm, '录')
    expect(submit).toBeEnabled()
    await user.click(submit)

    await waitFor(() => expect(deleteSpy).toHaveBeenCalledWith(
      '/api/settings/alerts-diagnostics-history',
      { confirmation: '清除历史记录' },
    ))
    for (const queryKey of [['tasks'], ['diagnostics'], ['ops-sessions'], ['audit'], ['system'], ['gateway-stats']]) {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey })
    }
  })

  it('keeps the dialog open and shows a conflict reason', async () => {
    const { deleteSpy, user } = renderPage()
    deleteSpy.mockRejectedValueOnce(new ApiError(409, '存在正在执行的运维任务'))
    await user.click(await screen.findByRole('button', { name: '清除告警与诊断信息' }))
    const dialog = screen.getByRole('dialog', { name: '清除告警与诊断信息' })
    await user.type(within(dialog).getByLabelText('输入确认短语'), '清除历史记录')

    await user.click(within(dialog).getByRole('button', { name: '永久清除' }))

    expect(await within(dialog).findByText('存在正在执行的运维任务')).toBeInTheDocument()
    expect(dialog).toBeInTheDocument()
    expect(within(dialog).getByLabelText('输入确认短语')).toHaveValue('清除历史记录')
  })
})
