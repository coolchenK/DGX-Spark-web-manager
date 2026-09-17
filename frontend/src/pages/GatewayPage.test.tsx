import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type { ApiKeyRecord, GatewayStats, UpstreamGateway, UpstreamTestResult } from '../api/types'
import { GatewayPage } from './GatewayPage'


const stats: GatewayStats = {
  total_requests: 0,
  failed_requests: 0,
  error_rate: 0,
  average_latency_ms: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  requests_last_minute: 0,
  tokens_per_second: 0,
  throughput_window_seconds: 300,
  active_requests: 0,
}

const keys: ApiKeyRecord[] = []

function renderPage(
  upstream: UpstreamGateway,
  options: { testResult?: UpstreamTestResult; putImpl?: (path: string, body: unknown) => Promise<unknown> } = {},
) {
  vi.spyOn(api, 'get').mockImplementation((path: string) => {
    if (path === '/api/gateway/upstream') return Promise.resolve(upstream)
    if (path === '/api/gateway/stats') return Promise.resolve(stats)
    return Promise.resolve(keys)
  })
  const putSpy = vi.spyOn(api, 'put').mockImplementation(
    (options.putImpl ?? (() => Promise.resolve(upstream))) as never,
  )
  const postSpy = vi
    .spyOn(api, 'post')
    .mockResolvedValue((options.testResult ?? { status: 'ok', latency_ms: 0, model_count: 0, detail: null }) as never)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  })
  render(
    <ConfigProvider locale={zhCN}>
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/gateway']}>
          <GatewayPage />
        </MemoryRouter>
      </QueryClientProvider>
    </ConfigProvider>,
  )
  return { putSpy, postSpy, user: userEvent.setup() }
}

const unsetUpstream: UpstreamGateway = {
  base_url: null,
  api_key_configured: false,
  source: 'unset',
  enabled: false,
}

const configuredUpstream: UpstreamGateway = {
  base_url: 'https://upstream.example/v1',
  api_key_configured: true,
  source: 'database',
  enabled: true,
}


describe('GatewayPage upstream section', () => {
  it('invites configuration when no upstream is set', async () => {
    renderPage(unsetUpstream)

    expect(await screen.findByText('未配置上游网关')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '配置上游' })).toBeInTheDocument()
  })

  it('shows the configured base URL, key state and source', async () => {
    renderPage(configuredUpstream)

    expect(await screen.findByText('https://upstream.example/v1')).toBeInTheDocument()
    expect(screen.getByText('已配置')).toBeInTheDocument()
    expect(screen.getByText('面板配置')).toBeInTheDocument()
  })

  it('labels environment-sourced configuration', async () => {
    renderPage({ ...configuredUpstream, source: 'environment' })

    expect(await screen.findByText('环境变量')).toBeInTheDocument()
  })

  it('renders the discovered model count after a successful test', async () => {
    const { postSpy, user } = renderPage(configuredUpstream, {
      testResult: { status: 'ok', latency_ms: 128, model_count: 37, detail: null },
    })

    await user.click(await screen.findByRole('button', { name: /测试连接/ }))

    expect(postSpy).toHaveBeenCalledWith('/api/gateway/upstream/test')
    expect(await screen.findByText('连接正常 · 发现 37 个模型 · 128 ms')).toBeInTheDocument()
  })

  it('renders the bounded reason after a failed test', async () => {
    const { user } = renderPage(configuredUpstream, {
      testResult: {
        status: 'unavailable',
        latency_ms: 12,
        model_count: 0,
        detail: 'ConnectError: connection refused',
      },
    })

    await user.click(await screen.findByRole('button', { name: /测试连接/ }))

    expect(await screen.findByText('上游网关不可用')).toBeInTheDocument()
    expect(screen.getByText('ConnectError: connection refused')).toBeInTheDocument()
  })

  it('disables the test action when no upstream is configured', async () => {
    renderPage(unsetUpstream)

    expect(await screen.findByRole('button', { name: /测试连接/ })).toBeDisabled()
  })

  it('rejects a base URL without an http scheme', async () => {
    const { putSpy, user } = renderPage(unsetUpstream)

    await user.click(await screen.findByRole('button', { name: '配置上游' }))
    await user.type(await screen.findByLabelText('Base URL'), 'upstream.example/v1')
    await user.click(screen.getByRole('button', { name: /保\s*存/ }))

    expect(await screen.findByText('地址必须以 http:// 或 https:// 开头')).toBeInTheDocument()
    expect(putSpy).not.toHaveBeenCalled()
  })

  it('saves the base URL and omits an empty API key', async () => {
    const { putSpy, user } = renderPage(
      unsetUpstream,
      { putImpl: () => Promise.resolve(configuredUpstream) },
    )

    await user.click(await screen.findByRole('button', { name: '配置上游' }))
    await user.type(await screen.findByLabelText('Base URL'), 'https://upstream.example/v1')
    await user.click(screen.getByRole('button', { name: /保\s*存/ }))

    await waitFor(() => {
      expect(putSpy).toHaveBeenCalledWith('/api/gateway/upstream', {
        base_url: 'https://upstream.example/v1',
      })
    })
  })

  it('clears the configuration after confirmation', async () => {
    const { putSpy, user } = renderPage(configuredUpstream, {
      putImpl: () => Promise.resolve(unsetUpstream),
    })

    await user.click(await screen.findByRole('button', { name: '修改配置' }))
    await user.click(await screen.findByRole('button', { name: '清除配置' }))
    await user.click(await screen.findByRole('button', { name: /确\s*定/ }))

    await waitFor(() => {
      expect(putSpy).toHaveBeenCalledWith('/api/gateway/upstream', { base_url: null })
    })
  })
})
