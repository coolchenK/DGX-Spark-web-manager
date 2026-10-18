import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type { ApiKeyRecord } from '../api/types'
import { GatewayPage } from './GatewayPage'

const existingKey: ApiKeyRecord = {
  id: 'existing-key',
  name: '旧客户端',
  prefix: 'dgx_123',
  created_at: '2026-10-01T00:00:00Z',
  last_used_at: '2026-10-17T00:00:00Z',
  revoked_at: null,
}

function renderPage(status: 'ok' | 'unavailable' | Error = 'ok') {
  const getSpy = vi.spyOn(api, 'get').mockImplementation((path) => status instanceof Error ? Promise.reject(status) : Promise.resolve(
    path === '/api/magpie/status' ? { status, detail: status === 'unavailable' ? '服务暂时离线' : undefined } : [existingKey],
  ))
  const deleteSpy = vi.spyOn(api, 'delete').mockResolvedValue(undefined)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<ConfigProvider locale={zhCN}><QueryClientProvider client={queryClient}><GatewayPage /></QueryClientProvider></ConfigProvider>)
  return { getSpy, deleteSpy, user: userEvent.setup() }
}

describe('GatewayPage Magpie management', () => {
  it('embeds the gateway with the existing public API address and a full-page link', async () => {
    renderPage()
    const frame = await screen.findByTitle('Magpie API 网关')
    expect(frame).toHaveAttribute('src', '/magpie/?view=gateway')
    expect(screen.getByText(`${window.location.origin}/v1`)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '在新窗口打开' })).toHaveAttribute('href', '/magpie/?view=gateway')
    fireEvent.load(frame)
    expect(screen.getByText('已连接')).toBeInTheDocument()
    expect(screen.queryByText('正在加载 Magpie…')).not.toBeInTheDocument()
  })

  it('keeps Magpie errors recoverable without removing access-key management', async () => {
    const { getSpy, user } = renderPage('unavailable')
    expect(await screen.findByText('服务暂时离线')).toBeInTheDocument()
    expect(screen.queryByTitle('Magpie API 网关')).not.toBeInTheDocument()
    getSpy.mockImplementation((path) => Promise.resolve(path === '/api/magpie/status' ? { status: 'ok' } : [existingKey]))
    await user.click(screen.getByRole('button', { name: /重\s*试/ }))
    expect(await screen.findByTitle('Magpie API 网关')).toBeInTheDocument()
    expect(screen.getByText('现有访问密钥')).toBeInTheDocument()
  })

  it('shows a failed readiness request instead of a blank frame', async () => {
    renderPage(new Error('服务连接失败'))
    expect(await screen.findByText('服务连接失败')).toBeInTheDocument()
    expect(screen.queryByTitle('Magpie API 网关')).not.toBeInTheDocument()
  })

  it('preserves revocation of existing keys behind explicit confirmation', async () => {
    const { deleteSpy, user } = renderPage()
    await user.click(screen.getByText('现有访问密钥'))
    await user.click(await screen.findByRole('button', { name: /吊\s*销/ }))
    expect(deleteSpy).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: /确\s*定/ }))
    await waitFor(() => expect(deleteSpy).toHaveBeenCalledWith('/api/keys/existing-key'))
  })
})
