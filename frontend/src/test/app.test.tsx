import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, test, vi } from 'vitest'

import App from '../App'


function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}


test('shows login when the admin session is missing', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
    jsonResponse({ authenticated: false, user: null, csrf_token: null }),
  )

  render(<App />)

  expect(await screen.findByRole('heading', { name: 'DGX Spark 管理器' })).toBeInTheDocument()
  expect(screen.getByLabelText('用户名')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /登\s*录/ })).toBeInTheDocument()
})


test('opens the real dashboard after login and can switch theme', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input)
    if (url.endsWith('/api/auth/session')) {
      return jsonResponse({ authenticated: false, user: null, csrf_token: null })
    }
    if (url.endsWith('/api/auth/login') && init?.method === 'POST') {
      return jsonResponse({ user: { username: 'admin', role: 'admin' }, csrf_token: 'csrf-test' })
    }
    if (url.endsWith('/api/system')) {
      return jsonResponse({
        hostname: 'gx10-test',
        architecture: 'aarch64',
        os: 'Ubuntu 24.04',
        kernel: '6.17.0-nvidia',
        cpu: { percent: 10, cores: 20 },
        memory: { total_bytes: 128000, used_bytes: 64000, available_bytes: 64000 },
        disk: { total_bytes: 1000000, used_bytes: 200000, free_bytes: 800000 },
        gpus: [{ name: 'NVIDIA GB10', driver_version: '580.173.02', temperature_c: 41, power_w: 12, memory_used_bytes: null, utilization_percent: 0 }],
        uptime_seconds: 3600,
      })
    }
    if (url.endsWith('/api/deployments')) return jsonResponse([])
    if (url.endsWith('/api/tasks?limit=6')) return jsonResponse([])
    if (url.endsWith('/api/gateway/stats')) return jsonResponse({ total_requests: 0, failed_requests: 0, error_rate: 0, average_latency_ms: 0, prompt_tokens: 0, completion_tokens: 0 })
    return jsonResponse([])
  })

  render(<App />)
  await screen.findByLabelText('用户名')
  fireEvent.change(screen.getByLabelText('用户名'), { target: { value: 'admin' } })
  fireEvent.change(screen.getByLabelText('密码'), { target: { value: 'password' } })
  fireEvent.click(screen.getByRole('button', { name: /登\s*录/ }))

  expect(await screen.findByRole('heading', { name: '系统概览' })).toBeInTheDocument()
  expect(await screen.findByText('gx10-test')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '切换主题' }))
  fireEvent.click(await screen.findByText('深色'))
  await waitFor(() => expect(document.documentElement.dataset.theme).toBe('dark'))
  expect(fetchMock).toHaveBeenCalled()
})
