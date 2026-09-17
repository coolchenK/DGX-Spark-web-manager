import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { api } from '../api/client'
import type { Provider, ProviderProbeResult } from '../api/types'
import { ProvidersPage } from './ProvidersPage'


const provider: Provider = {
  id: 'provider-1',
  name: 'Operations AI',
  base_url: 'https://provider.example/v1',
  default_model: 'missing',
  api_key_masked: 'sk-***',
  timeout_seconds: 60,
  headers: {},
  enabled: true,
  last_test_status: null,
  last_test_result: {},
  last_tested_at: null,
  created_at: '2026-08-18T00:00:00Z',
  updated_at: '2026-08-18T00:00:00Z',
}

function renderPage(result: ProviderProbeResult) {
  vi.spyOn(api, 'get').mockResolvedValue([provider])
  const postSpy = vi.spyOn(api, 'post').mockResolvedValue(result)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  })
  const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/providers']}>
        <ProvidersPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { invalidateSpy, postSpy, user: userEvent.setup() }
}


describe('ProvidersPage structured readiness probe', () => {
  it('distinguishes connection health from default model failure', async () => {
    const result: ProviderProbeResult = {
      status: 'failed',
      connection: { status: 'healthy', models_seen: 8 },
      default_model: { status: 'failed', model: 'missing', error: 'model not found' },
    }
    const { invalidateSpy, postSpy, user } = renderPage(result)

    await user.click(await screen.findByRole('button', { name: '测试连接' }))

    expect(postSpy).toHaveBeenCalledWith('/api/providers/provider-1/test')
    expect(await screen.findByText('API 连接正常')).toBeInTheDocument()
    expect(screen.getByText('已发现 8 个模型')).toBeInTheDocument()
    expect(screen.getByText('默认模型不可用')).toBeInTheDocument()
    expect(screen.getByText('model not found')).toBeInTheDocument()
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['providers'] })
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['diagnostics'] })
    })
  })
})
