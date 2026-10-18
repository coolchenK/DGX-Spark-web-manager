import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import { ImagesPage } from './ImagesPage'

afterEach(() => vi.restoreAllMocks())
it('upscales a selected result only after explicit confirmation and keeps its source link', async () => {
  const source = { id: 'source-job', title: '文生图', status: 'succeeded', progress: 100, log: '', deployment_id: 'viggle', created_at: '2026-10-01', request: { prompt: 'A paper boat on calm water', seed: 42 }, result: { width: 1024, height: 1024, seed: 42, steps: 6, seconds: 12, image_url: '/api/images/source-job/output' } }
  vi.spyOn(api, 'get').mockImplementation(async path => path === '/api/deployments' ? [{ id: 'viggle', name: 'Viggle', runtime: 'viggle', status: 'running' }] as never : [source] as never)
  const post = vi.spyOn(api, 'post').mockResolvedValue({ id: 'refine-job' })
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={cache}><ImagesPage /></QueryClientProvider>)
  fireEvent.click(await screen.findByRole('button', { name: '满意这张，快速超分' }))
  expect(post).not.toHaveBeenCalled()
  expect(screen.getByAltText('选中的放大源图')).toHaveAttribute('src', '/api/images/source-job/output')
  fireEvent.click(screen.getByRole('button', { name: '开始放大' }))
  await waitFor(() => expect(post).toHaveBeenCalledWith('/api/images', expect.objectContaining({ operation: 'upscale', source_task_id: 'source-job', sr_engine: 'realesrgan', sr_resolution: 4096, deployment_id: 'viggle' })))
  expect(post.mock.calls[0][1]).not.toHaveProperty('images')
  cache.clear()
})
it('submits recommended defaults and requires a reference in edit mode', async () => {
  vi.spyOn(api, 'get').mockImplementation(async path => path === '/api/deployments' ? [{ id: 'viggle', name: 'Viggle', runtime: 'viggle', status: 'running' }] as never : [] as never)
  const post = vi.spyOn(api, 'post').mockResolvedValue({ id: 'job' })
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={cache}><ImagesPage /></QueryClientProvider>)
  await waitFor(() => expect(screen.getByRole('button', { name: '生成图片' })).toBeEnabled())
  fireEvent.change(screen.getByLabelText('提示词'), { target: { value: 'A paper boat on calm water' } })
  fireEvent.click(screen.getByRole('button', { name: '生成图片' }))
  await waitFor(() => expect(post).toHaveBeenCalledWith('/api/images', expect.objectContaining({ prompt: 'A paper boat on calm water', steps: 6, seed: null, resolution: 1024, images: [], deployment_id: 'viggle' })))
  fireEvent.click(screen.getByRole('tab', { name: '图片编辑' }))
  expect(screen.getByRole('button', { name: '生成图片' })).toBeDisabled()
  expect(screen.getByText('添加参考图片（0/3）')).toBeInTheDocument()
  cache.clear()
})
