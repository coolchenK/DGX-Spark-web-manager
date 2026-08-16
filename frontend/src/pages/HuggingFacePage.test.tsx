import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, test, vi } from 'vitest'

import { api } from '../api/client'
import type { HuggingFaceModel } from '../api/types'
import { HuggingFacePage } from './HuggingFacePage'


describe('HuggingFacePage', () => {
  test('renders Spark compatibility metadata with stable styling hooks', async () => {
    const models: HuggingFaceModel[] = [
      {
        id: 'nvidia/DGX-Spark-ready-model',
        downloads: 12_345,
        likes: 678,
        pipeline_tag: 'text-generation',
        private: false,
        gated: false,
        last_modified: '2026-08-16T00:00:00Z',
        tags: [],
        spark_compatibility: {
          level: 'recommended',
          score: 100,
          reasons: ['统一内存满足模型需求', '原生支持 ARM64'],
        },
      },
      {
        id: 'example/model-requiring-review',
        downloads: 42,
        likes: 3,
        pipeline_tag: 'image-classification',
        private: false,
        gated: false,
        last_modified: '2026-08-15T00:00:00Z',
        tags: [],
        spark_compatibility: {
          level: 'review',
          score: 25,
          reasons: ['NVFP4 量化', '需要额外运行时', '生成任务'],
        },
      },
    ]
    vi.spyOn(api, 'get').mockResolvedValue(models)
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    })
    const user = userEvent.setup()

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <HuggingFacePage />
        </MemoryRouter>
      </QueryClientProvider>,
    )

    await user.type(screen.getByPlaceholderText('搜索模型名称，例如 Qwen 或 Nemotron'), 'DGX Spark')
    await user.click(screen.getByRole('button', { name: /搜\s*索/ }))

    expect(await screen.findByText('nvidia/DGX-Spark-ready-model')).toBeInTheDocument()
    expect(screen.getByText('text-generation')).toBeInTheDocument()
    expect(screen.getByText('12,345 次下载')).toBeInTheDocument()
    expect(screen.getByText('678 赞')).toBeInTheDocument()
    expect(screen.getByText('DGX Spark 推荐').closest('.ant-tag')).toHaveClass('hf-spark-tag', 'hf-spark-tag-recommended')
    expect(screen.getByText('统一内存满足模型需求 · 原生支持 ARM64')).toHaveClass('hf-compatibility-reason')
    expect(screen.getByText('需评估').closest('.ant-tag')).toHaveClass('hf-spark-tag', 'hf-spark-tag-review')
    expect(screen.getByText('NVFP4 量化 · 需要额外运行时')).toHaveClass('hf-compatibility-reason')
    expect(screen.queryByText(/生成任务/)).not.toBeInTheDocument()
  })
})
