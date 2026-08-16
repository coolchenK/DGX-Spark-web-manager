import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { message } from 'antd'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { ApiError, api } from '../api/client'
import type { ModelAsset, TaskRecord } from '../api/types'
import { ModelsPage } from './ModelsPage'


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
    name: 'Local Model',
    alias: null,
    source: 'local',
    repository_id: null,
    revision: null,
    commit_hash: null,
    local_path: '/models/local-model',
    format: 'gguf',
    quantization: null,
    parameter_count: null,
    size_bytes: 3 * GiB,
    status: 'available',
    capabilities: ['generate'],
    created_at: '2026-08-16T00:00:00Z',
    updated_at: '2026-08-16T00:00:00Z',
  },
]

const task: TaskRecord = {
  id: 'task-delete-1',
  type: 'model.delete',
  status: 'queued',
  title: '删除模型 Qwen/Qwen-Test',
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

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function renderModelsPage() {
  const user = userEvent.setup()
  vi.spyOn(api, 'get').mockResolvedValue(models)
  const deleteSpy = vi.spyOn(api, 'delete').mockResolvedValue(task)
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  })
  const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/models']}>
        <ModelsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )

  return { user, deleteSpy, invalidateSpy }
}

async function openDeleteDialog(
  user: ReturnType<typeof userEvent.setup>,
  modelName = 'Qwen/Qwen-Test',
) {
  const deleteButton = await screen.findByRole('button', { name: `删除模型 ${modelName}` })
  await user.click(deleteButton)
  return screen.getByRole('dialog', { name: '永久删除模型' })
}


describe('ModelsPage permanent deletion', () => {
  it('offers model-specific delete actions in the mobile list', async () => {
    renderModelsPage()

    expect(await screen.findByRole('button', { name: '删除模型 Qwen/Qwen-Test' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '删除模型 Local Model' })).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('offers delete actions in the desktop model table', async () => {
    vi.spyOn(window, 'matchMedia').mockImplementation((query) => ({
      matches: query.includes('min-width'),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
    renderModelsPage()

    const table = await screen.findByRole('table')
    expect(within(table).getByRole('button', { name: '删除模型 Qwen/Qwen-Test' })).toBeInTheDocument()
    expect(within(table).getByRole('button', { name: '删除模型 Local Model' })).toBeInTheDocument()
  })

  it('shows model details and requires an exact full-name confirmation', async () => {
    const { user } = renderModelsPage()

    const dialog = await openDeleteDialog(user)

    expect(within(dialog).getAllByText('Qwen/Qwen-Test')).toHaveLength(2)
    expect(within(dialog).getByText('huggingface')).toBeInTheDocument()
    expect(within(dialog).getByText('7.0 GiB')).toBeInTheDocument()
    expect(within(dialog).getByText('safetensors')).toBeInTheDocument()
    expect(within(dialog).getByText(/不可逆/)).toBeInTheDocument()
    const submit = within(dialog).getByRole('button', { name: '永久删除' })
    const confirmation = within(dialog).getByLabelText('输入完整模型名称')
    expect(submit).toBeDisabled()

    await user.type(confirmation, 'Qwen/Qwen-Tes')
    expect(submit).toBeDisabled()
    await user.type(confirmation, 't')
    expect(submit).toBeEnabled()
  })

  it('submits confirmation, closes the dialog, and refreshes models and tasks', async () => {
    const { user, deleteSpy, invalidateSpy } = renderModelsPage()
    const dialog = await openDeleteDialog(user)
    await user.type(within(dialog).getByLabelText('输入完整模型名称'), 'Qwen/Qwen-Test')

    await user.click(within(dialog).getByRole('button', { name: '永久删除' }))

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith('/api/models/model-1', {
        confirmation: 'Qwen/Qwen-Test',
      })
    })
    await waitFor(() => expect(dialog).toHaveClass('ant-zoom-leave-active'))
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['models'] })
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['tasks'] })
  })

  it('keeps the dialog open and lists deployment references after a model-in-use conflict', async () => {
    const { user, deleteSpy } = renderModelsPage()
    deleteSpy.mockRejectedValueOnce(new ApiError(409, '请求失败 (409)', {
      code: 'model_in_use',
      references: [
        { deployment_id: 'deployment-base', deployment_name: 'Primary service', usage: 'base' },
        { deployment_id: 'deployment-draft', deployment_name: 'Draft service', usage: 'draft' },
        { deployment_id: 'deployment-legacy', deployment_name: 'Legacy service', usage: 'legacy_path' },
      ],
    }))
    const dialog = await openDeleteDialog(user)
    await user.type(within(dialog).getByLabelText('输入完整模型名称'), 'Qwen/Qwen-Test')

    await user.click(within(dialog).getByRole('button', { name: '永久删除' }))

    const referenceItems = await within(dialog).findAllByRole('listitem')
    expect(referenceItems[0]).toHaveTextContent('Primary service · 基础模型')
    expect(referenceItems[1]).toHaveTextContent('Draft service · Draft Model')
    expect(referenceItems[2]).toHaveTextContent('Legacy service · 旧路径引用')
    expect(within(dialog).getByRole('link', { name: 'Primary service' })).toHaveAttribute(
      'href',
      '/deployments?deployment=deployment-base',
    )
    expect(screen.getByRole('dialog', { name: '永久删除模型' })).toBeInTheDocument()
  })

  it('clears confirmation and conflict state before opening another model', async () => {
    const { user, deleteSpy } = renderModelsPage()
    deleteSpy.mockRejectedValueOnce(new ApiError(409, '请求失败 (409)', {
      code: 'model_in_use',
      references: [
        { deployment_id: 'deployment-base', deployment_name: 'Primary service', usage: 'base' },
      ],
    }))
    const firstDialog = await openDeleteDialog(user)
    await user.type(within(firstDialog).getByLabelText('输入完整模型名称'), 'Qwen/Qwen-Test')
    await user.click(within(firstDialog).getByRole('button', { name: '永久删除' }))
    expect(await within(firstDialog).findByText('Primary service')).toBeInTheDocument()

    await user.click(within(firstDialog).getByRole('button', { name: /取\s*消/ }))
    const secondDialog = await openDeleteDialog(user, 'Local Model')

    expect(within(secondDialog).getByLabelText('输入完整模型名称')).toHaveValue('')
    expect(within(secondDialog).queryByText('Primary service')).not.toBeInTheDocument()
    expect(within(secondDialog).getByText('/models/local-model')).toBeInTheDocument()
  })

  it('blocks Escape while pending and ignores a late success after another model is opened', async () => {
    const pending = deferred<TaskRecord>()
    const successMessageSpy = vi.spyOn(message, 'success')
    const { user, deleteSpy, invalidateSpy } = renderModelsPage()
    deleteSpy.mockReturnValueOnce(pending.promise)
    const secondDeleteButton = await screen.findByRole('button', { name: '删除模型 Local Model' })
    const firstDialog = await openDeleteDialog(user)
    await user.type(within(firstDialog).getByLabelText('输入完整模型名称'), 'Qwen/Qwen-Test')
    await user.click(within(firstDialog).getByRole('button', { name: '永久删除' }))
    await waitFor(() => expect(deleteSpy).toHaveBeenCalledTimes(1))

    await user.keyboard('{Escape}')
    expect(screen.getByRole('dialog', { name: '永久删除模型' })).toBeInTheDocument()
    fireEvent.click(secondDeleteButton)
    const secondDialog = screen.getByRole('dialog', { name: '永久删除模型' })
    expect(within(secondDialog).getByText('/models/local-model')).toBeInTheDocument()

    pending.resolve(task)

    const permanentDeleteButton = within(secondDialog).getByText('永久删除').closest('button')
    await waitFor(() => expect(permanentDeleteButton).not.toHaveClass('ant-btn-loading'))
    expect(within(secondDialog).getByText('/models/local-model')).toBeInTheDocument()
    expect(screen.queryByText('Primary service')).not.toBeInTheDocument()
    expect(successMessageSpy).toHaveBeenCalledWith('模型删除任务已创建')
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['models'] })
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['tasks'] })
  })

  it('ignores a late model-in-use conflict after another model is opened', async () => {
    const pending = deferred<TaskRecord>()
    const successMessageSpy = vi.spyOn(message, 'success')
    const { user, deleteSpy, invalidateSpy } = renderModelsPage()
    deleteSpy.mockReturnValueOnce(pending.promise)
    const secondDeleteButton = await screen.findByRole('button', { name: '删除模型 Local Model' })
    const firstDialog = await openDeleteDialog(user)
    await user.type(within(firstDialog).getByLabelText('输入完整模型名称'), 'Qwen/Qwen-Test')
    await user.click(within(firstDialog).getByRole('button', { name: '永久删除' }))
    await waitFor(() => expect(deleteSpy).toHaveBeenCalledTimes(1))
    fireEvent.click(secondDeleteButton)
    const secondDialog = screen.getByRole('dialog', { name: '永久删除模型' })

    pending.reject(new ApiError(409, '请求失败 (409)', {
      code: 'model_in_use',
      references: [
        { deployment_id: 'deployment-base', deployment_name: 'Primary service', usage: 'base' },
      ],
    }))

    const permanentDeleteButton = within(secondDialog).getByText('永久删除').closest('button')
    await waitFor(() => expect(permanentDeleteButton).not.toHaveClass('ant-btn-loading'))
    expect(within(secondDialog).getByText('/models/local-model')).toBeInTheDocument()
    expect(screen.queryByText('Primary service')).not.toBeInTheDocument()
    expect(secondDialog).toBeInTheDocument()
    expect(successMessageSpy).not.toHaveBeenCalled()
    expect(invalidateSpy).not.toHaveBeenCalled()
  })
})


describe('api.delete', () => {
  it('serializes an optional JSON request body', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify(task),
      { status: 202, headers: { 'Content-Type': 'application/json' } },
    ))

    await api.delete<TaskRecord>('/api/models/model-1', {
      confirmation: 'Qwen/Qwen-Test',
    })

    expect(fetchSpy).toHaveBeenCalledWith(
      '/api/models/model-1',
      expect.objectContaining({
        method: 'DELETE',
        body: JSON.stringify({ confirmation: 'Qwen/Qwen-Test' }),
      }),
    )
  })
})
