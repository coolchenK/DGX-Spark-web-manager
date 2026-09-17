import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Segmented, Space, Switch, Tag, Typography, message } from 'antd'
import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { UpstreamGateway, UpstreamModels } from '../api/types'
import { QueryState } from './QueryState'


type ExposureMode = 'all' | 'selected'


function sameSelection(left: Set<string>, right: string[]) {
  if (left.size !== right.length) return false
  return right.every((item) => left.has(item))
}

export function UpstreamModelPicker({ upstream }: { upstream: UpstreamGateway | undefined }) {
  const queryClient = useQueryClient()
  const [mode, setMode] = useState<ExposureMode>('all')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const models = useQuery({
    queryKey: ['upstream-models'],
    queryFn: () => api.get<UpstreamModels>('/api/gateway/upstream/models'),
    enabled: Boolean(upstream?.enabled),
  })

  // The stored configuration is the source of truth; re-sync the draft whenever
  // it reloads so a saved change is reflected immediately.
  useEffect(() => {
    if (!upstream) return
    setMode(upstream.expose_all ? 'all' : 'selected')
    setSelected(new Set(upstream.selected_models))
  }, [upstream])

  const save = useMutation({
    mutationFn: () =>
      api.put('/api/gateway/upstream/exposure', {
        expose_all: mode === 'all',
        selected_models: [...selected].sort(),
      }),
    onSuccess: () => {
      message.success('上游模型提供范围已更新')
      void queryClient.invalidateQueries({ queryKey: ['upstream-gateway'] })
      void queryClient.invalidateQueries({ queryKey: ['upstream-models'] })
      void queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })

  const changed =
    Boolean(upstream) &&
    ((mode === 'all') !== upstream?.expose_all ||
      (mode === 'selected' && !sameSelection(selected, upstream?.selected_models ?? [])))

  const toggle = (id: string, on: boolean) => {
    setSelected((current) => {
      const next = new Set(current)
      if (on) next.add(id)
      else next.delete(id)
      return next
    })
  }

  if (!upstream?.enabled) return null

  const discovered = models.data?.models ?? []
  const unavailable = models.data?.status === 'unavailable'

  return (
    <section className="content-section upstream-exposure">
      <div className="section-heading">
        <div>
          <h2>上游模型提供范围</h2>
          <p>选择哪些上游模型在本网关对外提供；未提供的模型不会出现在模型列表中，也无法调用</p>
        </div>
        <Space wrap>
          <Segmented
            value={mode}
            onChange={(value) => setMode(value as ExposureMode)}
            options={[
              { value: 'all', label: '提供全部' },
              { value: 'selected', label: '仅提供所选' },
            ]}
          />
          <Button type="primary" loading={save.isPending} disabled={!changed} onClick={() => save.mutate()}>
            保存范围
          </Button>
        </Space>
      </div>
      <QueryState loading={models.isLoading} error={models.error}>
        {unavailable && (
          <Alert
            type="warning"
            showIcon
            message="暂时无法读取上游模型列表"
            description="已保存的提供范围仍然有效；上游恢复后这里会重新显示模型列表。"
          />
        )}
        {!unavailable && discovered.length === 0 && (
          <Alert type="info" showIcon message="上游没有返回任何模型" />
        )}
        {!unavailable && discovered.length > 0 && (
          <>
            <Typography.Text type="secondary">
              {mode === 'all'
                ? `当前提供上游解析出的全部 ${discovered.length} 个模型。`
                : `已选择 ${selected.size} / ${discovered.length} 个模型。`}
            </Typography.Text>
            <ul className="upstream-model-list">
              {discovered.map((model) => {
                const on = mode === 'all' || selected.has(model.id)
                return (
                  <li key={model.id}>
                    <Typography.Text code>{model.id}</Typography.Text>
                    <Space size={8}>
                      {mode === 'selected' && !selected.has(model.id) && <Tag>不提供</Tag>}
                      <Switch
                        size="small"
                        checked={on}
                        disabled={mode === 'all' || save.isPending}
                        aria-label={`在本网关提供 ${model.id}`}
                        onChange={(checked) => toggle(model.id, checked)}
                      />
                    </Space>
                  </li>
                )
              })}
            </ul>
            {mode === 'selected' && selected.size === 0 && (
              <Alert
                type="warning"
                showIcon
                message="当前没有选择任何模型"
                description="保存后，本网关将不再提供任何上游模型，请求这些模型会返回 404。"
              />
            )}
          </>
        )}
      </QueryState>
    </section>
  )
}
