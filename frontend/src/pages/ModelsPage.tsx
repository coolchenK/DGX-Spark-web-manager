import { AppstoreAddOutlined, DashboardOutlined, DeleteOutlined, ReloadOutlined, RocketOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Descriptions, Flex, Input, Modal, Space, Tag, Tooltip, Typography, message } from 'antd'
import { useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { ApiError, api } from '../api/client'
import type { ModelAsset, ModelInUseDetail, ModelReference, TaskRecord } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes, formatDate } from '../utils/format'

interface ModelFamily {
  key: string
  name: string
  variants: ModelAsset[]
  baseVariants: ModelAsset[]
  draftVariants: ModelAsset[]
  primary: ModelAsset
}

const DRAFT_ASSET_SUFFIX = /[-_](d(?:spark|flash))$/i
const BASE_VARIANT_SUFFIX = /[-_](?:nvfp4|fp8|fp4|awq)$/i

function isDraftAsset(model: ModelAsset): boolean {
  return DRAFT_ASSET_SUFFIX.test(model.repository_id ?? model.name)
}

function modelFamilyKey(model: ModelAsset): string {
  return modelFamilyStem(model).toLowerCase()
}

function modelFamilyStem(model: ModelAsset): string {
  const identity = model.repository_id ?? model.name
  return identity.trim().replace(DRAFT_ASSET_SUFFIX, '').replace(BASE_VARIANT_SUFFIX, '')
}

function modelFamilyName(model: ModelAsset): string {
  const identity = model.repository_id ?? model.name
  return identity.trim().replace(DRAFT_ASSET_SUFFIX, '')
}

function variantLabel(model: ModelAsset, familyName: string): string {
  const identity = model.repository_id ?? model.name
  if (identity === familyName) {
    if (model.name !== familyName) return model.name.split('/').pop() || model.name
    return model.status === 'unavailable' ? 'HF 缓存' : '基础模型'
  }
  const draftSuffix = identity.match(DRAFT_ASSET_SUFFIX)?.[1]
  if (draftSuffix) return draftSuffix.replace(/^d/i, 'D')
  return identity.slice(modelFamilyStem(model).length).replace(/^[-_]/, '') || '变体'
}

function modelBenchmark(model: ModelAsset) {
  if (model.benchmark_tps == null) {
    return <Typography.Text type="secondary">未测试</Typography.Text>
  }
  return (
    <Tooltip title={model.benchmark_tested_at ? `最近测试：${formatDate(model.benchmark_tested_at)}` : '最近一次成功测试'}>
      <Typography.Text strong><DashboardOutlined /> {model.benchmark_tps.toFixed(2)} tok/s</Typography.Text>
    </Tooltip>
  )
}

function familyBenchmark(family: ModelFamily) {
  if (family.baseVariants.length <= 1) {
    return family.baseVariants[0]
      ? modelBenchmark(family.baseVariants[0])
      : <Typography.Text type="secondary">未测试</Typography.Text>
  }
  return (
    <Space direction="vertical" size={0}>
      {family.baseVariants.map((variant) => (
        <span key={variant.id}>{variantLabel(variant, family.name)}: {modelBenchmark(variant)}</span>
      ))}
    </Space>
  )
}


export function ModelsPage() {
  const [filter, setFilter] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<ModelAsset | null>(null)
  const deleteTargetRef = useRef<ModelAsset | null>(null)
  const [confirmation, setConfirmation] = useState('')
  const [references, setReferences] = useState<ModelReference[]>([])
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: () => api.get<ModelAsset[]>('/api/models') })
  const scan = useMutation({ mutationFn: () => api.post('/api/discovery/scan'), onSuccess: () => { message.success('模型扫描完成'); queryClient.invalidateQueries({ queryKey: ['models'] }) } })
  const closeDelete = (targetId?: string) => {
    if (targetId && deleteTargetRef.current?.id !== targetId) return
    deleteTargetRef.current = null
    setDeleteTarget(null)
    setConfirmation('')
    setReferences([])
  }
  const openDelete = (model: ModelAsset) => {
    deleteTargetRef.current = model
    setConfirmation('')
    setReferences([])
    setDeleteTarget(model)
  }
  const remove = useMutation({
    mutationFn: (model: ModelAsset) => api.delete<TaskRecord>(`/api/models/${model.id}`, { confirmation: model.name }),
    onSuccess: (_task, model) => {
      closeDelete(model.id)
      message.success('模型删除任务已创建')
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error, model) => {
      if (deleteTargetRef.current?.id !== model.id) return
      if (error instanceof ApiError && error.status === 409 && isModelInUseDetail(error.detail)) {
        setReferences(error.detail.references)
        return
      }
      message.error(error instanceof Error ? error.message : '删除模型失败')
    },
  })
  const data = useMemo<ModelFamily[]>(() => {
    const groups = new Map<string, ModelAsset[]>()
    for (const item of models.data ?? []) {
      const key = modelFamilyKey(item)
      groups.set(key, [...(groups.get(key) ?? []), item])
    }
    return [...groups.entries()].map(([key, variants]) => {
      const ordered = [...variants].sort((left, right) => {
        const leftAuxiliary = isDraftAsset(left) ? 1 : 0
        const rightAuxiliary = isDraftAsset(right) ? 1 : 0
        return leftAuxiliary - rightAuxiliary || left.name.localeCompare(right.name)
      })
      const baseVariants = ordered.filter((item) => !isDraftAsset(item))
      const draftVariants = ordered.filter(isDraftAsset)
      const base = baseVariants[0]
      const name = base
        ? (ordered.length > 1 ? modelFamilyName(base) : base.name)
        : modelFamilyName(ordered[0])
      return {
        key,
        name,
        variants: ordered,
        baseVariants,
        draftVariants,
        primary: base?.status === 'available'
          ? base
          : ordered.find((item) => item.status === 'available') ?? base ?? ordered[0],
      }
    }).filter((family) => {
      const query = filter.trim().toLowerCase()
      if (!query) return true
      return `${family.name} ${family.variants.map((item) => `${item.name} ${item.repository_id ?? ''}`).join(' ')}`
        .toLowerCase()
        .includes(query)
    }).sort((left, right) => left.name.localeCompare(right.name))
  }, [models.data, filter])
  const deleteButton = (item: ModelAsset) => (
    <Tooltip title="删除模型">
      <Button danger size="small" icon={<DeleteOutlined />} aria-label={`删除模型 ${item.name}`} onClick={() => openDelete(item)} />
    </Tooltip>
  )
  const sizeLabel = (item: ModelAsset) => (
    item.status === 'unavailable' && item.size_bytes === 0 ? '缓存不完整' : formatBytes(item.size_bytes)
  )
  return (
    <div className="page-stack">
      <PageHeader title="模型库" description="统一查看 Hugging Face 缓存、本地目录与部署关联" extra={<Button icon={<ReloadOutlined />} loading={scan.isPending} onClick={() => scan.mutate()}>扫描模型</Button>} />
      <div className="filter-bar"><Input.Search allowClear placeholder="按名称或仓库筛选" value={filter} onChange={(event) => setFilter(event.target.value)} /></div>
      <QueryState loading={models.isLoading} error={models.error} empty={!data.length} onRetry={() => models.refetch()}>
        <ResponsiveDataView data={data} rowKey="key" columns={[
          { title: '模型家族', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.variants.length > 1 ? `${item.variants.length} 个相关变体已合并` : (item.primary.repository_id ?? item.primary.local_path)}</small><Space size={[2, 4]} wrap>{item.baseVariants.map((variant) => <Tag key={variant.id}>{variantLabel(variant, item.name)}</Tag>)}{item.draftVariants.map((variant) => <Tag color="blue" key={variant.id}>Draft · {variantLabel(variant, item.name)}</Tag>)}</Space></div> },
          { title: '来源', dataIndex: 'source', width: 120, render: (_, item) => <Tag>{item.primary.source}</Tag> },
          { title: '大小', dataIndex: 'size_bytes', width: 140, render: (_, item) => <Space direction="vertical" size={0}>{item.baseVariants.map((variant) => <span key={variant.id}>{variantLabel(variant, item.name)}: {sizeLabel(variant)}</span>)}{item.draftVariants.map((variant) => <Typography.Text type="secondary" key={variant.id}>Draft: {sizeLabel(variant)}</Typography.Text>)}</Space> },
          { title: '能力', dataIndex: 'capabilities', render: (_, item) => <Space size={[2, 4]} wrap>{[...new Set(item.variants.flatMap((variant) => variant.capabilities))].map((value) => <Tag key={value}>{value}</Tag>)}</Space> },
          { title: 'TPS', dataIndex: 'benchmark_tps', width: 180, render: (_, item) => familyBenchmark(item) },
          { title: '状态', dataIndex: 'status', width: 100, render: (_, item) => <StatusBadge status={item.primary.status} /> },
          { title: '', width: 180, render: (_, item) => <Space direction="vertical" size="small">{item.baseVariants.map((variant) => <Space size="small" key={variant.id}><Button size="small" icon={<RocketOutlined />} disabled={variant.status !== 'available'} onClick={() => navigate(`/deployments?model=${variant.id}`)}>部署 {variantLabel(variant, item.name)}</Button>{deleteButton(variant)}</Space>)}{item.draftVariants.map((variant) => <Space size="small" key={variant.id}><Tag color="blue">Draft · {variantLabel(variant, item.name)}</Tag>{deleteButton(variant)}</Space>)}</Space> },
        ]} renderMobile={(item) => <div className="mobile-record"><Flex justify="space-between"><Space><AppstoreAddOutlined /><strong>{item.name}</strong></Space><StatusBadge status={item.primary.status} /></Flex><Typography.Text type="secondary">{item.variants.length > 1 ? `${item.variants.length} 个相关变体已合并` : (item.primary.repository_id ?? item.primary.local_path)}</Typography.Text><dl><div><dt>大小</dt><dd><Space direction="vertical" size={0}>{item.baseVariants.map((variant) => <span key={variant.id}>{variantLabel(variant, item.name)}: {sizeLabel(variant)}</span>)}{item.draftVariants.map((variant) => <Typography.Text type="secondary" key={variant.id}>Draft: {sizeLabel(variant)}</Typography.Text>)}</Space></dd></div><div><dt>TPS</dt><dd>{familyBenchmark(item)}</dd></div><div><dt>更新</dt><dd>{formatDate(item.primary.updated_at)}</dd></div></dl><Space direction="vertical" style={{ width: '100%' }}>{item.baseVariants.map((variant) => <Flex gap="small" key={variant.id}><Button block icon={<RocketOutlined />} disabled={variant.status !== 'available'} onClick={() => navigate(`/deployments?model=${variant.id}`)}>部署模型 {variantLabel(variant, item.name)}</Button>{deleteButton(variant)}</Flex>)}{item.draftVariants.map((variant) => <Flex align="center" justify="space-between" key={variant.id}><Tag color="blue">Draft · {variantLabel(variant, item.name)}</Tag>{deleteButton(variant)}</Flex>)}</Space></div>} />
      </QueryState>
      <Modal
        title="永久删除模型"
        open={Boolean(deleteTarget)}
        onCancel={() => {
          if (!remove.isPending) closeDelete()
        }}
        onOk={() => deleteTarget && remove.mutate(deleteTarget)}
        okText="永久删除"
        cancelText="取消"
        confirmLoading={remove.isPending}
        okButtonProps={{
          danger: true,
          disabled: !deleteTarget || confirmation !== deleteTarget.name,
        }}
        cancelButtonProps={{ disabled: remove.isPending }}
        destroyOnHidden
        keyboard={!remove.isPending}
        closable={!remove.isPending}
        maskClosable={!remove.isPending}
      >
        {deleteTarget && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Alert type="warning" showIcon message="此操作不可逆" description="模型文件将被永久删除，且无法从模型库恢复。" />
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="模型名称">{deleteTarget.name}</Descriptions.Item>
              <Descriptions.Item label="来源">{deleteTarget.source}</Descriptions.Item>
              <Descriptions.Item label={deleteTarget.repository_id ? '仓库' : '本地路径'}>
                {deleteTarget.repository_id ?? deleteTarget.local_path}
              </Descriptions.Item>
              <Descriptions.Item label="格式">{deleteTarget.format ?? '未知'}</Descriptions.Item>
              <Descriptions.Item label="大小">{sizeLabel(deleteTarget)}</Descriptions.Item>
            </Descriptions>
            {references.length > 0 && (
              <Alert
                type="error"
                showIcon
                message="模型正在被部署使用"
                description={(
                  <ul>
                    {references.map((reference) => (
                      <li key={reference.deployment_id}>
                        <Link to={`/deployments?deployment=${reference.deployment_id}`}>{reference.deployment_name}</Link>
                        {' · '}{modelReferenceUsageLabels[reference.usage]}
                      </li>
                    ))}
                  </ul>
                )}
              />
            )}
            <label htmlFor="model-delete-confirmation">输入完整模型名称</label>
            <Input
              id="model-delete-confirmation"
              value={confirmation}
              autoComplete="off"
              placeholder={deleteTarget.name}
              onChange={(event) => {
                setConfirmation(event.target.value)
                setReferences([])
              }}
            />
          </Space>
        )}
      </Modal>
    </div>
  )
}


function isModelInUseDetail(detail: unknown): detail is ModelInUseDetail {
  if (typeof detail !== 'object' || detail === null) return false
  const value = detail as Partial<ModelInUseDetail>
  return value.code === 'model_in_use' && Array.isArray(value.references)
}

const modelReferenceUsageLabels: Record<ModelReference['usage'], string> = {
  base: '基础模型',
  draft: 'Draft Model',
  legacy_path: '旧路径引用',
}
