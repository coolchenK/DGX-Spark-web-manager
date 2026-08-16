import { AppstoreAddOutlined, DeleteOutlined, ReloadOutlined, RocketOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Descriptions, Flex, Input, Modal, Space, Tag, Tooltip, Typography, message } from 'antd'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { ApiError, api } from '../api/client'
import type { ModelAsset, ModelInUseDetail, ModelReference, TaskRecord } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes, formatDate } from '../utils/format'


export function ModelsPage() {
  const [filter, setFilter] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<ModelAsset | null>(null)
  const [confirmation, setConfirmation] = useState('')
  const [references, setReferences] = useState<ModelReference[]>([])
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: () => api.get<ModelAsset[]>('/api/models') })
  const scan = useMutation({ mutationFn: () => api.post('/api/discovery/scan'), onSuccess: () => { message.success('模型扫描完成'); queryClient.invalidateQueries({ queryKey: ['models'] }) } })
  const closeDelete = () => {
    setDeleteTarget(null)
    setConfirmation('')
    setReferences([])
  }
  const openDelete = (model: ModelAsset) => {
    setConfirmation('')
    setReferences([])
    setDeleteTarget(model)
  }
  const remove = useMutation({
    mutationFn: (model: ModelAsset) => api.delete<TaskRecord>(`/api/models/${model.id}`, { confirmation: model.name }),
    onSuccess: () => {
      closeDelete()
      message.success('模型删除任务已创建')
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409 && isModelInUseDetail(error.detail)) {
        setReferences(error.detail.references)
        return
      }
      message.error(error instanceof Error ? error.message : '删除模型失败')
    },
  })
  const data = useMemo(() => (models.data ?? []).filter((item) => `${item.name} ${item.repository_id ?? ''}`.toLowerCase().includes(filter.toLowerCase())), [models.data, filter])
  const deleteButton = (item: ModelAsset) => (
    <Tooltip title="删除模型">
      <Button danger size="small" icon={<DeleteOutlined />} aria-label="删除模型" onClick={() => openDelete(item)} />
    </Tooltip>
  )
  return (
    <div className="page-stack">
      <PageHeader title="模型库" description="统一查看 Hugging Face 缓存、本地目录与部署关联" extra={<Button icon={<ReloadOutlined />} loading={scan.isPending} onClick={() => scan.mutate()}>扫描模型</Button>} />
      <div className="filter-bar"><Input.Search allowClear placeholder="按名称或仓库筛选" value={filter} onChange={(event) => setFilter(event.target.value)} /></div>
      <QueryState loading={models.isLoading} error={models.error} empty={!data.length} onRetry={() => models.refetch()}>
        <ResponsiveDataView data={data} rowKey="id" columns={[
          { title: '模型', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.repository_id ?? item.local_path}</small></div> },
          { title: '来源', dataIndex: 'source', width: 120, render: (value) => <Tag>{value}</Tag> },
          { title: '大小', dataIndex: 'size_bytes', width: 110, render: (value) => formatBytes(value) },
          { title: '能力', dataIndex: 'capabilities', render: (values: string[]) => <Space size={[2, 4]} wrap>{values.map((value) => <Tag key={value}>{value}</Tag>)}</Space> },
          { title: '状态', dataIndex: 'status', width: 100, render: (value) => <StatusBadge status={value} /> },
          { title: '', width: 132, render: (_, item) => <Space size="small"><Button size="small" icon={<RocketOutlined />} onClick={() => navigate(`/deployments?model=${item.id}`)}>部署</Button>{deleteButton(item)}</Space> },
        ]} renderMobile={(item) => <div className="mobile-record"><Flex justify="space-between"><Space><AppstoreAddOutlined /><strong>{item.name}</strong></Space><StatusBadge status={item.status} /></Flex><Typography.Text type="secondary">{item.repository_id ?? item.local_path}</Typography.Text><dl><div><dt>大小</dt><dd>{formatBytes(item.size_bytes)}</dd></div><div><dt>更新</dt><dd>{formatDate(item.updated_at)}</dd></div></dl><Flex gap="small"><Button block icon={<RocketOutlined />} onClick={() => navigate(`/deployments?model=${item.id}`)}>部署模型</Button>{deleteButton(item)}</Flex></div>} />
      </QueryState>
      <Modal
        title="永久删除模型"
        open={Boolean(deleteTarget)}
        onCancel={closeDelete}
        destroyOnHidden
        closable={!remove.isPending}
        maskClosable={!remove.isPending}
        footer={[
          <Button key="cancel" disabled={remove.isPending} onClick={closeDelete}>取消</Button>,
          <Button
            key="delete"
            danger
            type="primary"
            loading={remove.isPending}
            disabled={!deleteTarget || confirmation !== deleteTarget.name}
            onClick={() => deleteTarget && remove.mutate(deleteTarget)}
          >
            永久删除
          </Button>,
        ]}
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
              <Descriptions.Item label="大小">{formatBytes(deleteTarget.size_bytes)}</Descriptions.Item>
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
