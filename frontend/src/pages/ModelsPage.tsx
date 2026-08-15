import { AppstoreAddOutlined, ReloadOutlined, RocketOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Flex, Input, Space, Tag, Typography, message } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '../api/client'
import type { ModelAsset } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes, formatDate } from '../utils/format'


export function ModelsPage() {
  const [filter, setFilter] = useState('')
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: () => api.get<ModelAsset[]>('/api/models') })
  const scan = useMutation({ mutationFn: () => api.post('/api/discovery/scan'), onSuccess: () => { message.success('模型扫描完成'); queryClient.invalidateQueries({ queryKey: ['models'] }) } })
  const data = useMemo(() => (models.data ?? []).filter((item) => `${item.name} ${item.repository_id ?? ''}`.toLowerCase().includes(filter.toLowerCase())), [models.data, filter])
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
          { title: '', width: 92, render: (_, item) => <Button size="small" icon={<RocketOutlined />} onClick={() => navigate(`/deployments?model=${item.id}`)}>部署</Button> },
        ]} renderMobile={(item) => <div className="mobile-record"><Flex justify="space-between"><Space><AppstoreAddOutlined /><strong>{item.name}</strong></Space><StatusBadge status={item.status} /></Flex><Typography.Text type="secondary">{item.repository_id ?? item.local_path}</Typography.Text><dl><div><dt>大小</dt><dd>{formatBytes(item.size_bytes)}</dd></div><div><dt>更新</dt><dd>{formatDate(item.updated_at)}</dd></div></dl><Button block icon={<RocketOutlined />} onClick={() => navigate(`/deployments?model=${item.id}`)}>部署模型</Button></div>} />
      </QueryState>
    </div>
  )
}
