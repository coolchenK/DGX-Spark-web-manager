import { CloudDownloadOutlined, LockOutlined, SearchOutlined } from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Button, Form, Input, List, Modal, Space, Tag, Typography, message } from 'antd'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api, queryString } from '../api/client'
import type { HuggingFaceModel, TaskRecord } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'


export function HuggingFacePage() {
  const [query, setQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [selected, setSelected] = useState<HuggingFaceModel | null>(null)
  const navigate = useNavigate()
  const results = useQuery({ queryKey: ['hf-search', activeQuery], queryFn: () => api.get<HuggingFaceModel[]>(`/api/huggingface/search${queryString({ query: activeQuery, limit: 30 })}`), enabled: Boolean(activeQuery) })
  const download = useMutation({ mutationFn: (values: { revision: string }) => api.post<TaskRecord>('/api/huggingface/downloads', { repository_id: selected?.id, revision: values.revision }), onSuccess: () => { message.success('下载任务已创建'); setSelected(null); navigate('/tasks') } })
  return (
    <div className="page-stack">
      <PageHeader title="Hugging Face" description="搜索、检查并下载模型到本机缓存" />
      <form className="hf-search" onSubmit={(event) => { event.preventDefault(); setActiveQuery(query.trim()) }}><Input size="large" allowClear prefix={<SearchOutlined />} placeholder="搜索模型名称，例如 Qwen 或 Nemotron" value={query} onChange={(event) => setQuery(event.target.value)} /><Button size="large" type="primary" htmlType="submit" disabled={!query.trim()}>搜索</Button></form>
      {!activeQuery ? <div className="search-idle"><CloudDownloadOutlined /><h2>查找 Hugging Face 模型</h2><p>搜索结果会显示任务类型、受限状态和社区使用情况。</p></div> : <QueryState loading={results.isLoading} error={results.error} empty={!results.data?.length}><List className="hf-results" dataSource={results.data} renderItem={(model) => <List.Item actions={[<Button key="download" icon={<CloudDownloadOutlined />} onClick={() => setSelected(model)}>下载</Button>]}><List.Item.Meta title={<Space><strong>{model.id}</strong>{model.gated && <Tag icon={<LockOutlined />} color="warning">需授权</Tag>}</Space>} description={<Space wrap><Tag>{model.pipeline_tag ?? '未分类'}</Tag><Typography.Text type="secondary">{model.downloads.toLocaleString()} 次下载</Typography.Text><Typography.Text type="secondary">{model.likes.toLocaleString()} 赞</Typography.Text></Space>} /></List.Item>} /></QueryState>}
      <Modal title={`下载 ${selected?.id ?? ''}`} open={Boolean(selected)} onCancel={() => setSelected(null)} footer={null} destroyOnClose><Form layout="vertical" initialValues={{ revision: 'main' }} onFinish={(values) => download.mutate(values)}><Form.Item name="revision" label="Revision" rules={[{ required: true }]}><Input /></Form.Item><Typography.Paragraph type="secondary">下载会作为后台任务执行，支持暂停、继续和断点续传。开始前服务端会检查仓库与磁盘路径。</Typography.Paragraph><Button block type="primary" htmlType="submit" loading={download.isPending}>创建下载任务</Button></Form></Modal>
    </div>
  )
}
