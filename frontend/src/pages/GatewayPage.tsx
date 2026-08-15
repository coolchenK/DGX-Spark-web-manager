import { CopyOutlined, DeleteOutlined, KeyOutlined, PlusOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Descriptions, Form, Input, Modal, Popconfirm, Table, Typography, message } from 'antd'
import { useState } from 'react'

import { api } from '../api/client'
import type { ApiKeyRecord, GatewayStats } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { formatDate } from '../utils/format'


export function GatewayPage() {
  const [open, setOpen] = useState(false)
  const [created, setCreated] = useState<ApiKeyRecord | null>(null)
  const queryClient = useQueryClient()
  const keys = useQuery({ queryKey: ['api-keys'], queryFn: () => api.get<ApiKeyRecord[]>('/api/keys') })
  const stats = useQuery({ queryKey: ['gateway-stats'], queryFn: () => api.get<GatewayStats>('/api/gateway/stats') })
  const create = useMutation({ mutationFn: (values: { name: string }) => api.post<ApiKeyRecord>('/api/keys', values), onSuccess: (value) => { setOpen(false); setCreated(value); queryClient.invalidateQueries({ queryKey: ['api-keys'] }) } })
  const revoke = useMutation({ mutationFn: (id: string) => api.delete(`/api/keys/${id}`), onSuccess: () => { message.success('API Key 已吊销'); queryClient.invalidateQueries({ queryKey: ['api-keys'] }) } })
  const baseUrl = `${window.location.origin}/v1`
  const example = `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${baseUrl}",\n    api_key="dgx_...",\n)\n\nresponse = client.chat.completions.create(\n    model="YOUR_MODEL",\n    messages=[{"role": "user", "content": "Hello"}],\n    stream=True,\n)`
  return (
    <div className="page-stack">
      <PageHeader title="API 网关" description="使用一个 OpenAI 兼容入口访问所有健康部署" extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>创建 API Key</Button>} />
      <section className="gateway-summary"><Descriptions column={{ xs: 1, sm: 2, lg: 4 }} items={[{ key: 'base', label: 'Base URL', children: <Typography.Text code copyable>{baseUrl}</Typography.Text> }, { key: 'requests', label: '累计请求', children: stats.data?.total_requests ?? 0 }, { key: 'latency', label: '平均延迟', children: `${stats.data?.average_latency_ms ?? 0} ms` }, { key: 'error', label: '错误率', children: `${((stats.data?.error_rate ?? 0) * 100).toFixed(1)}%` }]} /></section>
      <section className="content-section"><div className="section-heading"><div><h2>访问密钥</h2><p>密钥只在创建后显示一次</p></div></div><QueryState loading={keys.isLoading} error={keys.error} empty={!keys.data?.length}><Table size="small" rowKey="id" dataSource={keys.data} pagination={false} columns={[{ title: '名称', dataIndex: 'name' }, { title: '前缀', dataIndex: 'prefix', render: (value) => <Typography.Text code>{value}...</Typography.Text> }, { title: '创建时间', dataIndex: 'created_at', render: formatDate }, { title: '最后使用', dataIndex: 'last_used_at', render: formatDate }, { title: '状态', dataIndex: 'revoked_at', render: (value) => value ? '已吊销' : '有效' }, { title: '', render: (_, item) => !item.revoked_at && <Popconfirm title={`吊销 ${item.name}`} description="使用该密钥的客户端将立即无法调用网关。" onConfirm={() => revoke.mutate(item.id)}><Button danger size="small" icon={<DeleteOutlined />}>吊销</Button></Popconfirm> }]} /></QueryState></section>
      <section className="code-section"><div className="section-heading"><div><h2>Python SDK</h2><p>修改模型名称即可调用</p></div><Button icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(example)}>复制</Button></div><pre><code>{example}</code></pre></section>
      <Modal title="创建 API Key" open={open} footer={null} onCancel={() => setOpen(false)} destroyOnClose><Form layout="vertical" onFinish={(values) => create.mutate(values)}><Form.Item name="name" label="名称" rules={[{ required: true }]}><Input prefix={<KeyOutlined />} placeholder="例如：开发机 SDK" /></Form.Item><Button type="primary" htmlType="submit" loading={create.isPending} block>创建</Button></Form></Modal>
      <Modal title="API Key 已创建" open={Boolean(created)} onCancel={() => setCreated(null)} footer={<Button type="primary" onClick={() => setCreated(null)}>我已保存</Button>}><Alert type="warning" showIcon message="该密钥只显示一次" description="关闭窗口后无法再次查看，只能吊销并重新创建。" /><Input.TextArea className="secret-output" value={created?.key} autoSize readOnly /><Button icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(created?.key ?? '')}>复制密钥</Button></Modal>
    </div>
  )
}
