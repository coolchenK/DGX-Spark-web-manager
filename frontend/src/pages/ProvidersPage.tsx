import { ApiOutlined, DeleteOutlined, PlusOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Form, Input, InputNumber, Modal, Popconfirm, Space, Switch, Table, Tag, message } from 'antd'
import { useState } from 'react'

import { api } from '../api/client'
import type { Provider } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { StatusBadge } from '../components/StatusBadge'
import { formatDate } from '../utils/format'


export function ProvidersPage() {
  const [open, setOpen] = useState(false)
  const queryClient = useQueryClient()
  const providers = useQuery({ queryKey: ['providers'], queryFn: () => api.get<Provider[]>('/api/providers') })
  const create = useMutation({ mutationFn: (values: Record<string, unknown>) => api.post('/api/providers', values), onSuccess: () => { setOpen(false); message.success('在线 AI 服务已保存'); queryClient.invalidateQueries({ queryKey: ['providers'] }) } })
  const test = useMutation({ mutationFn: (id: string) => api.post<{ status: string }>(`/api/providers/${id}/test`), onSuccess: (result) => { message[result.status === 'healthy' ? 'success' : 'error'](result.status === 'healthy' ? '连接成功' : '连接失败'); queryClient.invalidateQueries({ queryKey: ['providers'] }) } })
  const remove = useMutation({ mutationFn: (id: string) => api.delete(`/api/providers/${id}`), onSuccess: () => { message.success('服务已删除'); queryClient.invalidateQueries({ queryKey: ['providers'] }) } })
  return (
    <div className="page-stack"><PageHeader title="在线 AI 服务" description="配置 OpenAI 兼容接口，为部署建议和故障诊断提供模型" extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>添加服务</Button>} /><QueryState loading={providers.isLoading} error={providers.error} empty={!providers.data?.length}><Table size="small" rowKey="id" dataSource={providers.data} columns={[{ title: '名称', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.base_url}</small></div> }, { title: '默认模型', dataIndex: 'default_model' }, { title: '密钥', dataIndex: 'api_key_masked', render: (value) => <Tag>{value}</Tag> }, { title: '状态', dataIndex: 'last_test_status', render: (value) => <StatusBadge status={value ?? 'unknown'} /> }, { title: '最后测试', dataIndex: 'last_tested_at', render: formatDate }, { title: '操作', render: (_, item) => <Space><Button size="small" icon={<ApiOutlined />} loading={test.isPending} onClick={() => test.mutate(item.id)}>测试</Button><Popconfirm title={`删除 ${item.name}`} onConfirm={() => remove.mutate(item.id)}><Button size="small" danger icon={<DeleteOutlined />} aria-label="删除服务" /></Popconfirm></Space> }]} /></QueryState><Modal title="添加在线 AI 服务" open={open} footer={null} onCancel={() => setOpen(false)} destroyOnClose><Form layout="vertical" initialValues={{ timeout_seconds: 60, enabled: true }} onFinish={(values) => create.mutate({ ...values, headers: {} })}><Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item><Form.Item name="base_url" label="Base URL" rules={[{ required: true, type: 'url' }]}><Input placeholder="https://api.example.com/v1" /></Form.Item><Form.Item name="api_key" label="API Key" rules={[{ required: true }]}><Input.Password /></Form.Item><Form.Item name="default_model" label="默认模型" rules={[{ required: true }]}><Input /></Form.Item><div className="form-grid"><Form.Item name="timeout_seconds" label="超时（秒）"><InputNumber min={5} max={600} /></Form.Item><Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item></div><Button type="primary" htmlType="submit" loading={create.isPending} block>保存并加密</Button></Form></Modal></div>
  )
}
