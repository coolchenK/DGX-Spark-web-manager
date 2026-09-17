import { ApiOutlined, CheckCircleFilled, DeleteOutlined, EditOutlined, MinusCircleOutlined, PlusOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Form, Input, InputNumber, Modal, Popconfirm, Space, Switch, Tag, message } from 'antd'
import { useState } from 'react'

import { api } from '../api/client'
import type { Provider, ProviderProbeResult } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatDate } from '../utils/format'


interface ProviderForm {
  name: string
  base_url: string
  api_key?: string
  default_model: string
  timeout_seconds: number
  enabled: boolean
  header_pairs?: Array<{ key?: string; value?: string }>
}

function ProbeError({ title, error }: { title: string; error?: string }) {
  return (
    <Alert
      className="provider-probe-error"
      type="error"
      showIcon
      message={title}
      description={error && (
        <details>
          <summary>技术详情</summary>
          <code>{error}</code>
        </details>
      )}
    />
  )
}

function ProviderProbeSummary({ result }: { result: ProviderProbeResult }) {
  const connection = result.connection
  const defaultModel = result.default_model
  if (!connection && !defaultModel) return null
  return (
    <div className="provider-probe" aria-label="Provider 测试详情">
      {connection?.status === 'healthy'
        ? <div className="provider-probe-ok"><CheckCircleFilled /><span>API 连接正常</span>{connection.models_seen != null && <small>已发现 {connection.models_seen} 个模型</small>}</div>
        : connection?.status === 'failed' && <ProbeError title="API 连接失败" error={connection.error} />}
      {defaultModel?.status === 'healthy'
        ? <div className="provider-probe-ok"><CheckCircleFilled /><span>默认模型可用</span><small>{defaultModel.model}</small></div>
        : defaultModel?.status === 'failed'
          ? <ProbeError title="默认模型不可用" error={defaultModel.error} />
          : defaultModel?.status === 'not_tested' && <div className="provider-probe-muted"><MinusCircleOutlined /><span>默认模型未测试</span></div>}
    </div>
  )
}


export function ProvidersPage() {
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState<Provider | null>(null)
  const [form] = Form.useForm<ProviderForm>()
  const queryClient = useQueryClient()
  const providers = useQuery({ queryKey: ['providers'], queryFn: () => api.get<Provider[]>('/api/providers') })
  const save = useMutation({
    mutationFn: (values: ProviderForm) => {
      const headers = Object.fromEntries((values.header_pairs ?? []).filter((item) => item.key).map((item) => [item.key!, item.value ?? '']))
      const { header_pairs: _, ...fields } = values
      const payload = { ...fields, headers }
      if (!payload.api_key) delete payload.api_key
      return editing ? api.patch(`/api/providers/${editing.id}`, payload) : api.post('/api/providers', payload)
    },
    onSuccess: () => {
      setOpen(false)
      setEditing(null)
      message.success('在线 AI 服务已保存')
      queryClient.invalidateQueries({ queryKey: ['providers'] })
    },
  })
  const test = useMutation({
    mutationFn: (id: string) => api.post<ProviderProbeResult>(`/api/providers/${id}/test`),
    onSuccess: (result, id) => {
      queryClient.setQueryData<Provider[]>(['providers'], (current = []) => current.map((provider) => provider.id === id ? { ...provider, last_test_status: result.status ?? 'failed', last_test_result: result, last_tested_at: new Date().toISOString() } : provider))
      message[result.status === 'healthy' ? 'success' : 'error'](result.status === 'healthy' ? '连接与默认模型测试成功' : 'Provider 测试未通过')
      void queryClient.invalidateQueries({ queryKey: ['providers'] })
      void queryClient.invalidateQueries({ queryKey: ['diagnostics'] })
    },
    onError: (error) => message.error(error instanceof Error ? error.message : 'Provider 测试失败'),
  })
  const remove = useMutation({ mutationFn: (id: string) => api.delete(`/api/providers/${id}`), onSuccess: () => { message.success('服务已删除'); queryClient.invalidateQueries({ queryKey: ['providers'] }) } })

  const createProvider = () => {
    setEditing(null)
    form.setFieldsValue({ name: '', base_url: '', api_key: '', default_model: '', timeout_seconds: 60, enabled: true, header_pairs: [] })
    setOpen(true)
  }
  const editProvider = (provider: Provider) => {
    setEditing(provider)
    form.setFieldsValue({
      name: provider.name,
      base_url: provider.base_url,
      api_key: '',
      default_model: provider.default_model,
      timeout_seconds: provider.timeout_seconds,
      enabled: provider.enabled,
      header_pairs: Object.entries(provider.headers).map(([key, value]) => ({ key, value })),
    })
    setOpen(true)
  }
  const actions = (provider: Provider) => <Space wrap><Button aria-label="测试连接" size="small" icon={<ApiOutlined />} loading={test.isPending && test.variables === provider.id} onClick={() => test.mutate(provider.id)}>测试连接</Button><Button size="small" icon={<EditOutlined />} onClick={() => editProvider(provider)}>编辑</Button><Popconfirm title={`删除 ${provider.name}`} onConfirm={() => remove.mutate(provider.id)}><Button size="small" danger icon={<DeleteOutlined />} aria-label="删除服务" /></Popconfirm></Space>

  return (
    <div className="page-stack">
      <PageHeader title="在线 AI 服务" description="配置 OpenAI 兼容接口，为部署建议和故障诊断提供模型" extra={<Button type="primary" icon={<PlusOutlined />} onClick={createProvider}>添加服务</Button>} />
      <QueryState loading={providers.isLoading} error={providers.error} empty={!providers.data?.length}>
        <ResponsiveDataView data={providers.data ?? []} rowKey="id" columns={[
          { title: '名称', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.base_url}</small></div> },
          { title: '默认模型', dataIndex: 'default_model' },
          { title: '密钥', dataIndex: 'api_key_masked', render: (value) => <Tag>{value}</Tag> },
          { title: '状态', dataIndex: 'last_test_status', width: 280, render: (_, item) => <div className="provider-status"><StatusBadge status={item.last_test_status ?? 'unknown'} /><ProviderProbeSummary result={item.last_test_result} /></div> },
          { title: '最后测试', dataIndex: 'last_tested_at', render: formatDate },
          { title: '操作', render: (_, item) => actions(item) },
        ]} renderMobile={(item) => <div className="mobile-record"><Space><strong>{item.name}</strong><StatusBadge status={item.last_test_status ?? 'unknown'} /></Space><span>{item.base_url}</span><dl><div><dt>默认模型</dt><dd>{item.default_model}</dd></div><div><dt>最后测试</dt><dd>{formatDate(item.last_tested_at)}</dd></div></dl><ProviderProbeSummary result={item.last_test_result} />{actions(item)}</div>} />
      </QueryState>
      <Modal title={editing ? '编辑在线 AI 服务' : '添加在线 AI 服务'} open={open} footer={null} onCancel={() => { setOpen(false); setEditing(null) }} destroyOnHidden>
        <Form form={form} layout="vertical" onFinish={(values) => save.mutate(values)}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="base_url" label="Base URL" rules={[{ required: true, type: 'url' }]}><Input placeholder="https://api.example.com/v1" /></Form.Item>
          <Form.Item name="api_key" label={editing ? 'API Key（留空则保持不变）' : 'API Key'} rules={editing ? [] : [{ required: true }]}><Input.Password /></Form.Item>
          <Form.Item name="default_model" label="默认模型" rules={[{ required: true }]}><Input /></Form.Item>
          <div className="form-grid"><Form.Item name="timeout_seconds" label="超时（秒)"><InputNumber min={5} max={600} /></Form.Item><Form.Item name="enabled" label="启用" valuePropName="checked"><Switch /></Form.Item></div>
          <Form.Item label="自定义请求头">
            <Form.List name="header_pairs">{(fields, { add, remove: removeHeader }) => <Space direction="vertical" className="provider-headers">{fields.map((field) => <Space.Compact key={field.key} block><Form.Item name={[field.name, 'key']} noStyle><Input aria-label="请求头名称" placeholder="Header" /></Form.Item><Form.Item name={[field.name, 'value']} noStyle><Input aria-label="请求头值" placeholder="Value" /></Form.Item><Button icon={<MinusCircleOutlined />} aria-label="删除请求头" onClick={() => removeHeader(field.name)} /></Space.Compact>)}<Button type="dashed" icon={<PlusOutlined />} onClick={() => add()}>添加请求头</Button></Space>}</Form.List>
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={save.isPending} block>保存并加密</Button>
        </Form>
      </Modal>
    </div>
  )
}
