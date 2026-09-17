import { ApiOutlined, CopyOutlined, DeleteOutlined, FileTextOutlined, KeyOutlined, PictureOutlined, PlusOutlined, VideoCameraOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Descriptions, Form, Input, Modal, Popconfirm, Segmented, Space, Tag, Typography, message } from 'antd'
import { useState } from 'react'

import { api } from '../api/client'
import type { ApiKeyRecord, GatewayStats, UpstreamGateway, UpstreamTestResult } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { formatDate } from '../utils/format'


const upstreamSourceLabels = {
  database: '面板配置',
  environment: '环境变量',
  unset: '未配置',
} as const

export function GatewayPage() {
  const [exampleMode, setExampleMode] = useState<'text' | 'image' | 'video'>('text')
  const [open, setOpen] = useState(false)
  const [created, setCreated] = useState<ApiKeyRecord | null>(null)
  const [upstreamOpen, setUpstreamOpen] = useState(false)
  const [upstreamResult, setUpstreamResult] = useState<UpstreamTestResult | null>(null)
  const [upstreamError, setUpstreamError] = useState<string | null>(null)
  const [upstreamForm] = Form.useForm<{ base_url: string; api_key?: string }>()
  const queryClient = useQueryClient()
  const keys = useQuery({ queryKey: ['api-keys'], queryFn: () => api.get<ApiKeyRecord[]>('/api/keys') })
  const stats = useQuery({ queryKey: ['gateway-stats'], queryFn: () => api.get<GatewayStats>('/api/gateway/stats') })
  const upstream = useQuery({ queryKey: ['upstream-gateway'], queryFn: () => api.get<UpstreamGateway>('/api/gateway/upstream') })
  const saveUpstream = useMutation({
    mutationFn: (values: { base_url: string; api_key?: string }) =>
      api.put<UpstreamGateway>('/api/gateway/upstream', {
        base_url: values.base_url,
        ...(values.api_key ? { api_key: values.api_key } : {}),
      }),
    onSuccess: () => {
      message.success('上游网关已更新')
      setUpstreamOpen(false)
      setUpstreamResult(null)
      upstreamForm.resetFields()
      void queryClient.invalidateQueries({ queryKey: ['upstream-gateway'] })
    },
    onError: (error) => setUpstreamError(error instanceof Error ? error.message : '保存失败'),
  })
  const clearUpstream = useMutation({
    mutationFn: () => api.put<UpstreamGateway>('/api/gateway/upstream', { base_url: null }),
    onSuccess: () => {
      message.success('已清除上游网关')
      setUpstreamResult(null)
      void queryClient.invalidateQueries({ queryKey: ['upstream-gateway'] })
    },
  })
  const testUpstream = useMutation({
    mutationFn: () => api.post<UpstreamTestResult>('/api/gateway/upstream/test'),
    onSuccess: (result) => setUpstreamResult(result),
    onError: (error) => setUpstreamResult({ status: 'error', latency_ms: 0, model_count: 0, detail: error instanceof Error ? error.message : '测试失败' }),
  })

  const openUpstreamDialog = () => {
    setUpstreamError(null)
    upstreamForm.setFieldsValue({ base_url: upstream.data?.base_url ?? '', api_key: '' })
    setUpstreamOpen(true)
  }
  const create = useMutation({ mutationFn: (values: { name: string }) => api.post<ApiKeyRecord>('/api/keys', values), onSuccess: (value) => { setOpen(false); setCreated(value); queryClient.invalidateQueries({ queryKey: ['api-keys'] }) } })
  const revoke = useMutation({ mutationFn: (id: string) => api.delete(`/api/keys/${id}`), onSuccess: () => { message.success('API Key 已吊销'); queryClient.invalidateQueries({ queryKey: ['api-keys'] }) } })
  const baseUrl = `${window.location.origin}/v1`
  const examples = {
    text: `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${baseUrl}",\n    api_key="dgx_...",\n)\n\nresponse = client.chat.completions.create(\n    model="YOUR_MODEL",\n    messages=[{"role": "user", "content": "Hello"}],\n    stream=True,\n)`,
    image: `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${baseUrl}",\n    api_key="dgx_...",\n)\n\nresponse = client.chat.completions.create(\n    model="YOUR_VISION_MODEL",\n    messages=[{\n        "role": "user",\n        "content": [\n            {"type": "image_url", "image_url": {\n                "url": "https://example.com/image.jpg",\n                "detail": "high",\n            }},\n            {"type": "text", "text": "Describe this image."},\n        ],\n    }],\n    stream=True,\n)`,
    video: `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${baseUrl}",\n    api_key="dgx_...",\n)\n\nresponse = client.chat.completions.create(\n    model="YOUR_VIDEO_MODEL",\n    messages=[{\n        "role": "user",\n        "content": [\n            {"type": "video_url", "video_url": {\n                "url": "https://example.com/video.mp4",\n            }},\n            {"type": "text", "text": "Summarize this video."},\n        ],\n    }],\n    stream=True,\n)`,
  }
  const example = examples[exampleMode]
  return (
    <div className="page-stack">
      <PageHeader title="API 网关" description="使用一个 OpenAI 兼容入口访问所有健康部署" extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setOpen(true)}>创建 API Key</Button>} />
      <section className="gateway-summary"><Descriptions column={{ xs: 1, sm: 2, lg: 5 }} items={[{ key: 'base', label: 'Base URL', children: <Typography.Text code copyable>{baseUrl}</Typography.Text> }, { key: 'requests', label: '累计请求', children: stats.data?.total_requests ?? 0 }, { key: 'active', label: '当前并发', children: stats.data?.active_requests ?? 0 }, { key: 'throughput', label: 'Token 吞吐', children: `${stats.data?.tokens_per_second ?? 0}/s` }, { key: 'error', label: '错误率', children: `${((stats.data?.error_rate ?? 0) * 100).toFixed(1)}%` }]} /></section>
      <section className="content-section">
        <div className="section-heading">
          <div>
            <h2>上游网关</h2>
            <p>本机没有的模型会转发到该 OpenAI 兼容端点，并在模型列表中一并列出</p>
          </div>
          <Space wrap>
            <Button icon={<ApiOutlined />} loading={testUpstream.isPending} disabled={!upstream.data?.enabled} onClick={() => testUpstream.mutate()}>测试连接</Button>
            <Button type="primary" onClick={openUpstreamDialog}>{upstream.data?.enabled ? '修改配置' : '配置上游'}</Button>
          </Space>
        </div>
        <QueryState loading={upstream.isLoading} error={upstream.error}>
          {upstream.data?.enabled ? (
            <Descriptions column={{ xs: 1, sm: 2, lg: 3 }} items={[
              { key: 'url', label: 'Base URL', children: <Typography.Text code>{upstream.data.base_url}</Typography.Text> },
              { key: 'key', label: 'API Key', children: upstream.data.api_key_configured ? '已配置' : '未配置' },
              { key: 'source', label: '配置来源', children: <Tag>{upstreamSourceLabels[upstream.data.source]}</Tag> },
            ]} />
          ) : (
            <Alert type="info" showIcon message="未配置上游网关" description="未配置时，请求本机不存在的模型会返回 404。" />
          )}
          {upstreamResult && (
            <Alert
              className="upstream-test-result"
              type={upstreamResult.status === 'ok' ? 'success' : upstreamResult.status === 'unset' ? 'info' : 'error'}
              showIcon
              message={
                upstreamResult.status === 'ok'
                  ? `连接正常 · 发现 ${upstreamResult.model_count} 个模型 · ${upstreamResult.latency_ms} ms`
                  : upstreamResult.status === 'unset' ? '尚未配置上游网关' : '上游网关不可用'
              }
              description={upstreamResult.detail ?? undefined}
            />
          )}
        </QueryState>
      </section>
      <section className="content-section"><div className="section-heading"><div><h2>访问密钥</h2><p>密钥只在创建后显示一次</p></div></div><QueryState loading={keys.isLoading} error={keys.error} empty={!keys.data?.length}><ResponsiveDataView data={keys.data ?? []} rowKey="id" columns={[{ title: '名称', dataIndex: 'name' }, { title: '前缀', dataIndex: 'prefix', render: (value) => <Typography.Text code>{value}...</Typography.Text> }, { title: '创建时间', dataIndex: 'created_at', render: formatDate }, { title: '最后使用', dataIndex: 'last_used_at', render: formatDate }, { title: '状态', dataIndex: 'revoked_at', render: (value) => value ? '已吊销' : '有效' }, { title: '', render: (_, item) => !item.revoked_at && <Popconfirm title={`吊销 ${item.name}`} description="使用该密钥的客户端将立即无法调用网关。" onConfirm={() => revoke.mutate(item.id)}><Button danger size="small" icon={<DeleteOutlined />}>吊销</Button></Popconfirm> }]} renderMobile={(item) => <div className="mobile-record"><div className="primary-cell"><strong>{item.name}</strong><Typography.Text code>{item.prefix}...</Typography.Text></div><dl><div><dt>状态</dt><dd>{item.revoked_at ? '已吊销' : '有效'}</dd></div><div><dt>最后使用</dt><dd>{formatDate(item.last_used_at)}</dd></div></dl>{!item.revoked_at && <Popconfirm title={`吊销 ${item.name}`} description="使用该密钥的客户端将立即无法调用网关。" onConfirm={() => revoke.mutate(item.id)}><Button danger block icon={<DeleteOutlined />}>吊销</Button></Popconfirm>}</div>} /></QueryState></section>
      <section className="code-section"><div className="section-heading"><div><h2>Python SDK</h2><p>修改模型名称即可调用</p></div><Space wrap><Segmented value={exampleMode} onChange={(value) => setExampleMode(value as 'text' | 'image' | 'video')} options={[{ value: 'text', label: '文本', icon: <FileTextOutlined /> }, { value: 'image', label: '图片', icon: <PictureOutlined /> }, { value: 'video', label: '视频', icon: <VideoCameraOutlined /> }]} /><Button icon={<CopyOutlined />} aria-label="复制 API 示例" onClick={() => navigator.clipboard.writeText(example)} /></Space></div><pre><code>{example}</code></pre></section>
      <Modal title="创建 API Key" open={open} footer={null} onCancel={() => setOpen(false)} destroyOnClose><Form layout="vertical" onFinish={(values) => create.mutate(values)}><Form.Item name="name" label="名称" rules={[{ required: true }]}><Input prefix={<KeyOutlined />} placeholder="例如：开发机 SDK" /></Form.Item><Button type="primary" htmlType="submit" loading={create.isPending} block>创建</Button></Form></Modal>
      <Modal title="API Key 已创建" open={Boolean(created)} onCancel={() => setCreated(null)} footer={<Button type="primary" onClick={() => setCreated(null)}>我已保存</Button>}><Alert type="warning" showIcon message="该密钥只显示一次" description="关闭窗口后无法再次查看，只能吊销并重新创建。" /><Input.TextArea className="secret-output" value={created?.key} autoSize readOnly /><Button icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(created?.key ?? '')}>复制密钥</Button></Modal>
      <Modal
        title="上游网关"
        open={upstreamOpen}
        footer={null}
        onCancel={() => setUpstreamOpen(false)}
        destroyOnClose
      >
        <Form
          form={upstreamForm}
          layout="vertical"
          onFinish={(values) => saveUpstream.mutate(values)}
        >
          <Form.Item
            name="base_url"
            label="Base URL"
            rules={[
              { required: true, message: '请输入上游网关地址' },
              { validator: (_rule, value: string) => (!value || /^https?:\/\//.test(value) ? Promise.resolve() : Promise.reject(new Error('地址必须以 http:// 或 https:// 开头'))) },
            ]}
            extra="可填写 https://host 或 https://host/v1，两种写法都会转发到同一个 /v1 路径"
          >
            <Input placeholder="https://upstream.example/v1" />
          </Form.Item>
          <Form.Item name="api_key" label="API Key" extra="留空表示不修改已保存的密钥">
            <Input.Password autoComplete="off" placeholder="留空表示不修改" />
          </Form.Item>
          {upstreamError && <Alert type="error" showIcon message={upstreamError} className="upstream-test-result" />}
          <Space>
            <Button type="primary" htmlType="submit" loading={saveUpstream.isPending}>保存</Button>
            {upstream.data?.enabled && (
              <Popconfirm title="清除上游网关" description="清除后，本机不存在的模型将重新返回 404。" onConfirm={() => clearUpstream.mutate()}>
                <Button danger loading={clearUpstream.isPending}>清除配置</Button>
              </Popconfirm>
            )}
          </Space>
        </Form>
      </Modal>
    </div>
  )
}
