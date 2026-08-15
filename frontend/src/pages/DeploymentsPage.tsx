import { DeleteOutlined, FileTextOutlined, PauseCircleOutlined, PlayCircleOutlined, PlusOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Descriptions, Drawer, Flex, Form, Input, InputNumber, Popconfirm, Segmented, Select, Slider, Space, Steps, Switch, Tag, Typography, message } from 'antd'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { api } from '../api/client'
import type { Deployment, ModelAsset, TaskRecord } from '../api/types'
import { LogViewer } from '../components/LogViewer'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'


interface DeployForm { name: string; model_id: string; model_path: string; api_model_name: string; runtime: 'vllm' | 'sglang'; image: string; port: number; context_length: number; memory_fraction: number; max_concurrency: number; trust_remote_code: boolean }


export function DeploymentsPage() {
  const [form] = Form.useForm<DeployForm>()
  const [searchParams] = useSearchParams()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null)
  const [logsFor, setLogsFor] = useState<Deployment | null>(null)
  const queryClient = useQueryClient()
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.get<Deployment[]>('/api/deployments'), refetchInterval: 8_000 })
  const models = useQuery({ queryKey: ['models'], queryFn: () => api.get<ModelAsset[]>('/api/models') })
  const logs = useQuery({ queryKey: ['deployment-logs', logsFor?.id], queryFn: () => api.get<{ logs: string }>(`/api/deployments/${logsFor?.id}/logs?tail=1000`), enabled: Boolean(logsFor), refetchInterval: logsFor ? 5_000 : false })
  useEffect(() => { if (searchParams.get('model')) setDrawerOpen(true) }, [searchParams])
  const previewMutation = useMutation({ mutationFn: (values: DeployForm) => api.post<Record<string, unknown>>('/api/deployments/preview', values), onSuccess: setPreview })
  const createMutation = useMutation({ mutationFn: (values: DeployForm) => api.post<TaskRecord>('/api/deployments', values), onSuccess: () => { message.success('部署任务已创建'); setDrawerOpen(false); setPreview(null); queryClient.invalidateQueries({ queryKey: ['tasks'] }) } })
  const action = useMutation({ mutationFn: ({ id, action }: { id: string; action: string }) => api.post(`/api/deployments/${id}/${action}`), onSuccess: () => { message.success('操作已加入任务队列'); queryClient.invalidateQueries() } })
  const runtime = Form.useWatch('runtime', form) ?? 'vllm'
  const selectModel = (modelId: string) => { const model = models.data?.find((item) => item.id === modelId); if (model) form.setFieldsValue({ name: model.name.split('/').pop(), model_path: model.local_path, api_model_name: model.alias ?? model.name.split('/').pop()?.toLowerCase(), model_id: model.id }) }
  return (
    <div className="page-stack">
      <PageHeader title="部署实例" description="管理已发现容器，并通过受控适配器创建新服务" extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setDrawerOpen(true)}>新建部署</Button>} />
      <QueryState loading={deployments.isLoading} error={deployments.error} empty={!deployments.data?.length}><ResponsiveDataView data={deployments.data ?? []} rowKey="id" columns={[
        { title: '实例', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.api_model_name}</small></div> },
        { title: '运行时', dataIndex: 'runtime', width: 100, render: (value) => <Tag>{value}</Tag> },
        { title: '端点', dataIndex: 'endpoint_url' },
        { title: '所有权', dataIndex: 'managed', width: 90, render: (value) => value ? '管理器' : '已发现' },
        { title: '状态', dataIndex: 'health', width: 100, render: (value) => <StatusBadge status={value} /> },
        { title: '操作', width: 210, render: (_, item) => <Space><Button size="small" icon={<FileTextOutlined />} onClick={() => setLogsFor(item)} aria-label="查看日志" />{item.status === 'running' ? <Button size="small" icon={<PauseCircleOutlined />} onClick={() => action.mutate({ id: item.id, action: 'stop' })}>停止</Button> : <Button size="small" icon={<PlayCircleOutlined />} onClick={() => action.mutate({ id: item.id, action: 'start' })}>启动</Button>}<Button size="small" icon={<ReloadOutlined />} onClick={() => action.mutate({ id: item.id, action: 'restart' })} aria-label="重启" />{item.managed && <Popconfirm title={`删除部署 ${item.name}`} description="容器会被删除，模型文件保留。" onConfirm={() => action.mutate({ id: item.id, action: 'delete' })}><Button size="small" danger icon={<DeleteOutlined />} aria-label="删除部署" /></Popconfirm>}</Space> },
      ]} renderMobile={(item) => <div className="mobile-record"><Flex justify="space-between"><strong>{item.name}</strong><StatusBadge status={item.health} /></Flex><Typography.Text type="secondary">{item.api_model_name}</Typography.Text><dl><div><dt>运行时</dt><dd>{item.runtime}</dd></div><div><dt>端点</dt><dd>{item.endpoint_url}</dd></div></dl><Space wrap><Button icon={<FileTextOutlined />} onClick={() => setLogsFor(item)}>日志</Button><Button onClick={() => action.mutate({ id: item.id, action: item.status === 'running' ? 'stop' : 'start' })}>{item.status === 'running' ? '停止' : '启动'}</Button><Button icon={<ReloadOutlined />} onClick={() => action.mutate({ id: item.id, action: 'restart' })}>重启</Button></Space></div>} /></QueryState>
      <Drawer title="新建模型部署" width={620} open={drawerOpen} onClose={() => { setDrawerOpen(false); setPreview(null) }} extra={<Steps size="small" current={preview ? 1 : 0} items={[{ title: '配置' }, { title: '确认' }]} />}>
        <Form form={form} layout="vertical" initialValues={{ model_id: searchParams.get('model') ?? undefined, runtime: 'vllm', image: 'vllm/vllm-openai:v0.27.1', port: 8100, context_length: 32768, memory_fraction: .8, max_concurrency: 8, trust_remote_code: false }} onFinish={(values) => preview ? createMutation.mutate(values) : previewMutation.mutate(values)} onValuesChange={(changed) => { setPreview(null); if ('runtime' in changed) form.setFieldValue('image', changed.runtime === 'vllm' ? 'vllm/vllm-openai:v0.27.1' : 'sglang-inkling:specforge') }}>
          {!preview ? <><Form.Item name="model_id" label="模型" rules={[{ required: true }]}><Select loading={models.isLoading} options={models.data?.map((item) => ({ value: item.id, label: item.name }))} onChange={selectModel} /></Form.Item><Form.Item name="name" label="部署名称" rules={[{ required: true }]}><Input /></Form.Item><Form.Item name="model_path" label="模型路径" rules={[{ required: true }]}><Input readOnly /></Form.Item><Form.Item name="api_model_name" label="API 模型名称" rules={[{ required: true }]}><Input /></Form.Item><Form.Item name="runtime" label="推理运行时"><Segmented block options={[{ label: 'vLLM', value: 'vllm' }, { label: 'SGLang', value: 'sglang' }]} /></Form.Item><Form.Item name="image" label="ARM64 镜像" rules={[{ required: true }]}><Select options={(runtime === 'vllm' ? ['vllm/vllm-openai:v0.27.1'] : ['sglang-inkling:specforge', 'lmsysorg/sglang:dev-cu13-inkling-dspark']).map((value) => ({ value, label: value }))} /></Form.Item><div className="form-grid"><Form.Item name="port" label="主机端口"><InputNumber min={1024} max={65535} /></Form.Item><Form.Item name="context_length" label="上下文长度"><InputNumber min={1024} step={1024} /></Form.Item><Form.Item name="max_concurrency" label="最大并发"><InputNumber min={1} max={1024} /></Form.Item><Form.Item name="trust_remote_code" label="信任远程代码" valuePropName="checked"><Switch /></Form.Item></div><Form.Item name="memory_fraction" label="统一内存比例"><Slider min={.05} max={.98} step={.01} tooltip={{ formatter: (value) => `${Math.round((value ?? 0) * 100)}%` }} /></Form.Item></> : <Descriptions bordered size="small" column={1} items={Object.entries(preview).filter(([, value]) => typeof value !== 'object').map(([key, value]) => ({ key, label: key, children: String(value) }))} />}
          <Button type="primary" htmlType="submit" block loading={previewMutation.isPending || createMutation.isPending}>{preview ? '确认并创建任务' : '检查部署配置'}</Button>
        </Form>
      </Drawer>
      <Drawer title={`${logsFor?.name ?? ''} 日志`} width={760} open={Boolean(logsFor)} onClose={() => setLogsFor(null)}><QueryState loading={logs.isLoading} error={logs.error}><LogViewer value={logs.data?.logs ?? ''} filename={`${logsFor?.name}.log`} /></QueryState></Drawer>
    </div>
  )
}
