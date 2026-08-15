import {
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  FileTextOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Descriptions,
  Drawer,
  Flex,
  Form,
  Input,
  InputNumber,
  List,
  Popconfirm,
  Segmented,
  Select,
  Slider,
  Space,
  Steps,
  Switch,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { api } from '../api/client'
import type { Deployment, ModelAsset, TaskRecord } from '../api/types'
import { LogViewer } from '../components/LogViewer'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { deploymentToFormValues, type DeploymentFormValues } from '../utils/deployments'
import { formatBytes } from '../utils/format'


const defaultValues: Partial<DeploymentFormValues> = {
  runtime: 'vllm',
  image: 'vllm/vllm-openai:v0.27.1',
  port: 8100,
  context_length: 32768,
  memory_fraction: 0.8,
  max_concurrency: 8,
  trust_remote_code: false,
}


export function DeploymentsPage() {
  const [form] = Form.useForm<DeploymentFormValues>()
  const [searchParams] = useSearchParams()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [editingDeployment, setEditingDeployment] = useState<Deployment | null>(null)
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null)
  const [logsFor, setLogsFor] = useState<Deployment | null>(null)
  const queryClient = useQueryClient()
  const deployments = useQuery({
    queryKey: ['deployments'],
    queryFn: () => api.get<Deployment[]>('/api/deployments'),
    refetchInterval: 8_000,
  })
  const models = useQuery({ queryKey: ['models'], queryFn: () => api.get<ModelAsset[]>('/api/models') })
  const logs = useQuery({
    queryKey: ['deployment-logs', logsFor?.id],
    queryFn: () => api.get<{ logs: string }>(`/api/deployments/${logsFor?.id}/logs?tail=1000`),
    enabled: Boolean(logsFor),
    refetchInterval: logsFor ? 5_000 : false,
  })

  const selectModel = (modelId: string) => {
    const model = models.data?.find((item) => item.id === modelId)
    if (!model) return
    const shortName = model.name.split('/').pop() ?? model.name
    form.setFieldsValue({
      name: shortName,
      model_path: model.local_path,
      api_model_name: model.alias ?? shortName.toLowerCase(),
      model_id: model.id,
      quantization: model.quantization as DeploymentFormValues['quantization'],
    })
  }

  const openCreate = () => {
    setEditingDeployment(null)
    setPreview(null)
    form.resetFields()
    form.setFieldsValue(defaultValues)
    setDrawerOpen(true)
  }

  const openFromDeployment = (deployment: Deployment, mode: 'edit' | 'clone') => {
    const model = models.data?.find((item) => item.id === deployment.model_id)
    if (!model) {
      message.error('该实例未关联可用的本地模型，无法编辑或克隆')
      return
    }
    setEditingDeployment(mode === 'edit' ? deployment : null)
    setPreview(null)
    form.setFieldsValue(deploymentToFormValues(deployment, model, mode))
    setDrawerOpen(true)
  }

  useEffect(() => {
    const modelId = searchParams.get('model')
    if (modelId && models.data?.some((item) => item.id === modelId)) {
      openCreate()
      selectModel(modelId)
    }
    // The URL model selector should only initialize the drawer when inventory arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.data, searchParams])

  const closeDrawer = () => {
    setDrawerOpen(false)
    setEditingDeployment(null)
    setPreview(null)
  }
  const previewMutation = useMutation({
    mutationFn: (values: DeploymentFormValues) =>
      api.post<Record<string, unknown>>('/api/deployments/preview', values),
    onSuccess: setPreview,
    onError: (error: Error) => message.error(error.message),
  })
  const saveMutation = useMutation({
    mutationFn: (values: DeploymentFormValues) => editingDeployment
      ? api.patch<TaskRecord>(`/api/deployments/${editingDeployment.id}`, values)
      : api.post<TaskRecord>('/api/deployments', values),
    onSuccess: () => {
      message.success(editingDeployment ? '部署更新任务已创建' : '部署任务已创建')
      closeDrawer()
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error: Error) => message.error(error.message),
  })
  const action = useMutation({
    mutationFn: ({ id, actionName }: { id: string; actionName: string }) =>
      api.post(`/api/deployments/${id}/${actionName}`),
    onSuccess: () => {
      message.success('操作已加入任务队列')
      queryClient.invalidateQueries()
    },
    onError: (error: Error) => message.error(error.message),
  })
  const runtime = Form.useWatch('runtime', form) ?? 'vllm'
  const compatibility = preview?.compatibility as {
    compatible: boolean
    architectures: string[]
    reasons: string[]
  } | undefined
  const operations = preview?.operations as string[] | undefined

  const operationButtons = (item: Deployment, mobile = false) => (
    <Space wrap>
      <Button size={mobile ? 'middle' : 'small'} icon={<FileTextOutlined />} onClick={() => setLogsFor(item)}>
        {mobile ? '日志' : null}
      </Button>
      <Button
        size={mobile ? 'middle' : 'small'}
        loading={action.isPending}
        icon={item.status === 'running' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
        onClick={() => action.mutate({ id: item.id, actionName: item.status === 'running' ? 'stop' : 'start' })}
      >
        {item.status === 'running' ? '停止' : '启动'}
      </Button>
      <Tooltip title="重启实例">
        <Button
          size={mobile ? 'middle' : 'small'}
          icon={<ReloadOutlined />}
          loading={action.isPending}
          onClick={() => action.mutate({ id: item.id, actionName: 'restart' })}
          aria-label="重启实例"
        />
      </Tooltip>
      {item.managed && <>
        <Tooltip title="编辑部署参数">
          <Button size={mobile ? 'middle' : 'small'} icon={<EditOutlined />} onClick={() => openFromDeployment(item, 'edit')} aria-label="编辑部署参数" />
        </Tooltip>
        <Tooltip title="克隆部署">
          <Button size={mobile ? 'middle' : 'small'} icon={<CopyOutlined />} onClick={() => openFromDeployment(item, 'clone')} aria-label="克隆部署" />
        </Tooltip>
        <Popconfirm
          title={`删除部署 ${item.name}`}
          description="容器会被删除，模型文件保留。"
          onConfirm={() => action.mutate({ id: item.id, actionName: 'delete' })}
        >
          <Button size={mobile ? 'middle' : 'small'} danger icon={<DeleteOutlined />} aria-label="删除部署" />
        </Popconfirm>
      </>}
    </Space>
  )

  return (
    <div className="page-stack">
      <PageHeader
        title="部署实例"
        description="管理已发现容器，并通过受控适配器创建新服务"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>新建部署</Button>}
      />
      <QueryState loading={deployments.isLoading} error={deployments.error} empty={!deployments.data?.length}>
        <ResponsiveDataView data={deployments.data ?? []} rowKey="id" columns={[
          { title: '实例', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.api_model_name}</small></div> },
          { title: '运行时', dataIndex: 'runtime', width: 100, render: (value) => <Tag>{value}</Tag> },
          { title: '端点', dataIndex: 'endpoint_url' },
          { title: '所有权', dataIndex: 'managed', width: 90, render: (value) => value ? '管理器' : '已发现' },
          { title: '状态', dataIndex: 'health', width: 100, render: (value) => <StatusBadge status={value} /> },
          { title: '操作', width: 330, render: (_, item) => operationButtons(item) },
        ]} renderMobile={(item) => (
          <div className="mobile-record">
            <Flex justify="space-between"><strong>{item.name}</strong><StatusBadge status={item.health} /></Flex>
            <Typography.Text type="secondary">{item.api_model_name}</Typography.Text>
            <dl><div><dt>运行时</dt><dd>{item.runtime}</dd></div><div><dt>端点</dt><dd>{item.endpoint_url}</dd></div></dl>
            {operationButtons(item, true)}
          </div>
        )} />
      </QueryState>
      <Drawer
        title={editingDeployment ? `编辑 ${editingDeployment.name}` : '新建模型部署'}
        width={620}
        open={drawerOpen}
        onClose={closeDrawer}
        extra={<Steps size="small" current={preview ? 1 : 0} items={[{ title: '配置' }, { title: '确认' }]} />}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={defaultValues}
          onFinish={(values) => preview ? saveMutation.mutate(values) : previewMutation.mutate(values)}
          onValuesChange={(changed) => {
            setPreview(null)
            if ('runtime' in changed) {
              form.setFieldValue('image', changed.runtime === 'vllm'
                ? 'vllm/vllm-openai:v0.27.1'
                : 'sglang-inkling:specforge')
            }
          }}
        >
          {!preview ? <>
            <Form.Item name="model_id" label="模型" rules={[{ required: true }]}>
              <Select loading={models.isLoading} options={models.data?.map((item) => ({ value: item.id, label: item.name }))} onChange={selectModel} />
            </Form.Item>
            <Form.Item name="name" label="部署名称" rules={[{ required: true }]}><Input /></Form.Item>
            <Form.Item name="model_path" label="模型路径" rules={[{ required: true }]}><Input readOnly /></Form.Item>
            <Form.Item name="api_model_name" label="实例模型名称" rules={[{ required: true }]} tooltip="传给上游运行时的唯一模型名称"><Input /></Form.Item>
            <Form.Item name="route_alias" label="共享网关别名（可选）" tooltip="多个部署填写相同别名时，网关会在健康实例间轮询"><Input placeholder="例如 qwen-production" /></Form.Item>
            <Form.Item name="runtime" label="推理运行时"><Segmented block options={[{ label: 'vLLM', value: 'vllm' }, { label: 'SGLang', value: 'sglang' }]} /></Form.Item>
            <Form.Item name="image" label="ARM64 镜像" rules={[{ required: true }]}>
              <Select options={(runtime === 'vllm'
                ? ['vllm/vllm-openai:v0.27.1']
                : ['sglang-inkling:specforge', 'lmsysorg/sglang:dev-cu13-inkling-dspark'])
                .map((value) => ({ value, label: value }))} />
            </Form.Item>
            <div className="form-grid">
              <Form.Item name="port" label="主机端口"><InputNumber min={1024} max={65535} /></Form.Item>
              <Form.Item name="context_length" label="上下文长度"><InputNumber min={1024} step={1024} /></Form.Item>
              <Form.Item name="max_concurrency" label="最大并发"><InputNumber min={1} max={1024} /></Form.Item>
              <Form.Item name="max_batched_tokens" label="批处理 Token 上限" tooltip={runtime === 'vllm' ? 'vLLM 每轮调度的最大 Token 数' : '当前 SGLang 适配器不设置此参数'}>
                <InputNumber min={1024} step={1024} disabled={runtime !== 'vllm'} />
              </Form.Item>
              <Form.Item name="quantization" label="量化加载方式"><Select allowClear placeholder="自动检测" options={['auto', 'awq', 'gptq', 'fp8', 'bitsandbytes', 'marlin'].map((value) => ({ value, label: value }))} /></Form.Item>
              <Form.Item name="trust_remote_code" label="信任远程代码" valuePropName="checked"><Switch /></Form.Item>
            </div>
            <Form.Item name="memory_fraction" label="统一内存比例"><Slider min={0.05} max={0.98} step={0.01} tooltip={{ formatter: (value) => `${Math.round((value ?? 0) * 100)}%` }} /></Form.Item>
          </> : <div className="deployment-preview">
            {editingDeployment && <Alert type="warning" showIcon message="更新会替换当前容器" description="旧容器会先停止并保留到新实例通过健康检查；更新失败时自动恢复旧容器。" />}
            <Alert type={compatibility?.compatible ? 'success' : 'warning'} showIcon message={compatibility?.compatible ? '模型结构与所选运行时兼容' : '兼容性检查需要确认'} description={compatibility?.compatible ? `检测到 ${compatibility.architectures.join(', ') || 'Transformers'} 架构` : compatibility?.reasons.join('；')} />
            <Descriptions bordered size="small" column={1} items={[
              { key: 'runtime', label: '运行时', children: String(preview.runtime) },
              { key: 'image', label: 'ARM64 镜像', children: String(preview.image) },
              { key: 'container', label: '容器', children: String(preview.container_name) },
              { key: 'port', label: '主机端口', children: String(preview.port) },
              { key: 'disk', label: '模型磁盘', children: formatBytes(Number(preview.estimated_disk_bytes ?? 0)) },
              { key: 'memory', label: '估算统一内存', children: formatBytes(Number(preview.estimated_memory_bytes ?? 0)) },
              { key: 'route', label: '网关模型名', children: String(preview.route_alias || form.getFieldValue('api_model_name')) },
            ]} />
            <div><Typography.Title level={5}>执行动作</Typography.Title><List size="small" dataSource={operations} renderItem={(item) => <List.Item>{item}</List.Item>} /></div>
            <div><Typography.Title level={5}>调用示例</Typography.Title><pre><code>{String(preview.api_example)}</code></pre></div>
            <Alert type="info" showIcon message="失败回滚" description={String(preview.rollback)} />
          </div>}
          <Button type="primary" htmlType="submit" block loading={previewMutation.isPending || saveMutation.isPending}>
            {preview ? (editingDeployment ? '确认并创建更新任务' : '确认并创建任务') : '检查部署配置'}
          </Button>
        </Form>
      </Drawer>
      <Drawer title={`${logsFor?.name ?? ''} 日志`} width={760} open={Boolean(logsFor)} onClose={() => setLogsFor(null)}>
        <QueryState loading={logs.isLoading} error={logs.error}><LogViewer value={logs.data?.logs ?? ''} filename={`${logsFor?.name}.log`} /></QueryState>
      </Drawer>
    </div>
  )
}
