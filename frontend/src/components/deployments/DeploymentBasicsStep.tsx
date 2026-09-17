import { Form, Input, InputNumber, Segmented, Select, Typography } from 'antd'

import type { ModelAsset, Provider, RuntimeName } from '../../api/types'


interface DeploymentBasicsStepProps {
  models: ModelAsset[]
  providers: Provider[]
  runtime: RuntimeName
  loading: boolean
  providersLoading: boolean
  onModelChange: (modelId: string) => void
  onProviderChange: (providerId: string) => void
}

const runtimeImages = {
  vllm: ['vllm/vllm-openai:v0.27.1'],
  sglang: [
    'dgx-local/sglang-qwen38-dflash2:61fa64a',
    'lmsysorg/sglang:qwen38-27b',
    'dgx-local/sglang-ssd-stream:5aeffa3-sm121-r3',
    'sglang-inkling:specforge',
    'lmsysorg/sglang:dev-cu13-inkling-dspark',
  ],
  llama_cpp: ['nvidia/cuda:12.9.0-devel-ubuntu24.04'],
} as const


export function DeploymentBasicsStep({
  models,
  providers,
  runtime,
  loading,
  providersLoading,
  onModelChange,
  onProviderChange,
}: DeploymentBasicsStepProps) {
  const availableModels = models.filter((model) => model.status === 'available')
  const enabledProviders = providers.filter((provider) => provider.enabled)

  return (
    <section className="deployment-step" aria-labelledby="deployment-basics-heading">
      <div className="deployment-step-heading">
        <Typography.Title level={4} id="deployment-basics-heading">基础模型</Typography.Title>
        <Typography.Text type="secondary">选择本地模型、ARM64 运行时和可选的 AI 推荐服务。</Typography.Text>
      </div>
      <Form.Item name="chat_template" hidden><Input /></Form.Item>
      <Form.Item name="model_id" label="模型" rules={[{ required: true, message: '请选择模型' }]}>
        <Select
          showSearch
          optionFilterProp="label"
          loading={loading}
          options={availableModels.map((item) => ({ value: item.id, label: item.name }))}
          onChange={onModelChange}
        />
      </Form.Item>
      <div className="form-grid">
        <Form.Item name="name" label="部署名称" rules={[{ required: true, message: '请输入部署名称' }]}>
          <Input />
        </Form.Item>
        <Form.Item
          name="api_model_name"
          label="实例模型名称"
          rules={[{ required: true, message: '请输入实例模型名称' }]}
          tooltip="传给上游运行时的唯一模型名称"
        >
          <Input />
        </Form.Item>
      </div>
      <Form.Item name="model_path" label="模型路径" rules={[{ required: true, message: '模型路径缺失' }]}>
        <Input readOnly />
      </Form.Item>
      <Form.Item
        name="route_alias"
        label="共享网关别名（可选）"
        tooltip="多个部署填写相同别名时，网关会在健康实例间轮询"
      >
        <Input placeholder="例如 qwen-production" />
      </Form.Item>
      <Form.Item name="runtime" label="推理运行时" rules={[{ required: true }]}>
        <Segmented
          block
          options={[
            { label: 'vLLM', value: 'vllm' },
            { label: 'SGLang', value: 'sglang' },
            { label: 'llama.cpp', value: 'llama_cpp' },
          ]}
        />
      </Form.Item>
      <Form.Item name="image" label="ARM64 镜像" rules={[{ required: true, message: '请选择运行时镜像' }]}>
        <Select options={runtimeImages[runtime].map((value) => ({ value, label: value }))} />
      </Form.Item>
      <div className="form-grid">
        <Form.Item name="port" label="主机端口" extra="留空则从 8000 起自动分配">
          <InputNumber min={1024} max={65535} />
        </Form.Item>
        <Form.Item name="provider_id" label="AI 推荐服务">
          <Select
            loading={providersLoading}
            options={[
              { value: '', label: '不使用 AI 补充' },
              ...enabledProviders.map((provider) => ({ value: provider.id, label: provider.name })),
            ]}
            onChange={onProviderChange}
          />
        </Form.Item>
      </div>
    </section>
  )
}
