import { ReloadOutlined, UndoOutlined } from '@ant-design/icons'
import {
  Alert,
  AutoComplete,
  Button,
  Collapse,
  Form,
  InputNumber,
  Select,
  Slider,
  Space,
  Switch,
  Tag,
  Typography,
} from 'antd'
import type { FormItemProps } from 'antd'
import type { ReactNode } from 'react'

import type { DeploymentRecommendation, RecommendedValue, RuntimeName } from '../../api/types'
import { RecommendationSourceTag } from './RecommendationSourceTag'


interface RecommendationStepProps {
  recommendation?: DeploymentRecommendation
  editedFields: ReadonlySet<string>
  loading: boolean
  refreshing: boolean
  error?: Error | null
  runtime: RuntimeName
  editing?: boolean
  onReapplyAll: () => void
  onRetryAI: () => void
}

const confidenceLabels = { high: '高置信度', medium: '中置信度', low: '低置信度' } as const


function RecommendedField({
  name,
  path,
  label,
  recommended,
  editedFields,
  rules,
  children,
}: {
  name: string | string[]
  path: string
  label: string
  recommended?: RecommendedValue
  editedFields: ReadonlySet<string>
  rules?: FormItemProps['rules']
  children: ReactNode
}) {
  return (
    <div className="recommendation-row">
      <Form.Item name={name} label={label} className="recommendation-field" rules={rules}>
        {children}
      </Form.Item>
      <div className="recommendation-field-meta">
        <span className="recommendation-meta">
          {recommended && <RecommendationSourceTag source={recommended.source} />}
          {recommended && <Tag>{confidenceLabels[recommended.confidence]}</Tag>}
          {editedFields.has(path) && <Tag color="gold">已手动修改</Tag>}
        </span>
        {recommended && (
          <Typography.Text className="recommendation-reason" type="secondary" id={`${path}-reason`}>
            {recommended.reason}
          </Typography.Text>
        )}
      </div>
    </div>
  )
}


export function RecommendationStep({
  recommendation,
  editedFields,
  loading,
  refreshing,
  error,
  runtime,
  editing = false,
  onReapplyAll,
  onRetryAI,
}: RecommendationStepProps) {
  const field = (name: string) => recommendation?.fields[name]
  const generation = (name: string) => recommendation?.generation_defaults[name]
  const currentQuantization = Form.useWatch('quantization', { preserve: true }) as string | undefined
  const quantizationMethods = recommendation
    ? recommendation.runtime_capabilities.quantization_methods ?? []
    : ['auto']
  const quantizationOptions: Array<{ value: string; label: string; disabled?: boolean }> = [...new Set(quantizationMethods)].map((value) => ({
    value,
    label: value === 'modelopt_fp4' ? 'NVFP4 / ModelOpt FP4' : value,
  }))
  if (
    editing
    && currentQuantization
    && currentQuantization !== 'auto'
    && !quantizationMethods.includes(currentQuantization)
  ) {
    quantizationOptions.unshift({
      value: currentQuantization,
      label: `${currentQuantization === 'modelopt_fp4' ? 'NVFP4 / ModelOpt FP4' : currentQuantization}（已保存）`,
      disabled: true,
    })
  }

  const generationItems = [
    {
      key: 'generation-defaults',
      label: '默认生成参数',
      children: (
        <div className="deployment-step">
          <div className="form-grid">
            <RecommendedField name={['generation_defaults', 'temperature']} path="generation_defaults.temperature" label="Temperature" recommended={generation('temperature')} editedFields={editedFields} rules={[{ type: 'number', min: 0, max: 2 }]}>
              <InputNumber min={0} max={2} step={0.1} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'top_p']} path="generation_defaults.top_p" label="Top P" recommended={generation('top_p')} editedFields={editedFields} rules={[{ type: 'number', min: 0.01, max: 1 }]}>
              <InputNumber min={0.01} max={1} step={0.05} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'top_k']} path="generation_defaults.top_k" label="Top K" recommended={generation('top_k')} editedFields={editedFields} rules={[{ type: 'number', min: 0, max: 1_000_000 }]}>
              <InputNumber min={0} max={1_000_000} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'min_p']} path="generation_defaults.min_p" label="Min P" recommended={generation('min_p')} editedFields={editedFields} rules={[{ type: 'number', min: 0, max: 1 }]}>
              <InputNumber min={0} max={1} step={0.01} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'repetition_penalty']} path="generation_defaults.repetition_penalty" label="重复惩罚" recommended={generation('repetition_penalty')} editedFields={editedFields} rules={[{ type: 'number', min: 0.01, max: 2 }]}>
              <InputNumber min={0.01} max={2} step={0.05} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'presence_penalty']} path="generation_defaults.presence_penalty" label="Presence Penalty" recommended={generation('presence_penalty')} editedFields={editedFields} rules={[{ type: 'number', min: -2, max: 2 }]}>
              <InputNumber min={-2} max={2} step={0.1} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'frequency_penalty']} path="generation_defaults.frequency_penalty" label="Frequency Penalty" recommended={generation('frequency_penalty')} editedFields={editedFields} rules={[{ type: 'number', min: -2, max: 2 }]}>
              <InputNumber min={-2} max={2} step={0.1} />
            </RecommendedField>
            <RecommendedField name={['generation_defaults', 'max_tokens']} path="generation_defaults.max_tokens" label="默认最大生成 Token" recommended={generation('max_tokens')} editedFields={editedFields} rules={[{ type: 'number', min: 1, max: 1_048_576 }]}>
              <InputNumber min={1} max={1_048_576} />
            </RecommendedField>
          </div>
          <RecommendedField name={['generation_defaults', 'stop']} path="generation_defaults.stop" label="停止序列" recommended={generation('stop')} editedFields={editedFields}>
            <Select mode="tags" tokenSeparators={[',']} maxCount={16} placeholder="输入后回车，可设置多个" />
          </RecommendedField>
          <div className="field-grid">
            <Form.Item
              name={['chat_template_kwargs', 'enable_thinking']}
              label="默认思考模式"
              tooltip="随启动参数下发，单次请求的 chat_template_kwargs 仍可覆盖。留空则跟随模型模板自身的默认值。"
            >
              <Select
                allowClear
                placeholder="跟随模型默认"
                options={[
                  { value: true, label: '开启' },
                  { value: false, label: '关闭' },
                ]}
              />
            </Form.Item>
            <Form.Item
              name={['chat_template_kwargs', 'reasoning_effort']}
              label="默认思考强度"
              tooltip="仅对模板实现了 reasoning_effort 的模型有效（如 Qwen3.8 支持 low/medium/xhigh）。可直接输入模型自定义的取值。"
            >
              <AutoComplete
                allowClear
                placeholder="跟随模型默认"
                options={[
                  { value: 'low' },
                  { value: 'medium' },
                  { value: 'high' },
                  { value: 'xhigh' },
                ]}
              />
            </Form.Item>
          </div>
        </div>
      ),
    },
  ]

  return (
    <section className="deployment-step" aria-labelledby="recommendation-heading">
      <div className="deployment-step-heading deployment-step-heading-actions">
        <div>
          <Typography.Title level={4} id="recommendation-heading">推荐配置</Typography.Title>
          <Typography.Text type="secondary">模型卡、运行时能力与实时统一内存共同决定初始值。</Typography.Text>
        </div>
        <Space wrap>
          <Button aria-label="重新分析" icon={<ReloadOutlined />} loading={refreshing} onClick={onRetryAI}>重新分析</Button>
          <Button aria-label="重新应用全部建议" icon={<UndoOutlined />} disabled={!recommendation} onClick={onReapplyAll}>重新应用全部建议</Button>
        </Space>
      </div>
      {loading && <Alert type="info" showIcon message="正在分析模型卡与设备资源" />}
      {error && (
        <Alert
          type="warning"
          showIcon
          message="推荐分析失败"
          description={error.message}
          action={<Button size="small" onClick={onRetryAI}>重试</Button>}
        />
      )}
      {recommendation?.status === 'partial' && (
        <Alert
          type="warning"
          showIcon
          message="AI 补充不可用，已使用确定性建议"
          description={recommendation.warnings.join('；') || '部分字段保留运行时默认值，可继续手动部署。'}
        />
      )}
      {recommendation?.status === 'unavailable' && (
        <Alert
          type="error"
          showIcon
          message="无法验证自动建议"
          description={recommendation.warnings.join('；') || '请手动检查模型、镜像和运行时配置。'}
        />
      )}
      <div className="form-grid">
        <RecommendedField name="context_length" path="context_length" label="上下文长度" recommended={field('context_length')} editedFields={editedFields} rules={[{ required: true, type: 'number', min: 1024, max: 1_048_576 }]}>
          <InputNumber min={1024} max={1_048_576} step={1024} />
        </RecommendedField>
        <RecommendedField name="max_total_tokens" path="max_total_tokens" label="运行时总 Token 槽" editedFields={editedFields} rules={[{ type: 'number', min: 1024, max: 1_048_576 }]}>
          <InputNumber min={1024} max={1_048_576} step={1024} disabled={runtime !== 'sglang'} />
        </RecommendedField>
        <RecommendedField name="max_concurrency" path="max_concurrency" label="最大并发" recommended={field('max_concurrency')} editedFields={editedFields} rules={[{ required: true, type: 'number', min: 1, max: 1024 }]}>
          <InputNumber min={1} max={1024} />
        </RecommendedField>
        <RecommendedField name="max_batched_tokens" path="max_batched_tokens" label="批处理 Token 上限" recommended={field('max_batched_tokens')} editedFields={editedFields} rules={[{ type: 'number', min: 1024, max: 1_048_576 }]}>
          <InputNumber min={1024} max={1_048_576} step={1024} disabled={runtime !== 'vllm'} />
        </RecommendedField>
        <RecommendedField name="quantization" path="quantization" label="量化加载方式" recommended={field('quantization')} editedFields={editedFields}>
          <Select options={quantizationOptions} />
        </RecommendedField>
      </div>
      <RecommendedField name="memory_fraction" path="memory_fraction" label="统一内存比例" recommended={field('memory_fraction')} editedFields={editedFields} rules={[{ required: true, type: 'number', min: 0.05, max: 0.98 }]}>
        <Slider min={0.05} max={0.98} step={0.01} tooltip={{ formatter: (value) => `${Math.round((value ?? 0) * 100)}%` }} />
      </RecommendedField>
      <Form.Item name="trust_remote_code" label="信任远程代码" valuePropName="checked">
        <Switch />
      </Form.Item>
      <Collapse items={generationItems} defaultActiveKey={['generation-defaults']} />
    </section>
  )
}
