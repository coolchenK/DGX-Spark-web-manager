import { Alert, Descriptions, List, Tag, Typography } from 'antd'

import type { DraftCandidate, ResourceEstimate, RuntimeCapabilities } from '../../api/types'
import { formatBytes } from '../../utils/format'
import type { DeploymentFormValues, SpeculativeSettings } from '../../utils/deployments'


export interface DeploymentPreview {
  runtime?: string
  image?: string
  container_name?: string
  port?: number
  route_alias?: string
  estimated_disk_bytes?: number
  estimated_memory_bytes?: number
  compatibility?: {
    compatible: boolean
    architectures: string[]
    reasons: string[]
  }
  spec?: Partial<DeploymentFormValues>
  resource_estimate?: Partial<ResourceEstimate>
  runtime_capabilities?: Partial<RuntimeCapabilities>
  draft_candidate?: DraftCandidate | null
  speculative?: SpeculativeSettings | null
  warnings?: string[]
  generation_defaults?: Record<string, unknown>
  mounts?: Record<string, unknown>
  command?: string[]
  operations?: string[]
  api_example?: string
  rollback?: string
}


function jsonPreview(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2)
}


export function DeploymentPreviewStep({
  preview,
  editing,
  fallbackRoute,
}: {
  preview: DeploymentPreview
  editing: boolean
  fallbackRoute: string
}) {
  const compatibility = preview.compatibility
  const resource = preview.resource_estimate
  const spec = preview.spec
  const decisionColor = resource?.decision === 'blocked'
    ? 'error'
    : resource?.decision === 'warning' ? 'warning' : 'success'

  return (
    <section className="deployment-step deployment-preview" aria-labelledby="deployment-preview-heading">
      <div className="deployment-step-heading">
        <Typography.Title level={4} id="deployment-preview-heading">部署预览</Typography.Title>
        <Typography.Text type="secondary">确认运行时命令、只读挂载、资源预算与网关默认值。</Typography.Text>
      </div>
      {editing && (
        <Alert
          type="warning"
          showIcon
          message="更新会替换当前容器"
          description="旧容器会保留到新实例通过健康检查；更新失败时自动恢复旧容器。"
        />
      )}
      <Alert
        type={compatibility?.compatible ? 'success' : 'warning'}
        showIcon
        message={compatibility?.compatible ? '模型结构与所选运行时兼容' : '兼容性检查需要确认'}
        description={compatibility?.compatible
          ? `检测到 ${compatibility.architectures.join(', ') || 'Transformers'} 架构`
          : compatibility?.reasons.join('；')}
      />
      <Descriptions bordered size="small" column={1} items={[
        { key: 'runtime', label: '运行时', children: preview.runtime },
        { key: 'image', label: 'ARM64 镜像', children: <span className="break-anywhere">{preview.image}</span> },
        { key: 'container', label: '容器', children: preview.container_name },
        { key: 'port', label: '主机端口', children: preview.port },
        { key: 'disk', label: '模型磁盘', children: formatBytes(Number(preview.estimated_disk_bytes ?? 0)) },
        { key: 'memory', label: '估算统一内存', children: formatBytes(Number(resource?.required_bytes ?? preview.estimated_memory_bytes ?? 0)) },
        { key: 'route', label: '网关模型名', children: preview.route_alias || fallbackRoute },
        { key: 'decision', label: '资源结论', children: <Tag color={decisionColor}>{resource?.decision ?? 'ok'}</Tag> },
      ]} />
      <div className="preview-section">
        <Typography.Title level={5}>解析后的部署参数</Typography.Title>
        <Descriptions bordered size="small" column={1} items={[
          { key: 'context', label: '上下文长度', children: spec?.context_length ?? '未设置' },
          { key: 'fraction', label: '统一内存比例', children: spec?.memory_fraction ?? '未设置' },
          { key: 'concurrency', label: '最大并发', children: spec?.max_concurrency ?? '未设置' },
          { key: 'batched', label: '批处理 Token 上限', children: spec?.max_batched_tokens ?? '不适用' },
          { key: 'quantization', label: '量化加载方式', children: spec?.quantization ?? 'auto' },
          {
            key: 'chat-template',
            label: '对话模板',
            children: spec?.chat_template === 'qwen-fixed-v22.4'
              ? <Tag color="green">Qwen Fixed v22.4</Tag>
              : '模型内置',
          },
        ]} />
      </div>
      <div className="preview-section">
        <Typography.Title level={5}>Draft Model 配置</Typography.Title>
        {preview.speculative ? (
          <Descriptions bordered size="small" column={1} items={[
            {
              key: 'draft',
              label: '候选模型',
              children: preview.draft_candidate?.name
                ?? (preview.speculative.method === 'mtp' && !preview.speculative.draft_model_id
                  ? '内置 MTP Head'
                  : preview.speculative.draft_model_id),
            },
            { key: 'method', label: '推测方法', children: preview.speculative.method },
            { key: 'tuning', label: '运行时参数', children: <pre><code>{jsonPreview(preview.speculative)}</code></pre> },
          ]} />
        ) : <Typography.Text type="secondary">未附带 Draft Model</Typography.Text>}
      </div>
      <div className="preview-section">
        <Typography.Title level={5}>资源明细</Typography.Title>
        <Descriptions bordered size="small" column={1} items={[
          { key: 'required', label: '总需求', children: formatBytes(Number(resource?.required_bytes ?? 0)) },
          { key: 'weight', label: '基础权重', children: formatBytes(Number(resource?.weight_bytes ?? 0)) },
          { key: 'draft-weight', label: 'Draft 权重', children: formatBytes(Number(resource?.draft_weight_bytes ?? 0)) },
          { key: 'kv', label: 'KV Cache', children: formatBytes(Number(resource?.kv_cache_bytes ?? 0)) },
          { key: 'overhead', label: '运行时开销', children: formatBytes(Number(resource?.runtime_overhead_bytes ?? 0)) },
          { key: 'resource-decision', label: '结论', children: <Tag color={decisionColor}>{resource?.decision ?? 'ok'}</Tag> },
        ]} />
        {(resource?.reasons ?? []).map((reason) => (
          <Alert key={reason} type={resource?.decision === 'blocked' ? 'error' : 'info'} showIcon message={reason} />
        ))}
      </div>
      {(preview.warnings ?? []).map((warning) => (
        <Alert key={`preview-${warning}`} type="warning" showIcon message={warning} />
      ))}
      {(preview.runtime_capabilities?.warnings ?? []).map((warning) => (
        <Alert key={`runtime-${warning}`} type="warning" showIcon message={warning} />
      ))}
      <div className="preview-section">
        <Typography.Title level={5}>容器命令</Typography.Title>
        <pre><code>{(preview.command ?? []).join(' ')}</code></pre>
      </div>
      <div className="preview-section preview-split">
        <div>
          <Typography.Title level={5}>模型挂载</Typography.Title>
          <pre><code>{jsonPreview(preview.mounts)}</code></pre>
        </div>
        <div>
          <Typography.Title level={5}>网关默认值</Typography.Title>
          <pre><code>{jsonPreview(preview.generation_defaults)}</code></pre>
        </div>
      </div>
      <div className="preview-section">
        <Typography.Title level={5}>执行动作</Typography.Title>
        <List size="small" dataSource={preview.operations ?? []} renderItem={(item) => <List.Item>{item}</List.Item>} />
      </div>
      <div className="preview-section">
        <Typography.Title level={5}>调用示例</Typography.Title>
        <pre><code>{preview.api_example}</code></pre>
      </div>
      <Alert type="info" showIcon message="失败回滚" description={preview.rollback} />
    </section>
  )
}
