import { Alert, Descriptions, List, Tag, Typography } from 'antd'

import { formatBytes } from '../../utils/format'


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
  resource_estimate?: {
    required_bytes?: number
    weight_bytes?: number
    draft_weight_bytes?: number
    kv_cache_bytes?: number
    runtime_overhead_bytes?: number
    decision?: string
    reasons?: string[]
  }
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
        { key: 'decision', label: '资源结论', children: <Tag color={resource?.decision === 'warning' ? 'warning' : 'success'}>{resource?.decision ?? 'ok'}</Tag> },
      ]} />
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
