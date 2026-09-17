import { CheckOutlined, CloseOutlined, LockOutlined } from '@ant-design/icons'
import { Alert, Button, Flex, Popconfirm, Space, Typography } from 'antd'

import type { OperationPlan } from '../api/types'
import { OpsExecutionOutput } from './OpsExecutionOutput'
import { StatusBadge } from './StatusBadge'

const riskLabels: Record<string, string> = { low: '低', medium: '中', high: '高', critical: '严重' }


export function ApprovalPanel({ plan, onApprove, onReject, busy }: { plan: OperationPlan; onApprove: () => void; onReject: () => void; busy?: boolean }) {
  const pending = plan.status === 'pending'
  const executableSteps = plan.steps.filter((step) => step.executable)
  return (
    <article className="approval-panel" aria-label={`操作审批：${plan.summary}`}>
      <Flex justify="space-between" align="flex-start" gap={12} wrap>
        <div>
          <Space wrap><StatusBadge status={plan.status} /><Typography.Text type="secondary">风险：{riskLabels[plan.risk] ?? plan.risk}</Typography.Text></Space>
          <h3>{plan.summary}</h3>
        </div>
        {pending && (
          <Space wrap>
            <Button aria-label="拒绝" icon={<CloseOutlined />} onClick={onReject} loading={busy}>拒绝</Button>
            <Popconfirm
              title={`批准“${plan.summary}”`}
              description={`将在 DGX 上执行下方列出的 ${executableSteps.length} 个确切命令。`}
              okText="确认执行"
              cancelText="取消"
              onConfirm={onApprove}
            >
              <Button aria-label="批准执行" type="primary" icon={<CheckOutlined />} loading={busy} disabled={!executableSteps.length}>批准执行</Button>
            </Popconfirm>
          </Space>
        )}
      </Flex>
      <Typography.Paragraph>{plan.diagnosis}</Typography.Paragraph>
      <div className="approval-steps">
        {plan.steps.map((step, index) => (
          <section className="approval-step" key={step.id ?? index}>
            <div className="step-index">{index + 1}</div>
            <div className="approval-step-content">
              <Space wrap>
                <strong>{step.operation}</strong>
                {!step.executable && <Typography.Text type="secondary"><LockOutlined /> 仅说明</Typography.Text>}
              </Space>
              {step.command && <pre className="approval-command"><code>{step.command}</code></pre>}
              {step.command && <div className="approval-command-meta"><span>目录：<code>{step.cwd ?? '/'}</code></span><span>超时：{step.timeout ?? 60} 秒</span></div>}
              <small>{step.reason}</small>
              {step.impact && <small>影响：{step.impact}</small>}
              {step.rollback && <small>回滚：{step.rollback}</small>}
            </div>
          </section>
        ))}
      </div>
      {pending && !executableSteps.length && <Alert type="info" showIcon message="此方案没有可执行步骤，只可保留为诊断说明。" />}
      <OpsExecutionOutput result={plan.result} />
    </article>
  )
}
