import { RobotOutlined, ToolOutlined, UserOutlined } from '@ant-design/icons'
import { Collapse, Empty, Space, Typography } from 'antd'

import type { OpsMessage, OpsSession, OpsToolRun, OperationPlan } from '../api/types'
import { ApprovalPanel } from './ApprovalPanel'
import { StatusBadge } from './StatusBadge'


type TimelineEntry =
  | { id: string; kind: 'message'; at: string; value: OpsMessage }
  | { id: string; kind: 'tool'; at: string; value: OpsToolRun }
  | { id: string; kind: 'plan'; at: string; value: OperationPlan }

function json(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function timeline(session: OpsSession): TimelineEntry[] {
  const linkedPlans = new Set(session.messages.map((item) => item.operation_plan_id).filter(Boolean))
  return [
    ...session.messages.map((value): TimelineEntry => ({ id: value.id, kind: 'message', at: value.created_at, value })),
    ...session.tool_runs.map((value): TimelineEntry => ({ id: value.id, kind: 'tool', at: value.created_at, value })),
    ...session.plans.filter((value) => !linkedPlans.has(value.id)).map((value): TimelineEntry => ({ id: value.id, kind: 'plan', at: value.created_at, value })),
  ].sort((left, right) => left.at.localeCompare(right.at) || left.id.localeCompare(right.id))
}

function ToolRun({ run }: { run: OpsToolRun }) {
  return (
    <div className="ops-tool-run">
      <Collapse
        ghost
        items={[{
          key: run.id,
          label: (
            <div className="ops-tool-label">
              <Space size={8}><ToolOutlined /><strong>{run.tool_name}</strong></Space>
              <Space size={8}><Typography.Text type="secondary">自动执行 · 只读</Typography.Text><StatusBadge status={run.status} /></Space>
            </div>
          ),
          children: (
            <div className="ops-tool-detail">
              <div><Typography.Text type="secondary">参数</Typography.Text><pre><code>{json(run.arguments)}</code></pre></div>
              <div><Typography.Text type="secondary">结果</Typography.Text><pre><code>{json(run.result)}</code></pre></div>
              {run.error && <Typography.Text type="danger">{run.error}</Typography.Text>}
            </div>
          ),
        }]}
      />
    </div>
  )
}

export function OpsConversation({
  session,
  busy,
  onApprove,
  onReject,
}: {
  session: OpsSession
  busy?: boolean
  onApprove: (plan: OperationPlan) => void
  onReject: (plan: OperationPlan) => void
}) {
  const plans = new Map(session.plans.map((plan) => [plan.id, plan]))
  const entries = timeline(session)
  if (!entries.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="会话尚无记录" />

  return (
    <div className="ops-conversation" aria-live="polite">
      {entries.map((entry) => {
        if (entry.kind === 'tool') return <ToolRun key={entry.id} run={entry.value} />
        if (entry.kind === 'plan') {
          return <ApprovalPanel key={entry.id} plan={entry.value} busy={busy} onApprove={() => onApprove(entry.value)} onReject={() => onReject(entry.value)} />
        }
        const message = entry.value
        const linkedPlan = message.operation_plan_id ? plans.get(message.operation_plan_id) : undefined
        return (
          <div className={`ops-message ops-message-${message.role}`} key={message.id}>
            <div className="ops-message-role">
              {message.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
              <span>{message.role === 'user' ? '你' : '运维助手'}</span>
            </div>
            <Typography.Paragraph>{message.content}</Typography.Paragraph>
            {linkedPlan && <ApprovalPanel plan={linkedPlan} busy={busy} onApprove={() => onApprove(linkedPlan)} onReject={() => onReject(linkedPlan)} />}
          </div>
        )
      })}
    </div>
  )
}
