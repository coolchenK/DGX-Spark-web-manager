import { CheckOutlined, CloseOutlined, LockOutlined } from '@ant-design/icons'
import { Alert, Button, Collapse, Flex, List, Popconfirm, Space, Typography } from 'antd'

import type { OperationPlan } from '../api/types'
import { StatusBadge } from './StatusBadge'


export function ApprovalPanel({ plan, onApprove, onReject, busy }: { plan: OperationPlan; onApprove: () => void; onReject: () => void; busy?: boolean }) {
  const pending = plan.status === 'pending'
  return (
    <article className="approval-panel">
      <Flex justify="space-between" align="flex-start" gap={12} wrap>
        <div><Space><StatusBadge status={plan.status} /><Typography.Text type="secondary">风险：{plan.risk}</Typography.Text></Space><h3>{plan.summary}</h3></div>
        {pending && <Space><Button icon={<CloseOutlined />} onClick={onReject} loading={busy}>拒绝</Button><Popconfirm title={`批准“${plan.summary}”`} description="仅执行标记为可执行的白名单操作，所有步骤会记录审计。" onConfirm={onApprove}><Button type="primary" icon={<CheckOutlined />} loading={busy}>批准执行</Button></Popconfirm></Space>}
      </Flex>
      <Typography.Paragraph>{plan.diagnosis}</Typography.Paragraph>
      <Collapse ghost items={[{ key: 'steps', label: `操作计划 (${plan.steps.length})`, children: <List dataSource={plan.steps} renderItem={(step, index) => <List.Item><Flex gap={10} align="flex-start"><span className="step-index">{index + 1}</span><div className="primary-cell"><Space><strong>{step.operation}</strong>{!step.executable && <Typography.Text type="secondary"><LockOutlined /> 仅说明</Typography.Text>}</Space><small>{step.reason}</small>{step.impact && <small>影响：{step.impact}</small>}{step.rollback && <small>回滚：{step.rollback}</small>}</div></Flex></List.Item>} /> }]} />
      {pending && !plan.steps.some((step) => step.executable) && <Alert type="info" showIcon message="此方案没有可执行步骤，只可保留为诊断说明。" />}
    </article>
  )
}
