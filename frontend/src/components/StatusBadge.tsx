import { CheckCircleFilled, ClockCircleFilled, CloseCircleFilled, MinusCircleFilled, SyncOutlined, WarningFilled } from '@ant-design/icons'
import { Tag } from 'antd'


const labels: Record<string, string> = {
  healthy: '健康', running: '运行中', succeeded: '已完成', available: '可用',
  active: '进行中', answered: '已回复', queued: '排队中', pending: '待审批',
  processing: '处理中', executing: '执行中', approval_required: '待审批', needs_input: '待补充',
  failed: '失败', unhealthy: '异常', cancelled: '已取消', rejected: '已拒绝',
  paused: '已暂停', exited: '已停止', stopped: '已停止', approved: '已批准', unknown: '未知',
}


export function StatusBadge({ status }: { status: string }) {
  const normalized = status?.toLowerCase() || 'unknown'
  if (['healthy', 'running', 'succeeded', 'available', 'completed', 'active', 'answered'].includes(normalized)) {
    return <Tag color="success" icon={<CheckCircleFilled />}>{labels[normalized] ?? status}</Tag>
  }
  if (['queued', 'processing', 'executing'].includes(normalized)) {
    return <Tag color="processing" icon={<SyncOutlined spin={normalized !== 'queued'} />}>{labels[normalized]}</Tag>
  }
  if (['failed', 'unhealthy', 'cancelled', 'rejected'].includes(normalized)) {
    return <Tag color="error" icon={<CloseCircleFilled />}>{labels[normalized] ?? status}</Tag>
  }
  if (['paused', 'exited', 'stopped', 'pending', 'approval_required', 'needs_input'].includes(normalized)) {
    return <Tag color="warning" icon={['pending', 'approval_required'].includes(normalized) ? <ClockCircleFilled /> : <WarningFilled />}>{labels[normalized]}</Tag>
  }
  return <Tag icon={<MinusCircleFilled />}>{labels[normalized] ?? status}</Tag>
}
