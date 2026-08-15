import { Alert, Button, Flex, Progress, Space, Typography } from 'antd'
import { PauseOutlined, PlayCircleOutlined, StopOutlined } from '@ant-design/icons'

import type { TaskRecord } from '../api/types'
import { formatBytes, formatDate } from '../utils/format'
import { StatusBadge } from './StatusBadge'


export function TaskProgress({ task, onPause, onResume, onCancel, busy }: { task: TaskRecord; onPause?: () => void; onResume?: () => void; onCancel?: () => void; busy?: boolean }) {
  const canPause = ['queued', 'running'].includes(task.status)
  const canResume = ['paused', 'failed', 'cancelled'].includes(task.status)
  const canCancel = ['queued', 'running', 'paused'].includes(task.status)
  return (
    <article className="task-progress">
      <Flex justify="space-between" align="flex-start" gap={12} wrap>
        <div className="primary-cell"><Space><StatusBadge status={task.status} /><strong>{task.title}</strong></Space><small>{task.type} · {formatDate(task.updated_at)}</small></div>
        <Space>
          {canPause && onPause && <Button size="small" icon={<PauseOutlined />} aria-label="暂停任务" loading={busy} onClick={onPause} />}
          {canResume && onResume && <Button size="small" icon={<PlayCircleOutlined />} aria-label="继续任务" loading={busy} onClick={onResume} />}
          {canCancel && onCancel && <Button size="small" danger icon={<StopOutlined />} aria-label="取消任务" loading={busy} onClick={onCancel} />}
        </Space>
      </Flex>
      <Progress percent={Math.round(task.progress)} status={task.status === 'failed' ? 'exception' : task.status === 'succeeded' ? 'success' : 'active'} />
      {task.total_bytes != null && <Typography.Text type="secondary">{formatBytes(task.completed_bytes)} / {formatBytes(task.total_bytes)}</Typography.Text>}
      {task.error && <Alert type="error" showIcon message={task.error} />}
    </article>
  )
}
