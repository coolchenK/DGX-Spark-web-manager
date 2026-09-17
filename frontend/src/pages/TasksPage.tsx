import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Drawer, Segmented } from 'antd'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import type { TaskRecord } from '../api/types'
import { LogViewer } from '../components/LogViewer'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { TaskProgress } from '../components/TaskProgress'


export function TasksPage() {
  const [filter, setFilter] = useState('all')
  const [detail, setDetail] = useState<TaskRecord | null>(null)
  const queryClient = useQueryClient()
  const tasks = useQuery({ queryKey: ['tasks'], queryFn: () => api.get<TaskRecord[]>('/api/tasks?limit=200'), refetchInterval: 2_000 })
  const action = useMutation({ mutationFn: ({ id, action }: { id: string; action: 'pause' | 'resume' | 'cancel' }) => action === 'cancel' ? api.delete(`/api/tasks/${id}`) : api.post(`/api/tasks/${id}/${action}`), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['tasks'] }) })
  const data = useMemo(() => (tasks.data ?? []).filter((item) => filter === 'all' || item.status === filter), [tasks.data, filter])
  return (
    <div className="page-stack"><PageHeader title="任务中心" description="下载、部署、诊断和运维动作的持久执行记录" extra={<Segmented value={filter} onChange={(value) => setFilter(String(value))} options={[{ label: '全部', value: 'all' }, { label: '运行中', value: 'running' }, { label: '失败', value: 'failed' }, { label: '已完成', value: 'succeeded' }]} />} /><QueryState loading={tasks.isLoading} error={tasks.error} empty={!data.length}>{data.map((task) => <div key={task.id} onDoubleClick={() => setDetail(task)}><TaskProgress task={task} busy={action.isPending} onPause={() => action.mutate({ id: task.id, action: 'pause' })} onResume={() => action.mutate({ id: task.id, action: 'resume' })} onCancel={() => action.mutate({ id: task.id, action: 'cancel' })} /></div>)}</QueryState><Drawer title={detail?.title} width={760} open={Boolean(detail)} onClose={() => setDetail(null)}><LogViewer value={detail?.log ?? ''} filename={`task-${detail?.id}.log`} /></Drawer></div>
  )
}
