import { ApiOutlined, DatabaseOutlined, HddOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Flex, List, Space, Typography, message } from 'antd'

import { api } from '../api/client'
import type { Deployment, GatewayStats, OperationPlan, SystemSnapshot, TaskRecord } from '../api/types'
import { MetricStrip } from '../components/MetricStrip'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes, formatDate, formatDuration, percent } from '../utils/format'


export function DashboardPage() {
  const queryClient = useQueryClient()
  const system = useQuery({ queryKey: ['system'], queryFn: () => api.get<SystemSnapshot>('/api/system'), refetchInterval: 15_000 })
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.get<Deployment[]>('/api/deployments'), refetchInterval: 10_000 })
  const tasks = useQuery({ queryKey: ['tasks', 6], queryFn: () => api.get<TaskRecord[]>('/api/tasks?limit=6'), refetchInterval: 3_000 })
  const stats = useQuery({ queryKey: ['gateway-stats'], queryFn: () => api.get<GatewayStats>('/api/gateway/stats'), refetchInterval: 15_000 })
  const diagnostics = useQuery({ queryKey: ['diagnostics'], queryFn: () => api.get<OperationPlan[]>('/api/diagnostics'), refetchInterval: 15_000 })
  const scan = useMutation({
    mutationFn: () => api.post<{ models: number; deployments: number }>('/api/discovery/scan'),
    onSuccess: (result) => {
      message.success(`发现 ${result.models} 个模型和 ${result.deployments} 个部署`)
      queryClient.invalidateQueries()
    },
  })

  const snapshot = system.data
  const gpu = snapshot?.gpus[0]
  const metrics = snapshot ? [
    { label: 'CPU', value: `${snapshot.cpu.percent.toFixed(0)}%`, detail: `${snapshot.cpu.cores} 核心`, percent: snapshot.cpu.percent, icon: <ThunderboltOutlined /> },
    { label: '统一内存', value: formatBytes(snapshot.memory.used_bytes), detail: `共 ${formatBytes(snapshot.memory.total_bytes)}`, percent: percent(snapshot.memory.used_bytes, snapshot.memory.total_bytes), icon: <DatabaseOutlined /> },
    { label: '系统磁盘', value: formatBytes(snapshot.disk.free_bytes), detail: '可用空间', percent: percent(snapshot.disk.used_bytes, snapshot.disk.total_bytes), icon: <HddOutlined /> },
    { label: 'GPU', value: gpu ? (gpu.utilization_percent == null ? '不支持' : `${gpu.utilization_percent}%`) : '未检测到', detail: gpu?.name ?? 'NVIDIA GPU', percent: gpu?.utilization_percent, icon: <ApiOutlined /> },
  ] : []

  return (
    <div className="page-stack">
      <PageHeader title="系统概览" description="设备资源、模型服务与 API 流量的实时状态" extra={<Button loading={scan.isPending} onClick={() => scan.mutate()}>重新发现</Button>} />
      <QueryState loading={system.isLoading} error={system.error} onRetry={() => system.refetch()}>
        {snapshot && (
          <>
            <section className="device-banner">
              <div><span className="online-pulse" /><div><Typography.Text type="secondary">设备</Typography.Text><strong>{snapshot.hostname}</strong></div></div>
              <dl>
                <div><dt>架构</dt><dd>{snapshot.architecture}</dd></div>
                <div><dt>系统</dt><dd>{snapshot.os}</dd></div>
                <div><dt>GPU 驱动</dt><dd>{gpu?.driver_version ?? '未检测到'}</dd></div>
                <div><dt>GPU 温度</dt><dd>{gpu ? (gpu.temperature_c == null ? '不支持' : `${gpu.temperature_c} °C`) : '未检测到'}</dd></div>
                <div><dt>GPU 功耗</dt><dd>{gpu ? (gpu.power_w == null ? '不支持' : `${gpu.power_w} W`) : '未检测到'}</dd></div>
                <div><dt>运行时间</dt><dd>{formatDuration(snapshot.uptime_seconds)}</dd></div>
              </dl>
            </section>
            <MetricStrip metrics={metrics} />
          </>
        )}
      </QueryState>

      <section className="dashboard-grid">
        <div className="content-section deployment-section">
          <div className="section-heading"><div><h2>推理服务</h2><p>已发现并接入网关的本机端点</p></div><Typography.Text type="secondary">{deployments.data?.length ?? 0} 个实例</Typography.Text></div>
          <QueryState loading={deployments.isLoading} error={deployments.error} empty={!deployments.data?.length}>
            <ResponsiveDataView data={deployments.data ?? []} rowKey="id" columns={[
              { title: '实例', dataIndex: 'name', render: (_, item) => <div className="primary-cell"><strong>{item.name}</strong><small>{item.api_model_name}</small></div> },
              { title: '运行时', dataIndex: 'runtime', width: 100 },
              { title: '端点', dataIndex: 'endpoint_url', responsive: ['lg'] },
              { title: '状态', dataIndex: 'health', width: 96, render: (value) => <StatusBadge status={value} /> },
            ]} renderMobile={(item) => <div className="mobile-record"><Flex justify="space-between"><strong>{item.name}</strong><StatusBadge status={item.health} /></Flex><Typography.Text type="secondary">{item.api_model_name}</Typography.Text><dl><div><dt>运行时</dt><dd>{item.runtime}</dd></div><div><dt>端点</dt><dd>{item.endpoint_url}</dd></div></dl></div>} />
          </QueryState>
        </div>
        <div className="content-section activity-section">
          <div className="section-heading"><div><h2>最近任务</h2><p>后台下载、部署和运维动作</p></div></div>
          <QueryState loading={tasks.isLoading} error={tasks.error} empty={!tasks.data?.length}>
            <List dataSource={tasks.data} renderItem={(task) => (
              <List.Item><Flex vertical gap={3} className="task-summary"><Space><StatusBadge status={task.status} /><strong>{task.title}</strong></Space><Typography.Text type="secondary">{formatDate(task.updated_at)}</Typography.Text></Flex></List.Item>
            )} />
          </QueryState>
        </div>
      </section>

      <section className="traffic-band">
        <div><span>最近一分钟请求</span><strong>{stats.data?.requests_last_minute ?? 0}</strong></div>
        <div><span>当前并发</span><strong>{stats.data?.active_requests ?? 0}</strong></div>
        <div><span>错误率</span><strong>{((stats.data?.error_rate ?? 0) * 100).toFixed(1)}%</strong></div>
        <div><span>Token 吞吐</span><strong>{stats.data?.tokens_per_second ?? 0}/s</strong></div>
      </section>
      {(tasks.data?.some((task) => task.status === 'failed') || diagnostics.data?.[0]) && <section className="content-section"><div className="section-heading"><div><h2>告警与诊断</h2><p>仅显示真实任务和 AI 诊断记录</p></div></div><Space direction="vertical" size={10} className="dashboard-alerts">{tasks.data?.filter((task) => task.status === 'failed').slice(0, 2).map((task) => <Alert key={task.id} type="error" showIcon message={task.title} description={task.error ?? '任务执行失败'} />)}{diagnostics.data?.slice(0, 2).map((plan) => <Alert key={plan.id} type={plan.status === 'pending' ? 'warning' : 'info'} showIcon message={plan.summary} description={`${plan.status} · ${plan.diagnosis}`} />)}</Space></section>}
    </div>
  )
}
