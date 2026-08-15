import { RobotOutlined, SafetyCertificateOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Form, Input, Select, Space, Typography, message } from 'antd'

import { api } from '../api/client'
import type { Deployment, OperationPlan, Provider, TaskRecord } from '../api/types'
import { ApprovalPanel } from '../components/ApprovalPanel'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'


export function DiagnosticsPage() {
  const queryClient = useQueryClient()
  const providers = useQuery({ queryKey: ['providers'], queryFn: () => api.get<Provider[]>('/api/providers') })
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.get<Deployment[]>('/api/deployments') })
  const plans = useQuery({ queryKey: ['diagnostics'], queryFn: () => api.get<OperationPlan[]>('/api/diagnostics'), refetchInterval: 5_000 })
  const diagnose = useMutation({ mutationFn: (values: Record<string, unknown>) => api.post<OperationPlan>('/api/diagnostics', values), onSuccess: () => { message.success('诊断方案已生成，等待审核'); queryClient.invalidateQueries({ queryKey: ['diagnostics'] }) } })
  const decide = useMutation({ mutationFn: ({ id, action }: { id: string; action: 'approve' | 'reject' }) => api.post<TaskRecord | OperationPlan>(`/api/diagnostics/${id}/${action}`), onSuccess: () => { message.success('方案状态已更新'); queryClient.invalidateQueries() } })
  return (
    <div className="page-stack"><PageHeader title="AI 运维助手" description="将真实设备指标与脱敏日志交给已配置的模型分析，所有动作先审核后执行" /><Alert type="info" showIcon icon={<SafetyCertificateOutlined />} message="AI 无法执行任意 Shell" description="只允许启动、停止、重启部署和重新扫描资产；未知步骤会自动降级为只读说明。" /><section className="diagnostic-composer"><Space align="start"><RobotOutlined className="composer-icon" /><div><h2>创建诊断</h2><Typography.Paragraph type="secondary">选择在线服务和目标部署，描述现象或期望优化的目标。</Typography.Paragraph></div></Space><Form layout="vertical" onFinish={(values) => diagnose.mutate(values)}><div className="form-grid"><Form.Item name="provider_id" label="在线 AI 服务" rules={[{ required: true }]}><Select loading={providers.isLoading} options={providers.data?.filter((item) => item.enabled).map((item) => ({ value: item.id, label: `${item.name} · ${item.default_model}` }))} /></Form.Item><Form.Item name="deployment_id" label="目标部署（可选）"><Select allowClear loading={deployments.isLoading} options={deployments.data?.map((item) => ({ value: item.id, label: item.name }))} /></Form.Item></div><Form.Item name="prompt" label="诊断请求" rules={[{ required: true }]}><Input.TextArea rows={4} placeholder="例如：分析两个模型同时运行时的统一内存占用，并给出不影响现有 API 的优化建议。" /></Form.Item><Button type="primary" htmlType="submit" icon={<RobotOutlined />} loading={diagnose.isPending}>生成诊断方案</Button></Form></section><section className="content-section"><div className="section-heading"><div><h2>诊断与操作计划</h2><p>按时间倒序显示</p></div></div><QueryState loading={plans.isLoading} error={plans.error} empty={!plans.data?.length}>{plans.data?.map((plan) => <ApprovalPanel key={plan.id} plan={plan} busy={decide.isPending} onApprove={() => decide.mutate({ id: plan.id, action: 'approve' })} onReject={() => decide.mutate({ id: plan.id, action: 'reject' })} />)}</QueryState></section></div>
  )
}
