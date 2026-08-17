import { HistoryOutlined, PlusOutlined, RobotOutlined, SafetyCertificateOutlined, SendOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Drawer, Empty, Form, Grid, Input, Select, Spin, Typography, message } from 'antd'
import dayjs from 'dayjs'
import { useEffect, useMemo, useState } from 'react'

import { api } from '../api/client'
import type { Deployment, OpsSession, OpsSessionSummary, OperationPlan, Provider, TaskRecord } from '../api/types'
import { OpsConversation } from '../components/OpsConversation'
import { PageHeader } from '../components/PageHeader'
import { StatusBadge } from '../components/StatusBadge'


interface ComposerValues {
  provider_id?: string
  deployment_id?: string
  content: string
}

const terminalTaskStatuses = new Set(['succeeded', 'failed', 'cancelled'])

function SessionList({
  sessions,
  selectedId,
  loading,
  error,
  onSelect,
}: {
  sessions: OpsSessionSummary[]
  selectedId: string | null
  loading: boolean
  error?: string
  onSelect: (id: string) => void
}) {
  if (loading) return <div className="ops-session-state"><Spin size="small" /></div>
  if (error) return <Alert type="error" showIcon message="会话加载失败" description={error} />
  if (!sessions.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚无运维会话" />
  return (
    <nav className="ops-session-list" aria-label="运维会话">
      {sessions.map((session) => (
        <button
          className={selectedId === session.id ? 'ops-session-item is-selected' : 'ops-session-item'}
          key={session.id}
          type="button"
          onClick={() => onSelect(session.id)}
          aria-current={selectedId === session.id ? 'page' : undefined}
        >
          <span className="ops-session-title">{session.title}</span>
          <span className="ops-session-meta">
            <span>{session.deployment_name ?? session.provider_name ?? '全局'}</span>
            <time dateTime={session.updated_at}>{dayjs(session.updated_at).format('MM-DD HH:mm')}</time>
          </span>
        </button>
      ))}
    </nav>
  )
}

export function DiagnosticsPage() {
  const queryClient = useQueryClient()
  const screens = Grid.useBreakpoint()
  const [form] = Form.useForm<ComposerValues>()
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [sessionDrawerOpen, setSessionDrawerOpen] = useState(false)
  const [pendingTaskId, setPendingTaskId] = useState<string | null>(null)

  const providers = useQuery({ queryKey: ['providers'], queryFn: () => api.get<Provider[]>('/api/providers') })
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.get<Deployment[]>('/api/deployments') })
  const sessions = useQuery({
    queryKey: ['diagnostics', 'sessions'],
    queryFn: () => api.get<OpsSessionSummary[]>('/api/diagnostics/sessions'),
    refetchInterval: 5_000,
  })
  const session = useQuery({
    queryKey: ['diagnostics', 'sessions', selectedSessionId],
    queryFn: () => api.get<OpsSession>(`/api/diagnostics/sessions/${selectedSessionId}`),
    enabled: Boolean(selectedSessionId),
    refetchInterval: (query) => query.state.data?.status === 'processing' ? 2_000 : false,
  })
  const pendingTask = useQuery({
    queryKey: ['tasks', pendingTaskId],
    queryFn: () => api.get<TaskRecord>(`/api/tasks/${pendingTaskId}`),
    enabled: Boolean(pendingTaskId),
    refetchInterval: (query) => terminalTaskStatuses.has(query.state.data?.status ?? '') ? false : 1_500,
  })

  useEffect(() => {
    if (!pendingTask.data || !terminalTaskStatuses.has(pendingTask.data.status)) return
    const failed = pendingTask.data.status !== 'succeeded'
    if (failed) message.error(pendingTask.data.error || 'AI 运维响应失败')
    void queryClient.invalidateQueries({ queryKey: ['diagnostics'] })
    void queryClient.invalidateQueries({ queryKey: ['tasks'] })
    setPendingTaskId(null)
  }, [pendingTask.data, queryClient])

  const send = useMutation({
    mutationFn: async (values: ComposerValues) => {
      const content = values.content.trim()
      let sessionId = selectedSessionId
      if (!sessionId) {
        const created = await api.post<OpsSessionSummary>('/api/diagnostics/sessions', {
          provider_id: values.provider_id,
          deployment_id: values.deployment_id || undefined,
          title: content.slice(0, 255),
        })
        sessionId = created.id
        setSelectedSessionId(created.id)
        queryClient.setQueryData<OpsSessionSummary[]>(['diagnostics', 'sessions'], (current = []) => [created, ...current])
      }
      const task = await api.post<TaskRecord>(`/api/diagnostics/sessions/${sessionId}/messages`, { content })
      return task
    },
    onSuccess: (task) => {
      setPendingTaskId(task.id)
      form.setFieldValue('content', '')
      message.success('请求已提交')
      void queryClient.invalidateQueries({ queryKey: ['diagnostics'] })
      void queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error) => message.error(error instanceof Error ? error.message : '请求提交失败'),
  })

  const decide = useMutation({
    mutationFn: ({ id, action }: { id: string; action: 'approve' | 'reject' }) => api.post<TaskRecord | OperationPlan>(`/api/diagnostics/${id}/${action}`),
    onSuccess: (result) => {
      if ('type' in result) setPendingTaskId(result.id)
      message.success('方案状态已更新')
      void queryClient.invalidateQueries({ queryKey: ['diagnostics'] })
      void queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error) => message.error(error instanceof Error ? error.message : '方案状态更新失败'),
  })

  const enabledProviders = useMemo(() => providers.data?.filter((item) => item.enabled) ?? [], [providers.data])
  const selectedSummary = sessions.data?.find((item) => item.id === selectedSessionId)
  const busy = send.isPending || Boolean(pendingTaskId) || selectedSummary?.status === 'processing'

  const selectSession = (id: string) => {
    setSelectedSessionId(id)
    setSessionDrawerOpen(false)
    form.setFieldValue('content', '')
  }
  const newSession = () => {
    setSelectedSessionId(null)
    setSessionDrawerOpen(false)
    form.resetFields()
  }
  const sessionList = (
    <SessionList
      sessions={sessions.data ?? []}
      selectedId={selectedSessionId}
      loading={sessions.isLoading}
      error={sessions.error?.message}
      onSelect={selectSession}
    />
  )

  return (
    <div className="page-stack diagnostics-page">
      <PageHeader
        title="AI 运维助手"
        description="基于实时设备状态分析问题；只读检查自动运行，变更命令逐项审批"
        extra={(
          <>
            {!screens.md && <Button icon={<HistoryOutlined />} onClick={() => setSessionDrawerOpen(true)}>历史会话</Button>}
            <Button type="primary" icon={<PlusOutlined />} onClick={newSession}>新建会话</Button>
          </>
        )}
      />
      <Alert
        type="info"
        showIcon
        icon={<SafetyCertificateOutlined />}
        message="Shell 变更受审批与审计保护"
        description="助手可自动采集只读证据；任何 Shell 命令必须显示完整命令、影响和回滚方式，并由管理员确认后执行。"
      />
      <section className="ops-workspace">
        {screens.md && (
          <aside className="ops-session-rail">
            <div className="ops-rail-heading"><span>会话</span><Typography.Text type="secondary">{sessions.data?.length ?? 0}</Typography.Text></div>
            {sessionList}
          </aside>
        )}
        <main className="ops-main">
          {selectedSessionId ? (
            <>
              <header className="ops-thread-header">
                <div>
                  <h2>{session.data?.title ?? selectedSummary?.title ?? '加载会话'}</h2>
                  <p>{session.data?.deployment_name ?? session.data?.provider_name ?? selectedSummary?.deployment_name ?? selectedSummary?.provider_name}</p>
                </div>
                <StatusBadge status={session.data?.status ?? selectedSummary?.status ?? 'unknown'} />
              </header>
              <div className="ops-thread-body">
                {session.isLoading && <div className="ops-session-state"><Spin /></div>}
                {session.error && <Alert type="error" showIcon message="会话加载失败" description={session.error.message} />}
                {session.data && (
                  <OpsConversation
                    session={session.data}
                    busy={decide.isPending}
                    onApprove={(plan) => decide.mutate({ id: plan.id, action: 'approve' })}
                    onReject={(plan) => decide.mutate({ id: plan.id, action: 'reject' })}
                  />
                )}
              </div>
            </>
          ) : (
            <div className="ops-new-session">
              <RobotOutlined />
              <h2>新建运维会话</h2>
              <Typography.Text type="secondary">选择负责分析的在线 AI 服务。</Typography.Text>
            </div>
          )}

          <Form<ComposerValues> form={form} layout="vertical" className="ops-composer" onFinish={(values) => send.mutate(values)}>
            {!selectedSessionId && providers.error && <Alert type="error" showIcon message="在线 AI 服务加载失败" description={providers.error.message} />}
            {!selectedSessionId && (
              <div className="form-grid">
                <Form.Item name="provider_id" label="在线 AI 服务" rules={[{ required: true, message: '请选择在线 AI 服务' }]}>
                  <Select
                    loading={providers.isLoading}
                    options={enabledProviders.map((item) => ({ value: item.id, label: `${item.name} · ${item.default_model}` }))}
                    placeholder="选择 Provider"
                  />
                </Form.Item>
                <Form.Item name="deployment_id" label="目标部署（可选）">
                  <Select
                    allowClear
                    loading={deployments.isLoading}
                    options={deployments.data?.map((item) => ({ value: item.id, label: item.name }))}
                    placeholder="全局诊断"
                  />
                </Form.Item>
              </div>
            )}
            <Form.Item name="content" label="运维请求" rules={[{ required: true, whitespace: true, message: '请输入运维请求' }]}>
              <Input.TextArea autoSize={{ minRows: 3, maxRows: 7 }} maxLength={10_000} placeholder="描述异常现象或需要完成的运维目标" />
            </Form.Item>
            <div className="ops-composer-actions">
              <Typography.Text type="secondary">{busy ? '正在处理当前请求' : '所有变更操作均需审批'}</Typography.Text>
              <Button aria-label="发送请求" type="primary" htmlType="submit" icon={<SendOutlined />} loading={send.isPending} disabled={busy}>发送请求</Button>
            </div>
          </Form>
        </main>
      </section>
      <Drawer title="运维会话" placement="left" width="min(360px, 100vw)" open={sessionDrawerOpen} onClose={() => setSessionDrawerOpen(false)}>
        {sessionList}
      </Drawer>
    </div>
  )
}
