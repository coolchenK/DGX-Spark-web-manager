import { CheckCircleOutlined, DeleteOutlined, MoonOutlined, SafetyCertificateOutlined, SunOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Descriptions, Form, Input, Modal, Radio, Space, Tag, Typography, message } from 'antd'
import { useState } from 'react'

import { api } from '../api/client'
import type { HistoryClearResult, ManagerSettings, SystemSnapshot } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { type ThemeMode, useThemeStore } from '../stores/theme'


const confirmationPhrase = '清除历史记录'
const affectedQueryKeys = [
  ['tasks'],
  ['diagnostics'],
  ['ops-sessions'],
  ['audit'],
  ['system'],
  ['gateway-stats'],
]

export function SettingsPage() {
  const mode = useThemeStore((state) => state.mode)
  const setMode = useThemeStore((state) => state.setMode)
  const queryClient = useQueryClient()
  const [clearOpen, setClearOpen] = useState(false)
  const [confirmation, setConfirmation] = useState('')
  const [clearError, setClearError] = useState<string | null>(null)
  const system = useQuery({ queryKey: ['system'], queryFn: () => api.get<SystemSnapshot>('/api/system') })
  const settings = useQuery({ queryKey: ['settings'], queryFn: () => api.get<ManagerSettings>('/api/settings') })
  const updateToken = useMutation({
    mutationFn: (token: string | null) => api.patch<{ token_configured: boolean }>('/api/settings/huggingface', { token }),
    onSuccess: () => {
      message.success('Hugging Face 设置已更新')
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
  })
  const clearHistory = useMutation({
    mutationFn: () => api.delete<HistoryClearResult>('/api/settings/alerts-diagnostics-history', { confirmation }),
    onSuccess: (result) => {
      const total = Object.values(result.deleted).reduce((sum, count) => sum + count, 0)
      message.success(`已物理删除 ${total} 条告警与诊断历史`)
      setClearOpen(false)
      setConfirmation('')
      setClearError(null)
      for (const queryKey of affectedQueryKeys) void queryClient.invalidateQueries({ queryKey })
    },
    onError: (error) => setClearError(error instanceof Error ? error.message : '历史记录清除失败'),
  })

  const openClearDialog = () => {
    setConfirmation('')
    setClearError(null)
    clearHistory.reset()
    setClearOpen(true)
  }
  const closeClearDialog = () => {
    if (clearHistory.isPending) return
    setClearOpen(false)
    setConfirmation('')
    setClearError(null)
  }

  return (
    <div className="page-stack">
      <PageHeader title="系统设置" description="管理界面偏好与当前运行环境" />

      <section className="settings-section">
        <div><h2>外观</h2><p>主题偏好会保存在当前浏览器。</p></div>
        <Radio.Group value={mode} onChange={(event) => setMode(event.target.value as ThemeMode)}>
          <Radio.Button value="light"><SunOutlined /> 浅色</Radio.Button>
          <Radio.Button value="dark"><MoonOutlined /> 深色</Radio.Button>
          <Radio.Button value="system">跟随系统</Radio.Button>
        </Radio.Group>
      </section>

      <section className="settings-section settings-secret">
        <div>
          <h2>Hugging Face 访问</h2>
          <p>{settings.data?.huggingface.token_configured ? 'Token 已加密配置，可访问私有或受限模型。' : '当前仅能访问公开模型。'}</p>
          <Typography.Text type="secondary">缓存：{settings.data?.huggingface.cache_dir ?? '加载中'}</Typography.Text>
        </div>
        <Form layout="inline" onFinish={(values) => updateToken.mutate(values.token)}>
          <Form.Item name="token" rules={[{ required: true, message: '请输入 Token' }]}><Input.Password placeholder="hf_..." autoComplete="off" /></Form.Item>
          <Button htmlType="submit" type="primary" loading={updateToken.isPending}>保存</Button>
          {settings.data?.huggingface.token_configured && <Button danger loading={updateToken.isPending} onClick={() => updateToken.mutate(null)}>清除</Button>}
        </Form>
      </section>

      <section className="settings-section">
        <div><h2>安全边界</h2><p>管理服务启用会话、CSRF、密钥加密和操作审计。</p></div>
        <Space wrap>
          <Tag icon={<CheckCircleOutlined />} color="success">HttpOnly 会话</Tag>
          <Tag icon={<CheckCircleOutlined />} color="success">CSRF 防护</Tag>
          <Tag icon={<SafetyCertificateOutlined />} color="success">AI 操作审批</Tag>
        </Space>
      </section>

      <section className="settings-danger">
        <div>
          <Typography.Text type="danger">危险操作</Typography.Text>
          <h2>告警与诊断历史</h2>
          <p>物理删除失败任务、诊断方案、AI 运维会话及相关审计记录。</p>
        </div>
        <Button danger icon={<DeleteOutlined />} aria-label="清除告警与诊断信息" onClick={openClearDialog}>清除告警与诊断信息</Button>
      </section>

      <section className="content-section">
        <div className="section-heading"><div><h2>运行环境</h2><p>由服务端实时检测</p></div></div>
        <QueryState loading={system.isLoading} error={system.error}>
          {system.data && (
            <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }} items={[
              { key: 'host', label: '主机名', children: system.data.hostname },
              { key: 'arch', label: '架构', children: system.data.architecture },
              { key: 'os', label: '操作系统', children: system.data.os },
              { key: 'kernel', label: '内核', children: system.data.kernel },
              { key: 'gpu', label: 'GPU', children: system.data.gpus[0]?.name ?? '未检测到' },
              { key: 'driver', label: '驱动', children: system.data.gpus[0]?.driver_version ?? '不支持' },
            ]} />
          )}
        </QueryState>
      </section>
      <Typography.Text type="secondary">DGX Spark Web Manager · ARM64 native</Typography.Text>

      <Modal
        title="清除告警与诊断信息"
        open={clearOpen}
        onCancel={closeClearDialog}
        closable={!clearHistory.isPending}
        maskClosable={!clearHistory.isPending}
        destroyOnHidden
        footer={[
          <Button key="cancel" onClick={closeClearDialog} disabled={clearHistory.isPending}>取消</Button>,
          <Button
            key="clear"
            type="primary"
            danger
            aria-label="永久清除"
            icon={<DeleteOutlined />}
            loading={clearHistory.isPending}
            disabled={confirmation !== confirmationPhrase}
            onClick={() => clearHistory.mutate()}
          >
            永久清除
          </Button>,
        ]}
      >
        <div className="history-clear-dialog">
          <Alert type="warning" showIcon message="此操作不可撤销" description="失败任务、诊断方案和 AI 运维会话将从数据库中物理删除。" />
          <p>同时删除会话消息、工具输出及相关审计明细。</p>
          <p>模型、部署、Provider、API Key、Hugging Face 设置、网关指标和成功任务不会删除。</p>
          {clearError && <Alert type="error" showIcon message="无法清除历史记录" description={clearError} />}
          <label htmlFor="history-clear-confirmation">输入 <strong>{confirmationPhrase}</strong> 以确认</label>
          <Input
            id="history-clear-confirmation"
            aria-label="输入确认短语"
            value={confirmation}
            status={clearError ? 'error' : undefined}
            disabled={clearHistory.isPending}
            autoComplete="off"
            onChange={(event) => {
              setConfirmation(event.target.value)
              setClearError(null)
            }}
          />
        </div>
      </Modal>
    </div>
  )
}
