import { CheckCircleOutlined, MoonOutlined, SafetyCertificateOutlined, SunOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Descriptions, Form, Input, Radio, Space, Tag, Typography, message } from 'antd'

import { api } from '../api/client'
import type { ManagerSettings, SystemSnapshot } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { type ThemeMode, useThemeStore } from '../stores/theme'


export function SettingsPage() {
  const mode = useThemeStore((state) => state.mode)
  const setMode = useThemeStore((state) => state.setMode)
  const queryClient = useQueryClient()
  const system = useQuery({ queryKey: ['system'], queryFn: () => api.get<SystemSnapshot>('/api/system') })
  const settings = useQuery({ queryKey: ['settings'], queryFn: () => api.get<ManagerSettings>('/api/settings') })
  const updateToken = useMutation({ mutationFn: (token: string | null) => api.patch<{ token_configured: boolean }>('/api/settings/huggingface', { token }), onSuccess: () => { message.success('Hugging Face 设置已更新'); queryClient.invalidateQueries({ queryKey: ['settings'] }) } })
  return (
    <div className="page-stack"><PageHeader title="系统设置" description="管理界面偏好与当前运行环境" /><section className="settings-section"><div><h2>外观</h2><p>主题偏好会保存在当前浏览器。</p></div><Radio.Group value={mode} onChange={(event) => setMode(event.target.value as ThemeMode)}><Radio.Button value="light"><SunOutlined /> 浅色</Radio.Button><Radio.Button value="dark"><MoonOutlined /> 深色</Radio.Button><Radio.Button value="system">跟随系统</Radio.Button></Radio.Group></section><section className="settings-section settings-secret"><div><h2>Hugging Face 访问</h2><p>{settings.data?.huggingface.token_configured ? 'Token 已加密配置，可访问私有或受限模型。' : '当前仅能访问公开模型。'}</p><Typography.Text type="secondary">缓存：{settings.data?.huggingface.cache_dir ?? '加载中'}</Typography.Text></div><Form layout="inline" onFinish={(values) => updateToken.mutate(values.token)}><Form.Item name="token" rules={[{ required: true, message: '请输入 Token' }]}><Input.Password placeholder="hf_..." autoComplete="off" /></Form.Item><Button htmlType="submit" type="primary" loading={updateToken.isPending}>保存</Button>{settings.data?.huggingface.token_configured && <Button danger loading={updateToken.isPending} onClick={() => updateToken.mutate(null)}>清除</Button>}</Form></section><section className="settings-section"><div><h2>安全边界</h2><p>管理服务启用会话、CSRF、密钥加密和操作审计。</p></div><Space wrap><Tag icon={<CheckCircleOutlined />} color="success">HttpOnly 会话</Tag><Tag icon={<CheckCircleOutlined />} color="success">CSRF 防护</Tag><Tag icon={<SafetyCertificateOutlined />} color="success">AI 操作白名单</Tag></Space></section><section className="content-section"><div className="section-heading"><div><h2>运行环境</h2><p>由服务端实时检测</p></div></div><QueryState loading={system.isLoading} error={system.error}>{system.data && <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }} items={[{ key: 'host', label: '主机名', children: system.data.hostname }, { key: 'arch', label: '架构', children: system.data.architecture }, { key: 'os', label: '操作系统', children: system.data.os }, { key: 'kernel', label: '内核', children: system.data.kernel }, { key: 'gpu', label: 'GPU', children: system.data.gpus[0]?.name ?? '未检测到' }, { key: 'driver', label: '驱动', children: system.data.gpus[0]?.driver_version ?? '不支持' }]} />}</QueryState></section><Typography.Text type="secondary">DGX Spark Web Manager · ARM64 native</Typography.Text></div>
  )
}
