import { LockOutlined, UserOutlined } from '@ant-design/icons'
import { Alert, Button, Form, Input, Typography } from 'antd'
import { useMutation } from '@tanstack/react-query'

import { api } from '../api/client'
import type { User } from '../api/types'
import { useAuthStore } from '../stores/auth'


interface LoginResponse { user: User; csrf_token: string }


export function LoginPage() {
  const setSession = useAuthStore((state) => state.setSession)
  const login = useMutation({
    mutationFn: (values: { username: string; password: string }) => api.post<LoginResponse>('/api/auth/login', values),
    onSuccess: (result) => setSession(result.user, result.csrf_token),
  })

  return (
    <main className="login-page">
      <section className="login-panel">
        <div className="login-brand"><span className="brand-mark brand-mark-large">DS</span></div>
        <Typography.Title level={1}>DGX Spark 管理器</Typography.Title>
        <Typography.Paragraph type="secondary">登录到本机模型与推理服务管理平面</Typography.Paragraph>
        {login.error && <Alert type="error" showIcon message="登录失败" description={login.error.message} />}
        <Form layout="vertical" requiredMark={false} onFinish={(values) => login.mutate(values)} initialValues={{ username: 'admin' }}>
          <Form.Item label="用户名" name="username" rules={[{ required: true, message: '请输入用户名' }]}>
            <Input prefix={<UserOutlined />} autoComplete="username" />
          </Form.Item>
          <Form.Item label="密码" name="password" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password prefix={<LockOutlined />} autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={login.isPending} block>登录</Button>
        </Form>
      </section>
      <aside className="login-status" aria-label="产品能力">
        <div><span>01</span><strong>模型资产</strong><small>发现本地缓存与 Hugging Face 模型</small></div>
        <div><span>02</span><strong>推理实例</strong><small>统一管理 SGLang 与 vLLM 服务</small></div>
        <div><span>03</span><strong>受控运维</strong><small>AI 诊断经过审核后才会执行</small></div>
      </aside>
    </main>
  )
}
