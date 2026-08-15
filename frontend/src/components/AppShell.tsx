import { useState, type ReactNode } from 'react'
import {
  ApiOutlined,
  AppstoreOutlined,
  AuditOutlined,
  CloudDownloadOutlined,
  DashboardOutlined,
  DeploymentUnitOutlined,
  MenuOutlined,
  MoonOutlined,
  RobotOutlined,
  SettingOutlined,
  SunOutlined,
  UnorderedListOutlined,
  UserOutlined,
} from '@ant-design/icons'
import { Avatar, Button, Drawer, Dropdown, Grid, Layout, Menu, Space, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'

import { api } from '../api/client'
import { useAuthStore } from '../stores/auth'
import { type ThemeMode, useThemeStore } from '../stores/theme'


const items = [
  { key: '/', icon: <DashboardOutlined />, label: '系统概览' },
  { key: '/models', icon: <AppstoreOutlined />, label: '模型库' },
  { key: '/huggingface', icon: <CloudDownloadOutlined />, label: 'Hugging Face' },
  { key: '/deployments', icon: <DeploymentUnitOutlined />, label: '部署实例' },
  { key: '/gateway', icon: <ApiOutlined />, label: 'API 网关' },
  { key: '/providers', icon: <RobotOutlined />, label: '在线 AI 服务' },
  { key: '/diagnostics', icon: <RobotOutlined />, label: 'AI 运维助手' },
  { key: '/tasks', icon: <UnorderedListOutlined />, label: '任务中心' },
  { key: '/audit', icon: <AuditOutlined />, label: '日志与审计' },
  { key: '/settings', icon: <SettingOutlined />, label: '系统设置' },
]


function Brand() {
  return (
    <div className="brand-lockup" aria-label="DGX Spark 管理器">
      <span className="brand-mark">DS</span>
      <span className="brand-copy">
        <strong>DGX Spark</strong>
        <small>Web Manager</small>
      </span>
    </div>
  )
}


export function ThemeMenu() {
  const mode = useThemeStore((state) => state.mode)
  const setMode = useThemeStore((state) => state.setMode)
  const options: Array<{ key: ThemeMode; label: string }> = [
    { key: 'light', label: '浅色' },
    { key: 'dark', label: '深色' },
    { key: 'system', label: '跟随系统' },
  ]
  return (
    <Dropdown
      trigger={['click']}
      menu={{
        selectedKeys: [mode],
        items: options,
        onClick: ({ key }) => setMode(key as ThemeMode),
      }}
    >
      <Button
        type="text"
        aria-label="切换主题"
        icon={mode === 'dark' ? <MoonOutlined /> : <SunOutlined />}
      />
    </Dropdown>
  )
}


export function AppShell({ children }: { children: ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const screens = Grid.useBreakpoint()
  const mobile = !screens.md
  const location = useLocation()
  const navigate = useNavigate()
  const user = useAuthStore((state) => state.user)
  const clearSession = useAuthStore((state) => state.clearSession)

  const menu = (
    <Menu
      mode="inline"
      selectedKeys={[location.pathname]}
      items={items}
      onClick={({ key }) => {
        navigate(key)
        setDrawerOpen(false)
      }}
    />
  )

  const signOut = async () => {
    try {
      await api.post('/api/auth/logout')
    } finally {
      clearSession()
    }
  }

  return (
    <Layout className="app-layout">
      {!mobile && (
        <Layout.Sider width={224} className="app-sider">
          <Brand />
          <nav aria-label="主导航">{menu}</nav>
          <div className="sider-foot">
            <span className="connection-dot" aria-hidden="true" />
            管理服务已连接
          </div>
        </Layout.Sider>
      )}
      <Layout>
        <Layout.Header className="app-header">
          <Space size={8}>
            {mobile && (
              <Button type="text" aria-label="打开导航" icon={<MenuOutlined />} onClick={() => setDrawerOpen(true)} />
            )}
            {mobile && <Brand />}
          </Space>
          <Space size={4}>
            <ThemeMenu />
            <Dropdown
              trigger={['click']}
              menu={{ items: [{ key: 'logout', label: '退出登录', onClick: signOut }] }}
            >
              <Button type="text" className="user-trigger">
                <Avatar size={26} icon={<UserOutlined />} />
                {!mobile && <Typography.Text>{user?.username}</Typography.Text>}
              </Button>
            </Dropdown>
          </Space>
        </Layout.Header>
        <Layout.Content className="app-content">{children}</Layout.Content>
      </Layout>
      <Drawer
        placement="left"
        width={280}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        title={<Brand />}
        styles={{ body: { padding: '8px 0' } }}
      >
        <nav aria-label="移动端主导航">{menu}</nav>
      </Drawer>
    </Layout>
  )
}
