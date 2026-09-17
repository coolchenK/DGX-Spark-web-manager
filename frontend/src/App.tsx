import { lazy, Suspense, useEffect, useMemo } from 'react'
import { ConfigProvider, Spin, theme as antdTheme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { api } from './api/client'
import type { User } from './api/types'
import { AppShell } from './components/AppShell'
import { LoginPage } from './pages/LoginPage'
import { DashboardPage } from './pages/DashboardPage'
import { useAuthStore } from './stores/auth'
import { resolveTheme, useThemeStore } from './stores/theme'
import './styles.css'

const ModelsPage = lazy(() => import('./pages/ModelsPage').then((module) => ({ default: module.ModelsPage })))
const HuggingFacePage = lazy(() => import('./pages/HuggingFacePage').then((module) => ({ default: module.HuggingFacePage })))
const DeploymentsPage = lazy(() => import('./pages/DeploymentsPage').then((module) => ({ default: module.DeploymentsPage })))
const GatewayPage = lazy(() => import('./pages/GatewayPage').then((module) => ({ default: module.GatewayPage })))
const ProvidersPage = lazy(() => import('./pages/ProvidersPage').then((module) => ({ default: module.ProvidersPage })))
const DiagnosticsPage = lazy(() => import('./pages/DiagnosticsPage').then((module) => ({ default: module.DiagnosticsPage })))
const TasksPage = lazy(() => import('./pages/TasksPage').then((module) => ({ default: module.TasksPage })))
const AuditPage = lazy(() => import('./pages/AuditPage').then((module) => ({ default: module.AuditPage })))
const SettingsPage = lazy(() => import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })))

interface SessionStatus {
  authenticated: boolean
  user: User | null
  csrf_token: string | null
}


function AuthenticatedApp() {
  const user = useAuthStore((state) => state.user)
  const setSession = useAuthStore((state) => state.setSession)
  const clearSession = useAuthStore((state) => state.clearSession)
  const session = useQuery({
    queryKey: ['auth', 'session'],
    queryFn: () => api.get<SessionStatus>('/api/auth/session'),
    enabled: !user,
    retry: false,
  })

  useEffect(() => {
    if (session.data?.authenticated && session.data.user) {
      setSession(session.data.user, session.data.csrf_token ?? '')
    } else if (session.data && !session.data.authenticated) {
      clearSession()
    }
  }, [session.data, setSession, clearSession])

  if (!user && session.isLoading) {
    return (
      <main className="full-page-state" aria-label="正在加载">
        <Spin size="large" />
      </main>
    )
  }
  if (!user) return <LoginPage />

  return (
    <AppShell>
      <Suspense fallback={<main className="route-loading"><Spin /></main>}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/huggingface" element={<HuggingFacePage />} />
          <Route path="/deployments" element={<DeploymentsPage />} />
          <Route path="/gateway" element={<GatewayPage />} />
          <Route path="/providers" element={<ProvidersPage />} />
          <Route path="/diagnostics" element={<DiagnosticsPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppShell>
  )
}


function ThemedApplication() {
  const mode = useThemeStore((state) => state.mode)
  const resolved = resolveTheme(mode)

  useEffect(() => {
    document.documentElement.dataset.theme = resolved
    document.documentElement.style.colorScheme = resolved
  }, [resolved])

  const themeConfig = useMemo(
    () => ({
      algorithm: resolved === 'dark' ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
      token: {
        colorPrimary: resolved === 'dark' ? '#c34813' : '#c34813',
        colorInfo: resolved === 'dark' ? '#3080ff' : '#3080ff',
        colorSuccess: '#05df72',
        colorError: resolved === 'dark' ? '#fb2c36' : '#e02b35',
        colorWarning: resolved === 'dark' ? '#ff6b00' : '#d97706',
        colorBgLayout: 'var(--color-bg)',
        colorBgContainer: 'var(--color-surface)',
        colorBgElevated: 'var(--color-surface-raised)',
        colorText: 'var(--color-ink)',
        colorTextSecondary: 'var(--color-muted)',
        colorBorder: 'var(--color-border)',
        colorBorderSecondary: 'var(--color-border)',
        borderRadius: 10,
        borderRadiusLG: 12,
        fontFamily: "Inter, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif",
        fontSize: 14,
        wireframe: false,
      },
      components: {
        Layout: {
          bodyBg: 'var(--color-bg)',
          headerBg: 'transparent',
          siderBg: 'var(--color-sidebar)',
        },
        Menu: {
          itemBorderRadius: 999,
          itemHeight: 40,
          itemMarginInline: 10,
          itemSelectedBg: 'rgba(195, 72, 19, 0.16)',
          itemSelectedColor: '#ff6b00',
          itemHoverColor: '#c34813',
        },
        Card: { borderRadiusLG: 12 },
        Table: { headerBorderRadius: 0, headerBg: 'color-mix(in srgb, var(--color-bg) 60%, transparent)' },
        Button: { borderRadius: 999, controlHeight: 36 },
        Tag: { borderRadiusSM: 999 },
      },
    }),
    [resolved],
  )

  return (
    <ConfigProvider locale={zhCN} theme={themeConfig}>
      <BrowserRouter>
        <AuthenticatedApp />
      </BrowserRouter>
    </ConfigProvider>
  )
}


export default function App() {
  const queryClient = useMemo(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 10_000, retry: 1, refetchOnWindowFocus: false },
          mutations: { retry: false },
        },
      }),
    [],
  )

  return (
    <QueryClientProvider client={queryClient}>
      <ThemedApplication />
    </QueryClientProvider>
  )
}
