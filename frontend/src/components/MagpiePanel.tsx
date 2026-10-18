import { ExportOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Space, Spin, Tag, Typography } from 'antd'
import { useEffect, useState } from 'react'

import { api } from '../api/client'

type MagpieStatus = { status: 'ok' | 'unavailable'; detail?: string }

export function MagpiePanel({ view }: { view: 'gateway' | 'providers' }) {
  const [revision, setRevision] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [frameError, setFrameError] = useState<string | null>(null)
  const status = useQuery({
    queryKey: ['magpie-status'],
    queryFn: () => api.get<MagpieStatus>('/api/magpie/status'),
    retry: false,
    staleTime: 30_000,
  })
  const ready = status.data?.status === 'ok' && !status.error
  const error = status.error?.message
    ?? (status.data?.status === 'unavailable' ? status.data.detail || 'Magpie 服务暂时不可用，请稍后重试。' : null)
    ?? frameError
  const src = `/magpie/?view=${view}`

  useEffect(() => {
    if (!ready || loaded || frameError) return
    const timeout = window.setTimeout(() => setFrameError('Magpie 页面加载超时，请重试或在新窗口打开。'), 30_000)
    return () => window.clearTimeout(timeout)
  }, [ready, loaded, frameError, revision])

  const reload = () => {
    setLoaded(false)
    setFrameError(null)
    setRevision((value) => value + 1)
    void status.refetch()
  }

  return (
    <section className="magpie-panel" aria-label={view === 'gateway' ? 'Magpie 网关管理' : 'Magpie 服务商管理'}>
      <div className="magpie-toolbar">
        <Space wrap>
          <strong>Magpie</strong>
          <Tag color={error ? 'error' : loaded ? 'success' : 'processing'}>{error ? '连接异常' : loaded ? '已连接' : '连接中'}</Tag>
        </Space>
        <Space wrap>
          <Button icon={<ReloadOutlined aria-hidden />} onClick={reload}>重新加载</Button>
          <Button icon={<ExportOutlined aria-hidden />} href={src} target="_blank" rel="noopener noreferrer">在新窗口打开</Button>
        </Space>
      </div>
      <div className="magpie-endpoint">
        <Typography.Text type="secondary">API Base URL</Typography.Text>
        <Typography.Text code copyable>{`${window.location.origin}/v1`}</Typography.Text>
      </div>
      <div className="magpie-frame-wrap" aria-busy={!loaded && !error}>
        {error ? (
          <div className="magpie-frame-state">
            <Alert type="error" showIcon message="Magpie 加载失败" description={error} action={<Button onClick={reload}>重试</Button>} />
          </div>
        ) : (
          <>
            {!loaded && <div className="magpie-frame-state" role="status"><Spin /><span>正在加载 Magpie…</span></div>}
            {ready && (
              <iframe
                key={`${view}-${revision}`}
                className={`magpie-frame${loaded ? '' : ' magpie-frame-loading'}`}
                title={view === 'gateway' ? 'Magpie API 网关' : 'Magpie 在线 AI 服务'}
                src={src}
                onLoad={() => setLoaded(true)}
                referrerPolicy="same-origin"
              />
            )}
          </>
        )}
      </div>
    </section>
  )
}
