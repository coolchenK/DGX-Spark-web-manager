import type { ReactNode } from 'react'
import { Alert, Button, Empty, Skeleton } from 'antd'


export function QueryState({ loading, error, empty, onRetry, children }: { loading: boolean; error: Error | null; empty?: boolean; onRetry?: () => void; children: ReactNode }) {
  if (loading) return <Skeleton active paragraph={{ rows: 5 }} />
  if (error) return <Alert type="error" showIcon message="数据加载失败" description={error.message} action={onRetry ? <Button size="small" onClick={onRetry}>重试</Button> : undefined} />
  if (empty) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无数据" />
  return <>{children}</>
}
