import type { ReactNode } from 'react'
import { Space, Typography } from 'antd'


export function PageHeader({ title, description, extra }: { title: string; description?: string; extra?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        <Typography.Title level={1}>{title}</Typography.Title>
        {description && <Typography.Paragraph type="secondary">{description}</Typography.Paragraph>}
      </div>
      {extra && <Space wrap>{extra}</Space>}
    </header>
  )
}
