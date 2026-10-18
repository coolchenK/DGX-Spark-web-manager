import { useMemo, useState } from 'react'
import { Button, Empty, Input, Space, Typography } from 'antd'
import { CopyOutlined, DownloadOutlined, SearchOutlined } from '@ant-design/icons'
import { formatLogLine } from '../utils/format'


export function LogViewer({ value, filename = 'service.log' }: { value: string; filename?: string }) {
  const [filter, setFilter] = useState('')
  const lines = useMemo(() => value.split('\n').map(formatLogLine).filter((line) => !filter || line.toLowerCase().includes(filter.toLowerCase())), [value, filter])
  const download = () => {
    const url = URL.createObjectURL(new Blob([value], { type: 'text/plain;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <Input size="small" allowClear prefix={<SearchOutlined />} placeholder="筛选日志" value={filter} onChange={(event) => setFilter(event.target.value)} />
        <Space>
          <Button size="small" icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(value)}>复制原始日志</Button>
          <Button size="small" icon={<DownloadOutlined />} onClick={download}>下载原始日志</Button>
        </Space>
      </div>
      <Typography.Text type="secondary">行首时间：本地时区（{Intl.DateTimeFormat().resolvedOptions().timeZone}）；日志正文中的时间保持原样</Typography.Text>
      {lines.length ? <pre>{lines.map((line, index) => <code key={`${index}-${line.slice(0, 20)}`}><span>{index + 1}</span>{line || ' '}</code>)}</pre> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有匹配日志" />}
    </div>
  )
}
