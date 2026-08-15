import { useMemo, useState } from 'react'
import { Button, Empty, Input, Space } from 'antd'
import { CopyOutlined, DownloadOutlined, SearchOutlined } from '@ant-design/icons'


export function LogViewer({ value, filename = 'service.log' }: { value: string; filename?: string }) {
  const [filter, setFilter] = useState('')
  const lines = useMemo(() => value.split('\n').filter((line) => !filter || line.toLowerCase().includes(filter.toLowerCase())), [value, filter])
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
          <Button size="small" icon={<CopyOutlined />} onClick={() => navigator.clipboard.writeText(value)}>复制</Button>
          <Button size="small" icon={<DownloadOutlined />} onClick={download}>下载</Button>
        </Space>
      </div>
      {lines.length ? <pre>{lines.map((line, index) => <code key={`${index}-${line.slice(0, 20)}`}><span>{index + 1}</span>{line || ' '}</code>)}</pre> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有匹配日志" />}
    </div>
  )
}
