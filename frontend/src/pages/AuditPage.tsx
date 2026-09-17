import { useQuery } from '@tanstack/react-query'
import { Input, Tag, Typography } from 'antd'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import type { AuditEvent } from '../api/types'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { StatusBadge } from '../components/StatusBadge'
import { formatDate } from '../utils/format'


export function AuditPage() {
  const [filter, setFilter] = useState('')
  const events = useQuery({ queryKey: ['audit'], queryFn: () => api.get<AuditEvent[]>('/api/audit?limit=500'), refetchInterval: 10_000 })
  const data = useMemo(() => (events.data ?? []).filter((item) => `${item.action} ${item.actor} ${item.resource_type}`.toLowerCase().includes(filter.toLowerCase())), [events.data, filter])
  return (
    <div className="page-stack"><PageHeader title="日志与审计" description="管理员、网关和自动化操作的不可见密钥脱敏记录" /><div className="filter-bar"><Input.Search allowClear placeholder="筛选动作、用户或资源" value={filter} onChange={(event) => setFilter(event.target.value)} /></div><QueryState loading={events.isLoading} error={events.error} empty={!data.length}><ResponsiveDataView data={data} rowKey="id" columns={[{ title: '时间', dataIndex: 'created_at', width: 170, render: formatDate }, { title: '动作', dataIndex: 'action', render: (value) => <Typography.Text code>{value}</Typography.Text> }, { title: '资源', render: (_, item) => <div className="primary-cell"><strong>{item.resource_type}</strong><small>{item.resource_id ?? 'system'}</small></div> }, { title: '用户', dataIndex: 'actor', width: 110 }, { title: '来源 IP', dataIndex: 'source_ip', width: 130, render: (value) => value ?? '本机' }, { title: '结果', dataIndex: 'outcome', width: 100, render: (value) => <StatusBadge status={value} /> }] } renderMobile={(item) => <div className="mobile-record"><div><Typography.Text code>{item.action}</Typography.Text> <StatusBadge status={item.outcome} /></div><Typography.Text type="secondary">{item.resource_type} · {item.resource_id ?? 'system'}</Typography.Text><Tag>{item.actor}</Tag><small>{formatDate(item.created_at)}</small></div>} /></QueryState></div>
  )
}
