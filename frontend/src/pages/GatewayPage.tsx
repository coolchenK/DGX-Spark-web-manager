import { DeleteOutlined, KeyOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Collapse, Popconfirm, Typography, message } from 'antd'

import { api } from '../api/client'
import type { ApiKeyRecord } from '../api/types'
import { MagpiePanel } from '../components/MagpiePanel'
import { PageHeader } from '../components/PageHeader'
import { QueryState } from '../components/QueryState'
import { ResponsiveDataView } from '../components/ResponsiveDataView'
import { formatDate } from '../utils/format'

function ExistingAccessKeys() {
  const queryClient = useQueryClient()
  const keys = useQuery({ queryKey: ['api-keys'], queryFn: () => api.get<ApiKeyRecord[]>('/api/keys') })
  const revoke = useMutation({
    mutationFn: (id: string) => api.delete(`/api/keys/${id}`),
    onSuccess: () => {
      message.success('访问密钥已吊销')
      void queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => message.error(error instanceof Error ? error.message : '吊销失败，请重试'),
  })
  const revokeButton = (item: ApiKeyRecord) => !item.revoked_at && (
    <Popconfirm
      title={`吊销 ${item.name}`}
      description="使用该密钥的客户端将立即无法调用网关。"
      onConfirm={() => revoke.mutate(item.id)}
    >
      <Button danger size="small" icon={<DeleteOutlined aria-hidden />} loading={revoke.isPending && revoke.variables === item.id}>吊销</Button>
    </Popconfirm>
  )

  return (
    <div className="magpie-existing-keys">
      <Typography.Paragraph type="secondary">现有客户端可继续使用这些密钥。新增密钥请在上方 Magpie 中管理。</Typography.Paragraph>
      <QueryState loading={keys.isLoading} error={keys.error} empty={!keys.data?.length} onRetry={() => void keys.refetch()}>
        <ResponsiveDataView
          data={keys.data ?? []}
          rowKey="id"
          columns={[
            { title: '名称', dataIndex: 'name' },
            { title: '前缀', dataIndex: 'prefix', render: (value) => <Typography.Text code>{value}…</Typography.Text> },
            { title: '创建时间', dataIndex: 'created_at', render: formatDate },
            { title: '最后使用', dataIndex: 'last_used_at', render: formatDate },
            { title: '状态', dataIndex: 'revoked_at', render: (value) => value ? '已吊销' : '有效' },
            { title: '操作', render: (_, item) => revokeButton(item) },
          ]}
          renderMobile={(item) => (
            <div className="mobile-record">
              <div className="primary-cell"><strong>{item.name}</strong><Typography.Text code>{item.prefix}…</Typography.Text></div>
              <dl><div><dt>状态</dt><dd>{item.revoked_at ? '已吊销' : '有效'}</dd></div><div><dt>最后使用</dt><dd>{formatDate(item.last_used_at)}</dd></div></dl>
              {revokeButton(item)}
            </div>
          )}
        />
      </QueryState>
    </div>
  )
}

export function GatewayPage() {
  return (
    <div className="page-stack">
      <PageHeader title="API 网关与 AI 服务" description="通过 Magpie 统一管理 API 密钥、服务商、模型路由和调用记录" />
      <MagpiePanel view="gateway" />
      <Collapse className="magpie-compatibility" items={[{
        key: 'existing-keys',
        label: <span><KeyOutlined aria-hidden /> 现有访问密钥</span>,
        children: <ExistingAccessKeys />,
      }]} />
    </div>
  )
}
