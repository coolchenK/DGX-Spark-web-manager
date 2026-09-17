import type { ReactNode } from 'react'
import { Grid, List, Table, type TableProps } from 'antd'


interface Props<T extends object> {
  data: T[]
  columns: TableProps<T>['columns']
  rowKey: string | ((record: T) => string)
  renderMobile: (record: T) => ReactNode
  loading?: boolean
}


export function ResponsiveDataView<T extends object>({ data, columns, rowKey, renderMobile, loading }: Props<T>) {
  const screens = Grid.useBreakpoint()
  if (screens.md) {
    return <Table<T> size="small" rowKey={rowKey} columns={columns} dataSource={data} loading={loading} scroll={{ x: 760 }} />
  }
  const keyFor = (record: T) => typeof rowKey === 'function' ? rowKey(record) : String(record[rowKey as keyof T])
  return <List loading={loading} dataSource={data} renderItem={(item) => <List.Item key={keyFor(item)}>{renderMobile(item)}</List.Item>} />
}
