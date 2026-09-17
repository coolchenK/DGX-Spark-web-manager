import type { ReactNode } from 'react'
import { Progress, Tooltip } from 'antd'


interface Metric {
  label: string
  value: string
  detail?: string
  percent?: number | null
  icon: ReactNode
}


export function MetricStrip({ metrics }: { metrics: Metric[] }) {
  return (
    <section className="metric-strip" aria-label="资源指标">
      {metrics.map((metric) => (
        <div className="metric-item" key={metric.label}>
          <span className="metric-icon" aria-hidden="true">{metric.icon}</span>
          <div className="metric-copy">
            <span className="metric-label">{metric.label}</span>
            <strong>{metric.value}</strong>
            {metric.detail && <Tooltip title={metric.detail}><small>{metric.detail}</small></Tooltip>}
          </div>
          {metric.percent != null && (
            <Progress percent={Math.round(metric.percent)} size="small" showInfo={false} strokeColor="var(--color-primary)" />
          )}
        </div>
      ))}
    </section>
  )
}
