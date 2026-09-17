import { Empty, Typography } from 'antd'


interface ExecutionStepResult {
  id?: string
  status?: string
  output?: string
  error?: string | null
  exit_code?: number | null
}

export function OpsExecutionOutput({ result }: { result: Record<string, unknown> }) {
  const steps = Array.isArray(result.steps) ? result.steps as ExecutionStepResult[] : []
  if (!steps.length) return null

  return (
    <section className="ops-execution" aria-label="执行输出">
      <Typography.Text strong>执行输出</Typography.Text>
      {steps.map((step, index) => {
        const output = [step.output, step.error].filter(Boolean).join('\n')
        return (
          <div className="ops-execution-step" key={step.id ?? index}>
            <div className="ops-execution-meta">
              <span>步骤 {index + 1}</span>
              <span>{step.status ?? 'unknown'}{step.exit_code == null ? '' : ` · exit ${step.exit_code}`}</span>
            </div>
            {output
              ? <pre><code>{output}</code></pre>
              : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="无命令输出" />}
          </div>
        )
      })}
    </section>
  )
}
