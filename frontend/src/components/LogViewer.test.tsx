import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { LogViewer } from './LogViewer'

describe('LogViewer timezone display', () => {
  it('filters displayed local timestamps while copying the original log', () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    const raw = '2026-10-11T05:53:16.049096123Z ready\nplain line'
    const local = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short', timeStyle: 'medium' })
      .format(new Date(Date.UTC(2026, 9, 11, 5, 53, 16)))
    const { container } = render(<LogViewer value={raw} />)

    expect(container.querySelector('pre')).toHaveTextContent(local + ' ready')
    expect(screen.getByText(/行首时间：本地时区/)).toHaveTextContent(Intl.DateTimeFormat().resolvedOptions().timeZone)
    fireEvent.change(screen.getByPlaceholderText('筛选日志'), { target: { value: local } })
    expect(container.querySelectorAll('code')).toHaveLength(1)
    expect(container.querySelector('pre')).not.toHaveTextContent('plain line')
    fireEvent.click(screen.getByRole('button', { name: /复制原始日志/ }))
    expect(writeText).toHaveBeenCalledWith(raw)
  })
})
