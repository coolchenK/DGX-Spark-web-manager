import { describe, expect, it } from 'vitest'
import { formatDate, formatLogLine } from './format'

describe('UTC API timestamps displayed in the browser timezone', () => {
  const expected = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short', timeStyle: 'medium' })
    .format(new Date(Date.UTC(2026, 9, 11, 5, 53, 16)))

  it.each([
    '2026-10-11T05:53:16.049096',
    '2026-10-11 05:53:16.049096',
    '2026-10-11T05:53:16',
    '2026-10-11T05:53:16.049096Z',
    '2026-10-11T13:53:16+08:00',
    '2026-10-10T22:53:16-07:00',
  ])('formats %s as the same instant without a double offset', (timestamp) => {
    expect(formatDate(timestamp)).toBe(expected)
  })

  it('handles missing or malformed timestamps without crashing the page', () => {
    expect(formatDate(null)).toBe('从未')
    expect(formatDate(undefined)).toBe('从未')
    expect(formatDate('')).toBe('从未')
    expect(formatDate('invalid timestamp')).toBe('invalid timestamp')
  })

  it('converts the Docker timestamp and preserves the entire application message', () => {
    const body = ' (APIServer pid=1) INFO 10-11 05:53:16 ready; embedded 2026-10-11T05:53:16Z'
    expect(formatLogLine('2026-10-11T05:53:16.049096123Z' + body)).toBe(expected + body)
    expect(formatLogLine('2026-10-11T13:53:16+08:00 ready')).toBe(expected + ' ready')
  })

  it('leaves untimestamped and invalid log lines unchanged', () => {
    for (const line of ['', '  traceback line', 'Downloaded 2 GiB', '2026-99-11T05:53:16Z invalid']) {
      expect(formatLogLine(line)).toBe(line)
    }
  })
})
