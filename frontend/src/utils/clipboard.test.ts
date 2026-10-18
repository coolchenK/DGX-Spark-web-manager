import { describe, expect, it, vi } from 'vitest'
import { copyText } from './clipboard'

describe('copyText', () => {
  it('copies the complete value using clipboard API', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, get: () => ({ writeText }) })
    await copyText('dgx_full-secret')
    expect(writeText).toHaveBeenCalledWith('dgx_full-secret')
  })

  it('supports HTTP fallback and removes the temporary secret', async () => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, get: () => undefined })
    const exec = vi.fn(() => {
      expect((document.querySelector('textarea') as HTMLTextAreaElement).value).toBe('dgx_full-secret')
      return true
    })
    Object.defineProperty(document, 'execCommand', { configurable: true, value: exec })
    await copyText('dgx_full-secret')
    expect(exec).toHaveBeenCalledWith('copy')
    expect(document.querySelector('textarea')).toBeNull()
  })
})
