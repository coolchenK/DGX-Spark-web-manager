import { describe, expect, test } from 'vitest'

import { buildDownloadPatterns } from './huggingface'


describe('buildDownloadPatterns', () => {
  test('downloads the complete snapshot in all-files mode', () => {
    expect(buildDownloadPatterns('all', ['config.json'])).toEqual([])
  })

  test('uses exact selected filenames in selected-files mode', () => {
    expect(buildDownloadPatterns('selected', ['config.json', 'model.safetensors'])).toEqual([
      'config.json',
      'model.safetensors',
    ])
  })
})
