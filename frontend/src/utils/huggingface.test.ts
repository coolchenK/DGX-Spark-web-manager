import { describe, expect, test } from 'vitest'

import { buildDownloadPatterns, getSparkCompatibilityPresentation } from './huggingface'


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

describe('getSparkCompatibilityPresentation', () => {
  test('presents recommended models as DGX Spark recommendations', () => {
    expect(getSparkCompatibilityPresentation('recommended')).toEqual({
      label: 'DGX Spark 推荐',
      color: 'success',
    })
  })

  test('presents compatible models as deployable', () => {
    expect(getSparkCompatibilityPresentation('compatible')).toEqual({
      label: '可部署',
      color: 'processing',
    })
  })

  test('presents models requiring review as needing evaluation', () => {
    expect(getSparkCompatibilityPresentation('review')).toEqual({
      label: '需评估',
      color: 'default',
    })
  })
})
