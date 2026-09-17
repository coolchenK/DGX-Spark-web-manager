import type { HuggingFaceModel } from '../api/types'

export type DownloadFileMode = 'all' | 'selected'

type SparkCompatibilityLevel = HuggingFaceModel['spark_compatibility']['level']
type SparkCompatibilityTagColor = 'success' | 'processing' | 'default'

const sparkCompatibilityPresentations: Record<
  SparkCompatibilityLevel,
  { label: string; color: SparkCompatibilityTagColor }
> = {
  recommended: { label: 'DGX Spark 推荐', color: 'success' },
  compatible: { label: '可部署', color: 'processing' },
  review: { label: '需评估', color: 'default' },
}


export function buildDownloadPatterns(mode: DownloadFileMode, selectedFiles: string[]) {
  return mode === 'selected' ? selectedFiles : []
}

export function getSparkCompatibilityPresentation(level: SparkCompatibilityLevel) {
  return sparkCompatibilityPresentations[level]
}
