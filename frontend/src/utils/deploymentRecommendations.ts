import type { DeploymentRecommendation } from '../api/types'
import type { DeploymentFormValues, GenerationDefaults } from './deployments'


const DEPLOYMENT_FIELDS = new Set<keyof DeploymentFormValues>([
  'context_length',
  'memory_fraction',
  'max_concurrency',
  'max_batched_tokens',
  'quantization',
])

const GENERATION_FIELDS = new Set<keyof GenerationDefaults>([
  'temperature',
  'top_p',
  'top_k',
  'min_p',
  'repetition_penalty',
  'presence_penalty',
  'frequency_penalty',
  'max_tokens',
  'stop',
])


function isNestedRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}


export function flattenChangedFields(
  changed: Record<string, unknown>,
  prefix = '',
): string[] {
  const paths: string[] = []
  for (const [key, value] of Object.entries(changed)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (isNestedRecord(value)) {
      paths.push(...flattenChangedFields(value, path))
    } else {
      paths.push(path)
    }
  }
  return paths.sort()
}


export function valuesFromRecommendation(
  recommendation: DeploymentRecommendation,
  editedFields: ReadonlySet<string>,
  force = false,
): Partial<DeploymentFormValues> {
  const values: Partial<DeploymentFormValues> = {}
  const writableValues = values as Record<string, unknown>

  for (const [field, recommended] of Object.entries(recommendation.fields)) {
    if (!DEPLOYMENT_FIELDS.has(field as keyof DeploymentFormValues)) continue
    if (!force && editedFields.has(field)) continue
    writableValues[field] = recommended.value
  }

  const generationDefaults: Partial<GenerationDefaults> = {}
  const writableGeneration = generationDefaults as Record<string, unknown>
  for (const [field, recommended] of Object.entries(recommendation.generation_defaults)) {
    if (!GENERATION_FIELDS.has(field as keyof GenerationDefaults)) continue
    if (!force && editedFields.has(`generation_defaults.${field}`)) continue
    writableGeneration[field] = recommended.value
  }
  if (Object.keys(generationDefaults).length) {
    values.generation_defaults = generationDefaults
  }
  return values
}
