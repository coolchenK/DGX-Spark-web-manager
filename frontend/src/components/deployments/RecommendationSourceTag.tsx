import { Tag } from 'antd'

import type { RecommendationSource } from '../../api/types'


const sourcePresentation = {
  model_card: ['模型卡明确推荐', 'success'],
  local_config: ['本地模型配置', 'blue'],
  runtime_default: ['运行时默认', 'default'],
  device_rule: ['DGX Spark 资源调整', 'warning'],
  ai: ['AI 补充', 'purple'],
} as const


export function RecommendationSourceTag({ source }: { source: RecommendationSource }) {
  const [label, color] = sourcePresentation[source]
  return <Tag color={color}>{label}</Tag>
}
