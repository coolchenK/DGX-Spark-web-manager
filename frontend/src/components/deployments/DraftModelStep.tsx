import { Alert, Checkbox, Collapse, Form, InputNumber, Radio, Switch, Tag, Typography } from 'antd'

import type { DraftCandidate, ResourceEstimate, RuntimeName } from '../../api/types'
import { formatBytes } from '../../utils/format'


interface DraftModelStepProps {
  candidates: DraftCandidate[]
  selectedId?: string
  selectedMethod?: string
  embeddedMtpAvailable?: boolean
  runtime: Exclude<RuntimeName, 'llama_cpp'>
  advanced: boolean
  resourceEstimate?: Partial<ResourceEstimate>
  resourceUnverified?: boolean
  validationError?: string | null
  onAdvancedChange: (value: boolean) => void
  onSelect: (candidate?: DraftCandidate | 'embedded-mtp') => void
}

const statusPresentation = {
  compatible: ['兼容', 'success'],
  review: ['待确认', 'warning'],
  incompatible: ['不兼容', 'error'],
} as const


export function DraftModelStep({
  candidates,
  selectedId,
  selectedMethod,
  embeddedMtpAvailable = false,
  runtime,
  advanced,
  resourceEstimate,
  resourceUnverified = false,
  validationError,
  onAdvancedChange,
  onSelect,
}: DraftModelStepProps) {
  const visibleCandidates = advanced
    ? candidates
    : candidates.filter((candidate) => candidate.status === 'compatible')
  const selectedCandidate = candidates.find((candidate) => candidate.model_id === selectedId)
  const embeddedMtpSelected = !selectedId && selectedMethod === 'mtp'
  const resourceDecision = resourceEstimate?.decision

  return (
    <section className="deployment-step" aria-labelledby="draft-model-heading">
      <div className="deployment-step-heading deployment-step-heading-actions">
        <div>
          <Typography.Title level={4} id="draft-model-heading">Draft Model</Typography.Title>
          <Typography.Text type="secondary">附带兼容的本地 Draft Model 以启用推测解码。</Typography.Text>
        </div>
        <label className="draft-advanced-toggle">
          <span>显示待确认及不兼容模型</span>
          <Switch checked={advanced} onChange={onAdvancedChange} aria-label="显示待确认及不兼容模型" />
        </label>
      </div>
      {validationError && <Alert type="error" showIcon message={validationError} />}
      <Radio.Group
        className="draft-candidate-list"
        value={embeddedMtpSelected ? 'embedded-mtp' : selectedId ?? ''}
        onChange={(event) => {
          if (event.target.value === 'embedded-mtp') {
            onSelect('embedded-mtp')
            return
          }
          const candidate = candidates.find((item) => item.model_id === event.target.value)
          onSelect(candidate)
        }}
      >
        <div className="draft-candidate draft-candidate-none">
          <Radio value="">不使用 Draft Model</Radio>
        </div>
        {embeddedMtpAvailable && (
          <div className="draft-candidate">
            <Radio value="embedded-mtp">
              <span className="draft-candidate-label">
                <strong className="draft-candidate-name">内置 MTP Head</strong>
                <Tag color="success">兼容</Tag>
                <Tag>mtp</Tag>
              </span>
              <span className="draft-candidate-details">
                使用当前模型自带的 MTP 权重，无需外挂 Draft Model
              </span>
            </Radio>
          </div>
        )}
        {visibleCandidates.map((candidate) => {
          const [label, color] = statusPresentation[candidate.status]
          return (
            <div className="draft-candidate" key={candidate.model_id}>
              <Radio value={candidate.model_id} disabled={candidate.status === 'incompatible'}>
                <span className="draft-candidate-label">
                  <strong className="draft-candidate-name">{candidate.name}</strong>
                  <Tag color={color}>{label}</Tag>
                  {candidate.method && <Tag>{candidate.method}</Tag>}
                </span>
                <span className="draft-candidate-details">
                  {candidate.reasons.join('；')}
                  {' · '}{formatBytes(candidate.size_bytes)}
                </span>
              </Radio>
            </div>
          )
        })}
      </Radio.Group>
      {selectedCandidate?.status === 'review' && (
        <Alert
          type="warning"
          showIcon
          message="该 Draft Model 需要人工确认"
          description={selectedCandidate.reasons.join('；')}
          action={(
            <Form.Item
              name={['speculative', 'manual_review_acknowledged']}
              valuePropName="checked"
              noStyle
            >
              <Checkbox>我已核对该 Draft Model 的兼容性风险</Checkbox>
            </Form.Item>
          )}
        />
      )}
      {(selectedCandidate || embeddedMtpSelected) && (
        <Collapse
          defaultActiveKey={['speculative-tuning']}
          items={[{
            key: 'speculative-tuning',
            label: '推测解码高级参数',
            children: (
              runtime === 'vllm' ? (
                <Form.Item
                  name={['speculative', 'num_speculative_tokens']}
                  label="每轮推测 Token"
                  rules={[{ required: false, type: 'number', min: 1, max: 64, message: '每轮推测 Token 必须在 1-64 之间' }]}
                >
                  <InputNumber />
                </Form.Item>
              ) : selectedCandidate?.method === 'dflash' ? (
                <Form.Item
                  name={['speculative', 'num_draft_tokens']}
                  label="DFlash Block Size"
                  rules={[{ required: true, type: 'number', min: 1, max: 256, message: 'DFlash Block Size 必须在 1-256 之间' }]}
                >
                  <InputNumber />
                </Form.Item>
              ) : (
                <div className="form-grid">
                  <Form.Item
                    name={['speculative', 'num_steps']}
                    label="推测步数"
                    rules={[{ required: false, type: 'number', min: 1, max: 32, message: '推测步数必须在 1-32 之间' }]}
                  >
                    <InputNumber />
                  </Form.Item>
                  <Form.Item
                    name={['speculative', 'eagle_top_k']}
                    label="EAGLE Top K"
                    rules={[{ required: false, type: 'number', min: 1, max: 32, message: 'EAGLE Top K 必须在 1-32 之间' }]}
                  >
                    <InputNumber />
                  </Form.Item>
                  <Form.Item
                    name={['speculative', 'num_draft_tokens']}
                    label="Draft Token 数"
                    rules={[{ required: false, type: 'number', min: 1, max: 256, message: 'Draft Token 数必须在 1-256 之间' }]}
                  >
                    <InputNumber />
                  </Form.Item>
                </div>
              )
            ),
          }]}
        />
      )}
      {resourceUnverified && (
        <Alert
          type="warning"
          showIcon
          message="Draft Model 资源未验证"
          description="候选模型缺少完整总资源估算，部署前必须确认当前统一内存余量。"
          action={(
            <Form.Item name="resource_warning_acknowledged" valuePropName="checked" noStyle>
              <Checkbox>我了解资源不足可能导致部署失败</Checkbox>
            </Form.Item>
          )}
        />
      )}
      {resourceDecision === 'warning' && !resourceUnverified && (
        <Alert
          type="warning"
          showIcon
          message="当前可用统一内存可能不足"
          description={resourceEstimate?.reasons?.join('；')}
          action={(
            <Form.Item name="resource_warning_acknowledged" valuePropName="checked" noStyle>
              <Checkbox>我了解资源不足可能导致部署失败</Checkbox>
            </Form.Item>
          )}
        />
      )}
      {resourceDecision === 'blocked' && (
        <Alert
          type="error"
          showIcon
          message="资源估算超过 DGX Spark 硬上限"
          description={resourceEstimate?.reasons?.join('；')}
        />
      )}
    </section>
  )
}
