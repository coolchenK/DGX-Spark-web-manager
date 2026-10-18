import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Card, Col, Empty, Form, Image, Input, InputNumber, Modal, Popconfirm, Progress, Row, Select, Space, Tabs, Tag, Typography, Upload } from 'antd'
import { useState } from 'react'
import { api } from '../api/client'
import type { Deployment, TaskRecord } from '../api/types'
import { PageHeader } from '../components/PageHeader'

interface Parameters {
  prompt: string; steps: number; resolution: number; aspect_ratio: string
  width: number; height: number; seed: number | null
  operation?: 'generate' | 'refine' | 'texture' | 'upscale'
  sr_engine?: 'realesrgan' | 'supir'
}
interface Enhancement {
  operation: 'upscale'; prompt: string; sr_engine: 'realesrgan' | 'supir'; sr_resolution: number
  sr_steps: number; sr_cfg: number; sr_control: number; tile_size: number; tile_overlap: number; seed: number | null
}
interface ImageTask extends TaskRecord { request: Parameters; deployment_id: string; source_task_id?: string }
interface Reference { name: string; data: string }
const defaults: Parameters = { prompt: '', steps: 6, resolution: 1024, aspect_ratio: 'auto', width: 1024, height: 1024, seed: null }
const active = (task: TaskRecord) => ['queued', 'running'].includes(task.status)
const labels: Record<string, string> = { queued: '排队中', running: '生成中', succeeded: '已完成', failed: '失败', cancelled: '已取消', paused: '已暂停' }
const stages: Record<string, string> = { generate: '基础生图', refine: '2K 精修', texture: '旧版纹理增强', upscale: '超分放大' }

export function ImagesPage() {
  const [form] = Form.useForm<Parameters>()
  const [enhanceForm] = Form.useForm<Enhancement>()
  const [source, setSource] = useState<ImageTask>()
  const engine = Form.useWatch('sr_engine', enhanceForm) ?? 'realesrgan'
  const outputResolution = Form.useWatch('sr_resolution', enhanceForm) ?? 4096
  const [mode, setMode] = useState('text')
  const [references, setReferences] = useState<Reference[]>([])
  const [uploadError, setUploadError] = useState('')
  const [selected, setSelected] = useState<string>()
  const [focused, setFocused] = useState<string>()
  const [reusedEdit, setReusedEdit] = useState(false)
  const ratio = Form.useWatch('aspect_ratio', form)
  const cache = useQueryClient()
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.get<Deployment[]>('/api/deployments'), refetchInterval: 5000 })
  const history = useQuery({ queryKey: ['images'], queryFn: () => api.get<ImageTask[]>('/api/images'), refetchInterval: 2000 })
  const models = (deployments.data ?? []).filter(d => d.runtime === 'viggle')
  const model = models.find(d => d.id === selected) ?? models[0]
  const items = history.data ?? []
  const current = items.find(t => t.id === focused) ?? items[0]
  const refresh = () => cache.invalidateQueries({ queryKey: ['images'] })
  const generate = useMutation({
    mutationFn: (values: Parameters) => api.post<ImageTask>('/api/images', { ...values, seed: values.seed ?? null, deployment_id: model?.id, images: mode === 'edit' ? references.map(r => r.data) : [] }),
    onSuccess: task => { setFocused(task.id); void refresh() },
  })
  const enhance = useMutation({
    mutationFn: (values: Enhancement) => api.post<ImageTask>('/api/images', { ...values, seed: values.seed ?? null, deployment_id: model?.id, source_task_id: source?.id }),
    onSuccess: task => { setSource(undefined); setFocused(task.id); void refresh() },
  })
  const action = useMutation({
    mutationFn: ({ id, operation }: { id: string; operation: 'cancel' | 'delete' | 'resume' }) => operation === 'delete' ? api.delete(`/api/images/${id}`) : operation === 'resume' ? api.post(`/api/tasks/${id}/resume`) : api.delete(`/api/tasks/${id}`),
    onSuccess: refresh,
  })
  const start = useMutation({ mutationFn: () => api.post(`/api/deployments/${model?.id}/start`), onSuccess: () => cache.invalidateQueries({ queryKey: ['deployments'] }) })
  const error = generate.error ?? action.error ?? start.error ?? history.error ?? deployments.error
  const result = current?.status === 'succeeded' ? current.result : undefined

  async function addReference(file: File) {
    setUploadError('')
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 16 * 1024 * 1024) {
      setUploadError('请选择不超过 16 MiB 的 PNG、JPEG 或 WebP 图片'); return
    }
    const reader = new FileReader()
    reader.onload = () => setReferences(previous => previous.length < 3 ? [...previous, { name: file.name, data: String(reader.result) }] : previous)
    reader.onerror = () => setUploadError('无法读取图片，请重新选择')
    reader.readAsDataURL(file)
  }
  function reuse(task: ImageTask) {
    if (task.request.operation === 'upscale') {
      const parent = items.find(t => t.id === task.source_task_id)
      if (parent) {
        prepareEnhancement(parent, task.request.sr_engine ?? 'realesrgan')
        enhanceForm.setFieldsValue({ ...task.request, operation: task.request.operation, seed: typeof task.result.seed === 'number' ? task.result.seed : task.request.seed })
        return
      }
    }
    form.setFieldsValue({ ...defaults, ...task.request, seed: typeof task.result.seed === 'number' ? task.result.seed : task.request.seed })
    const edit = Number(task.result.reference_count ?? 0) > 0
    setMode(edit ? 'edit' : 'text'); setReusedEdit(edit); setReferences([])
  }
  function prepareEnhancement(task: ImageTask, next: Enhancement['sr_engine']) {
    enhance.reset()
    const visited = new Set<string>()
    while (task.source_task_id && !visited.has(task.id)) {
      visited.add(task.id)
      const original = items.find(t => t.id === task.source_task_id && t.status === 'succeeded')
      if (!original) break
      task = original
    }
    setSource(task)
    enhanceForm.setFieldsValue({ operation: 'upscale', sr_engine: next, prompt: task.request.prompt, sr_resolution: Math.max(Number(task.result.width), Number(task.result.height)) < 1024 ? 2048 : 4096, sr_steps: 30, sr_cfg: 4, sr_control: 1, tile_size: 1024, tile_overlap: 256, seed: typeof task.result.seed === 'number' ? task.result.seed : null })
  }
  const sourceWidth = Number(source?.result.width ?? 1024)
  const sourceHeight = Number(source?.result.height ?? 1024)
  const factor = outputResolution / Math.max(sourceWidth, sourceHeight)
  const targetSize = [sourceWidth, sourceHeight].map(n => Math.round(n * factor / 32) * 32)
  const parent = current?.source_task_id ? items.find(t => t.id === current.source_task_id) : undefined

  return <div className="page-stack">
    <PageHeader title="图片生成" description="Qwen-Image-2.1 · Viggle Turbo — 文生图与参考图编辑" />
    <Alert type="info" showIcon message="选中满意的图片 → 快速超分 / 细节重建" description="Real-ESRGAN 提升清晰度；SUPIR-F 补充生成式细节。优先从最初原图处理，手动开始，结果另存，原图保留。" />
    {error && <Alert type="error" showIcon message={error.message} />}
    <Row gutter={[20, 20]}>
      <Col xs={24} xl={10}>
        <Card title="创作设置">
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Select aria-label="图片模型" style={{ width: '100%' }} value={model?.id} onChange={setSelected} placeholder="尚未配置图片模型" options={models.map(d => ({ value: d.id, label: `${d.name} · ${d.status === 'running' ? '在线' : d.status}` }))} />
            {model && model.status !== 'running' && <Alert type="info" message="图片模型尚未运行" description="启动后即可生成；内存不足时请先停止其他大模型。" action={<Button onClick={() => start.mutate()} loading={start.isPending}>启动模型</Button>} />}
            <Tabs activeKey={mode} onChange={value => { setMode(value); setReusedEdit(false); if (value === 'edit' && form.getFieldValue('resolution') === 2048) form.setFieldValue('resolution', 1024) }} items={[{ key: 'text', label: '文生图' }, { key: 'edit', label: '图片编辑' }]} />
            {mode === 'edit' && <>
              <Typography.Text type="secondary">上传 1–3 张参考图，按顺序在指令中称为图 1、图 2、图 3。每张不超过 16 MiB。</Typography.Text>
              {reusedEdit && <Alert type="info" message="已载入参数，请重新选择参考图" />}
              <Upload accept="image/png,image/jpeg,image/webp" showUploadList={false} beforeUpload={file => { void addReference(file); return false }} disabled={references.length >= 3}>
                <Button disabled={references.length >= 3}>添加参考图片（{references.length}/3）</Button>
              </Upload>
              {uploadError && <Alert type="error" message={uploadError} />}
              <Space wrap>{references.map((ref, i) => <div key={`${i}-${ref.name}`} style={{ width: 104 }}>
                <Image src={ref.data} alt={`参考图 ${i + 1}: ${ref.name}`} width={96} height={96} style={{ objectFit: 'cover' }} />
                <div>图 {i + 1} <Button size="small" aria-label={`移除参考图 ${i + 1}`} onClick={() => setReferences(v => v.filter((_, n) => n !== i))}>移除</Button></div>
                {i > 0 && <Button size="small" onClick={() => setReferences(v => { const next = [...v]; [next[i - 1], next[i]] = [next[i], next[i - 1]]; return next })}>前移</Button>}
              </div>)}</Space>
            </>}
            <Form form={form} initialValues={defaults} layout="vertical" onFinish={values => generate.mutate(values)}>
              <Form.Item name="prompt" label={mode === 'edit' ? '编辑指令' : '提示词'} rules={[{ required: true, whitespace: true, message: '请输入提示词或编辑指令' }]}>
                <Input.TextArea rows={5} maxLength={4000} showCount placeholder={mode === 'edit' ? '保留图 1 的主体，将背景改成夕阳下的海边。' : '描述画面主体、场景、风格、光线，以及需要出现的文字。'} />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}><Form.Item name="steps" label="采样步数"><Select options={[{ value: 4, label: '4 步 · 快速' }, { value: 6, label: '6 步 · 推荐' }, { value: 8, label: '8 步 · 密集文字' }]} /></Form.Item></Col>
                <Col span={12}><Form.Item name="resolution" label="分辨率档位"><Select options={(mode === 'edit' ? [1024, 1536] : [1024, 2048]).map(n => ({ value: n, label: `${n} × ${n} 等面积` }))} /></Form.Item></Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}><Form.Item name="aspect_ratio" label="画面比例"><Select options={['auto', '1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', 'custom'].map(v => ({ value: v, label: v === 'auto' ? '自动' : v === 'custom' ? '自定义尺寸' : v }))} /></Form.Item></Col>
                <Col span={12}><Form.Item name="seed" label="随机种子" extra="留空随机；复用种子可重现结果"><InputNumber min={0} max={4294967295} precision={0} placeholder="随机" style={{ width: '100%' }} /></Form.Item></Col>
              </Row>
              {ratio === 'custom' && <Row gutter={12}>
                <Col span={12}><Form.Item name="width" label="宽度（32 的倍数）"><InputNumber min={512} max={2816} step={32} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={12}><Form.Item name="height" label="高度（32 的倍数）"><InputNumber min={512} max={2816} step={32} style={{ width: '100%' }} /></Form.Item></Col>
              </Row>}
              <Typography.Paragraph type="secondary">自动比例：文生图为正方形，编辑跟随最后一张参考图（9:16 至 16:9）。默认采用作者推荐的 6 步调度，CFG 与 LoRA 强度固定为 1。</Typography.Paragraph>
              <Typography.Paragraph type="secondary">2K 输出实测峰值约 58 GiB；建议先从「部署实例」停止其他大模型。</Typography.Paragraph>
              <Button type="primary" size="large" htmlType="submit" block loading={generate.isPending} disabled={model?.status !== 'running' || (mode === 'edit' && references.length === 0)}>生成图片</Button>
            </Form>
          </Space>
        </Card>
      </Col>
      <Col xs={24} xl={14}>
        <Card title="生成结果" extra={current && <Tag>{labels[current.status] ?? current.status}</Tag>}>
          {!current ? <Empty description="输入提示词，开始创作第一张图片" /> : <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {active(current) && <><Progress percent={Math.round(current.progress)} status="active" /><Typography.Text>{current.log.trim().split('\n').at(-1) ?? '等待开始'}</Typography.Text><Button onClick={() => action.mutate({ id: current.id, operation: 'cancel' })} loading={action.isPending}>取消生成</Button></>}
            {current.error && <Alert type="error" showIcon message="生成失败" description={current.error} />}
            {result && <>
              <Image src={String(result.image_url)} alt={String(result.prompt)} style={{ maxHeight: '65vh', objectFit: 'contain', width: '100%' }} />
              <Typography.Text type="secondary">{String(result.width)} × {String(result.height)} · {result.sr_engine === 'realesrgan' ? 'Real-ESRGAN' : `${String(result.steps)} 步`} · {String(result.seconds)} 秒 · 种子 {String(result.seed)}</Typography.Text>
              <Tag>{result.sr_engine === 'supir' ? 'SUPIR-F 细节重建' : result.sr_engine === 'realesrgan' ? 'Real-ESRGAN 快速超分' : stages[String(result.operation ?? current.request.operation ?? 'generate')]}</Tag>
              {parent && <Space><Typography.Text type="secondary">源图</Typography.Text><Button onClick={() => setFocused(parent.id)}>查看放大前的图片</Button></Space>}
              <Space wrap><Button type="primary" href={`${String(result.image_url)}?download=true`}>下载 PNG</Button><Button onClick={() => reuse(current)}>复用参数与种子</Button></Space>
              <Space wrap>
                <Button onClick={() => prepareEnhancement(current, 'realesrgan')} disabled={model?.status !== 'running'}>满意这张，快速超分</Button>
                <Button onClick={() => prepareEnhancement(current, 'supir')} disabled={model?.status !== 'running'}>SUPIR 细节重建</Button>
              </Space>
            </>}
            {['failed', 'cancelled', 'paused'].includes(current.status) && <Button onClick={() => action.mutate({ id: current.id, operation: 'resume' })} loading={action.isPending}>重新生成</Button>}
            {!active(current) && <Popconfirm title="删除此生成记录和图片？" onConfirm={() => action.mutate({ id: current.id, operation: 'delete' })}><Button danger size="small">删除记录</Button></Popconfirm>}
          </Space>}
        </Card>
        <Card title="最近生成" style={{ marginTop: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 12 }}>
            {items.map(task => <button key={task.id} onClick={() => setFocused(task.id)} aria-label={`查看${task.title} ${task.created_at}`} style={{ textAlign: 'left', cursor: 'pointer', padding: 8, borderRadius: 8, border: task.id === current?.id ? '2px solid #1677ff' : '1px solid #8886', background: 'transparent', color: 'inherit' }}>
              {task.status === 'succeeded' ? <img src={String(task.result.image_url)} alt={task.request.prompt} loading="lazy" style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }} /> : <div style={{ padding: '28px 0' }}>{labels[task.status] ?? task.status}</div>}
              <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{task.request.prompt}</div>
              {task.request.operation && task.request.operation !== 'generate' && <Tag>{stages[task.request.operation]}</Tag>}
            </button>)}
          </div>
          {!items.length && <Empty description="暂无历史记录" />}
        </Card>
      </Col>
    </Row>
    <Modal title="放大满意的图片" open={Boolean(source)} onCancel={() => setSource(undefined)} footer={null} width={640}>
      {source && <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Space><Image src={String(source.result.image_url)} alt="选中的放大源图" width={96} /><Typography.Text>{sourceWidth} × {sourceHeight} → {targetSize.join(' × ')}<br />优先使用最初原图，放大结果另存为新记录。</Typography.Text></Space>
        {enhance.error && <Alert type="error" message={enhance.error.message} />}
        <Form form={enhanceForm} layout="vertical" onFinish={values => enhance.mutate(values)}>
          <Form.Item name="operation" hidden><Input /></Form.Item>
          <Form.Item name="sr_engine" label="放大算法"><Select options={[{ value: 'realesrgan', label: 'Real-ESRGAN · 快速超分' }, { value: 'supir', label: 'SUPIR-F · 细节重建' }]} /></Form.Item>
          <Form.Item name="sr_resolution" label="输出长边"><Select options={[2048, 4096].filter(n => n >= Math.max(sourceWidth, sourceHeight) && n <= 4 * Math.max(sourceWidth, sourceHeight)).map(n => ({ value: n, label: `${n} 像素` }))} /></Form.Item>
          <Form.Item name="prompt" label="画面与细节描述" hidden={engine !== 'supir'} rules={[{ required: true, whitespace: true, message: '请输入画面描述' }]}><Input.TextArea rows={3} maxLength={4000} /></Form.Item>
          {engine === 'supir' && <>
            <Row gutter={12}>
              <Col span={12}><Form.Item name="sr_steps" label="重建步数" extra="更多步数耗时更长"><InputNumber min={20} max={60} precision={0} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={12}><Form.Item name="sr_control" label="原图约束" extra="1 更保真；0.93 允许更多细节变化"><InputNumber min={0.8} max={1} step={0.01} style={{ width: '100%' }} /></Form.Item></Col>
            </Row>
            <Row gutter={12}>
              <Col span={12}><Form.Item name="sr_cfg" label="提示词引导强度"><InputNumber min={1} max={8} step={0.5} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={12}><Form.Item name="seed" label="随机种子"><InputNumber min={0} max={4294967295} precision={0} style={{ width: '100%' }} /></Form.Item></Col>
            </Row>
            <Row gutter={12}>
              <Col span={12}><Form.Item name="tile_size" label="分块大小"><Select options={[{ value: 1024, label: '1024 · 推荐' }, { value: 768, label: '768 · 省内存' }, { value: 512, label: '512 · 更省内存' }]} /></Form.Item></Col>
              <Col span={12}><Form.Item name="tile_overlap" label="重叠融合宽度"><Select options={[128, 192, 256].map(n => ({ value: n, label: `${n} 像素` }))} /></Form.Item></Col>
            </Row>
          </>}
          <Typography.Paragraph type="secondary">{engine === 'supir' ? 'DGX 实测 2K（20 步）约 8 分钟，4K 可能需数十分钟。生成式重建可能改变人脸和文字，请对比原图。' : 'DGX 实测 1K → 4K 约 9 秒。采用学习型超分提升边缘和纹理清晰度，不使用提示词重绘。'} 分块进度可见，支持取消。为控制内存，放大时会释放生图权重；返回基础生图时需等待重新加载。</Typography.Paragraph>
          <Button type="primary" htmlType="submit" block loading={enhance.isPending} disabled={model?.status !== 'running'}>开始放大</Button>
        </Form>
      </Space>}
    </Modal>
  </div>
}
