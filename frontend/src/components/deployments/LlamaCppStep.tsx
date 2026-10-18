import { Alert, Form, Input, InputNumber, Segmented, Switch, Typography } from 'antd'


export function LlamaCppStep() {
  const mtpEnabled = Form.useWatch(['llama_cpp', 'mtp_enabled'], { preserve: true }) === true

  return (
    <section className="deployment-step" aria-labelledby="llama-cpp-heading">
      <div className="deployment-step-heading">
        <Typography.Title level={4} id="llama-cpp-heading">llama.cpp</Typography.Title>
        <Typography.Text type="secondary">配置 GGUF、多模态投影和模型内置 MTP。</Typography.Text>
      </div>
      <Alert
        type="info"
        showIcon
        message="文件名留空时自动选择唯一主 GGUF，并优先使用 mmproj-F16.gguf。"
      />
      <div className="form-grid">
        <Form.Item
          name={['llama_cpp', 'model_file']}
          label="主模型文件"
          rules={[{ pattern: /^[^/\\]+\.gguf$/i, message: '请输入模型目录内的 GGUF 文件名' }]}
        >
          <Input placeholder="自动识别" />
        </Form.Item>
        <Form.Item
          name={['llama_cpp', 'mmproj_file']}
          label="多模态投影文件"
          rules={[{ pattern: /^mmproj[^/\\]*\.gguf$/i, message: '请输入模型目录内的 mmproj GGUF 文件名' }]}
        >
          <Input placeholder="优先自动选择 mmproj-F16.gguf" />
        </Form.Item>
      </div>
      <Form.Item name={['llama_cpp', 'gpu_layers']} label="GPU 卸载">
        <Segmented<string | number>
          block
          options={[
            { label: '全部层', value: 'all' },
            { label: '仅 CPU', value: 0 },
          ]}
        />
      </Form.Item>
      <Typography.Paragraph type="secondary">
        上下文长度按每个并发会话计算。量化 KV 缓存可降低内存占用，但可能影响精度；量化 V 缓存需要开启 Flash Attention。
      </Typography.Paragraph>
      <div className="form-grid">
        <Form.Item name={['llama_cpp', 'flash_attention']} label="Flash Attention" initialValue="auto">
          <Segmented options={['auto', 'on', 'off']} />
        </Form.Item>
        {(['cache_type_k', 'cache_type_v'] as const).map((field) => (
          <Form.Item key={field} name={['llama_cpp', field]} label={field === 'cache_type_k' ? 'K 缓存精度' : 'V 缓存精度'} initialValue="f16">
            <Segmented options={['f16', 'q8_0', 'q4_0']} />
          </Form.Item>
        ))}
        <Form.Item name={['llama_cpp', 'batch_size']} label="逻辑批大小" initialValue={2048}>
          <InputNumber min={32} max={8192} />
        </Form.Item>
        <Form.Item name={['llama_cpp', 'ubatch_size']} label="物理批大小" initialValue={512}>
          <InputNumber min={32} max={8192} />
        </Form.Item>
        <Form.Item name={['llama_cpp', 'cache_ram_mib']} label="提示词 RAM 缓存 (MiB，0 关闭)" initialValue={8192}>
          <InputNumber min={0} max={65536} />
        </Form.Item>
        <Form.Item name={['llama_cpp', 'context_checkpoints']} label="每会话上下文检查点" initialValue={32}>
          <InputNumber min={0} max={64} />
        </Form.Item>
      </div>
      <div className="form-grid">
        <Form.Item name={['llama_cpp', 'jinja']} label="Jinja 对话模板" valuePropName="checked">
          <Switch />
        </Form.Item>
        <Form.Item
          name={['llama_cpp', 'continuous_batching']}
          label="连续批处理"
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>
        <Form.Item name={['llama_cpp', 'mtp_enabled']} label="模型内置 MTP" valuePropName="checked">
          <Switch />
        </Form.Item>
        <Form.Item
          name={['llama_cpp', 'mtp_tokens']}
          label="每轮 MTP Token"
          rules={[{ required: mtpEnabled, type: 'number', min: 1, max: 64 }]}
        >
          <InputNumber min={1} max={64} disabled={!mtpEnabled} />
        </Form.Item>
      </div>
    </section>
  )
}
