import { render, screen } from '@testing-library/react'
import { Form } from 'antd'
import { describe, expect, it } from 'vitest'

import { LlamaCppStep } from './LlamaCppStep'


describe('LlamaCppStep', () => {
  it('shows GGUF, projector and MTP controls', () => {
    render(
      <Form initialValues={{ llama_cpp: { gpu_layers: 'all', mtp_enabled: true, mtp_tokens: 3 } }}>
        <LlamaCppStep />
      </Form>,
    )

    expect(screen.getByLabelText('主模型文件')).toBeInTheDocument()
    expect(screen.getByLabelText('多模态投影文件')).toBeInTheDocument()
    expect(screen.getByText('全部层')).toBeInTheDocument()
    expect(screen.getByLabelText('模型内置 MTP')).toBeChecked()
    expect(screen.getByLabelText('每轮 MTP Token')).toHaveValue('3')
  })
})
