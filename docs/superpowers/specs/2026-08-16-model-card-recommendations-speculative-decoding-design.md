# 模型卡推荐参数与 Draft Model 推测解码设计

日期：2026-08-16  
状态：已确认，待实施

## 1. 目标

在新建或编辑模型部署时，管理器应读取 Hugging Face 模型卡和本地模型配置，针对当前 DGX Spark 的实时资源生成可解释的部署建议，并自动预填部署表单。用户可以修改所有建议值，管理器不得在后台覆盖已经手动修改的字段。

当本地存在与基础模型兼容的 Draft Model 时，部署向导应允许启用 Draft Model 推测解码。系统必须为 vLLM 和 SGLang 分别生成受验证的运行时配置，并将基础模型、Draft Model、推荐来源、最终参数和 OpenAI API 默认生成参数统一保存在部署记录中。

本功能同时满足以下要求：

- 模型卡明确给出的设置优先于通用经验值。
- 模型卡信息不足时，可以调用用户配置的第三方 OpenAI 兼容 Provider 补充建议。
- AI 建议必须针对当前 DGX Spark 的架构、统一内存、可用资源、当前部署和所选运行时。
- AI 只能返回受类型约束的建议，不能生成或执行 Shell、Docker 命令或任意运行时参数。
- OpenAI API 请求未显式提供生成参数时，应用部署保存的默认生成参数。
- 桌面端、移动端、浅色和深色模式均保持完整可用。

## 2. 非目标

- 不自动训练、转换或量化 Draft Model。
- 不自动启用未经本地验证的异构词表映射。
- 不让 AI 直接创建、启动、停止或删除容器。
- 不从任意模型卡文本复制未经白名单验证的命令行参数。
- 不承诺推测解码在所有负载下都提升吞吐；界面明确将其描述为需要实测的延迟优化。
- 不在本功能中增加多节点调度或 Kubernetes 支持。

## 3. 选定方案

采用“结构化规则优先，AI 仅补充缺失项”的混合方案。

纯规则方案稳定但无法理解模型卡中的非结构化说明；AI 优先方案覆盖面更大，但结果会漂移并增加延迟和外部 API 成本。混合方案保留确定性、安全性和可解释性，同时允许在模型卡含糊时针对 DGX Spark 给出实用建议。

模型卡是带 YAML 元数据的 Markdown 文件，结构化元数据与正文承担不同职责，因此系统同时读取两部分，而不假设所有部署建议都在 `card_data` 中。参考：[Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards) 和 [huggingface_hub ModelCard](https://huggingface.co/docs/huggingface_hub/main/guides/model-cards)。

## 4. 总体架构

### 4.1 `DeploymentRecommendationService`

该服务接收基础模型、运行时镜像和可选 AI Provider，协调以下只读组件：

1. `ModelEvidenceLoader`：读取本地 `config.json`、`generation_config.json`、Tokenizer 配置、量化配置、模型大小、Hugging Face 元数据和完整模型卡。
2. `ModelCardEvidenceExtractor`：只从已识别的 JSON、YAML、Python 配置和命令代码块中提取白名单字段；普通说明文字不直接变成可执行参数。
3. `ResourceEstimator`：结合 DGX Spark 实时统一内存、模型权重、KV Cache、运行时开销、Draft Model 开销和当前部署生成资源预算。
4. `RuntimeCapabilityService`：按运行时镜像摘要探测并缓存支持的参数和推测解码方法。
5. `AIRecommendationClient`：仅在字段缺失或证据含糊时，调用选定的第三方 OpenAI 兼容 Provider。
6. `DraftCompatibilityService`：为基础模型生成兼容、待确认和不兼容的本地 Draft Model 候选。

服务返回一个完整的建议文档。每个建议字段都包含值、来源、置信度、解释和可选警告。建议生成本身不创建 `OperationPlan`，因为它是只读操作；真正部署仍经过现有预览、确认、任务和审计流程。

### 4.2 证据与优先级

系统按以下优先级计算初始值：

1. 模型卡中明确且可验证的运行时或生成参数。
2. 本地结构化文件，如 `generation_config.json`、`config.json` 和量化配置。
3. 所选运行时镜像的能力与默认行为。
4. 针对 DGX Spark 的确定性资源规则。
5. 第三方 AI 对仍未确定字段的补充建议。

上述顺序决定“期望值”的证据优先级，随后必须统一经过事实约束和 DGX Spark 资源预算修正。资源修正可以降低模型卡推荐的上下文、并发或批处理值，但不能突破模型配置硬上限。发生修正时，最终字段来源记为 `device_rule`，同时在解释中保留原始模型卡值和修正原因。

用户在表单中的手动修改位于最终优先级之上。重新获取建议时只更新未被用户修改的字段；“重新应用全部建议”是唯一可以覆盖手动值的入口，并需要明确点击。

量化格式、模型架构、上下文硬上限和运行时不支持项属于事实约束，AI 不得覆盖。AI 可以降低建议值，也可以解释为何采用更保守设置，但不能把不支持的组合标记为支持。

### 4.3 AI 输入与输出边界

AI 请求包含：

- 截断并去除不可见控制字符的模型卡正文；
- 结构化模型配置的白名单字段；
- 模型大小、量化方式和参数规模；
- DGX Spark 架构、总统一内存、当前可用内存和安全保留量；
- 当前运行部署的资源摘要；
- 所选运行时及探测到的能力；
- 尚未确定的字段列表和严格 JSON Schema。

不发送 Provider 密钥、Authorization 请求头、环境变量、完整容器日志或无关本地文件。AI 响应先经过 JSON 解析，再经过 Pydantic 类型、数值范围、运行时能力和资源预算校验。无效字段逐项丢弃，不因单个字段失败而丢弃全部确定性建议。

模型卡和配置文件一律视为不可信数据。发送给 AI 时使用明确的数据边界，并要求忽略其中要求泄露信息、改变系统角色、执行命令或绕过输出 Schema 的指令。即使 Provider 遵循了恶意模型卡内容，后端白名单和类型校验仍是最终安全边界。

部署向导提供“AI 推荐服务”选择器。只有一个已启用 Provider 时自动选中；存在多个 Provider 时优先恢复当前浏览器上次选择，用户仍可切换或禁用 AI 补充。没有 Provider 或 Provider 不健康时继续生成规则建议，并返回 `partial` 状态。

为避免自动预填重复产生外部调用，AI 解析结果按模型提交、模型卡证据哈希、运行时、镜像摘要、Provider 和推荐 Schema 版本缓存。实时资源修正每次重新计算，不缓存旧的可用内存。前端对选择变化做防抖，并提供显式“重新分析”操作来绕过 AI 缓存。

## 5. 推荐数据契约

新增接口：

```http
POST /api/deployments/recommendations
```

请求：

```json
{
  "model_id": "model-uuid",
  "runtime": "vllm",
  "image": "vllm/vllm-openai:v0.27.1",
  "provider_id": "provider-uuid-or-null"
}
```

响应的核心结构：

```json
{
  "status": "complete",
  "generated_at": "2026-08-16T12:00:00Z",
  "model_id": "model-uuid",
  "runtime": "vllm",
  "image_digest": "sha256:...",
  "fields": {
    "context_length": {
      "value": 32768,
      "source": "model_card",
      "confidence": "high",
      "reason": "模型卡的 vLLM 示例明确设置 max-model-len"
    }
  },
  "generation_defaults": {
    "temperature": {
      "value": 0.6,
      "source": "model_card",
      "confidence": "high",
      "reason": "模型卡推荐的采样设置"
    }
  },
  "resource_snapshot": {},
  "runtime_capabilities": {},
  "draft_candidates": [],
  "warnings": []
}
```

`status` 只允许：

- `complete`：所有关键字段均有有效建议。
- `partial`：可以部署，但有字段使用运行时默认值或 AI 补充失败。
- `unavailable`：基础模型、镜像或关键配置无法验证，不能自动应用建议。

建议来源只允许：`model_card`、`local_config`、`runtime_default`、`device_rule`、`ai`。置信度只允许：`high`、`medium`、`low`。

## 6. 部署规格与持久化

扩展现有 `DeploymentSpec`，增加两个受类型约束的对象：

```text
generation_defaults
speculative
```

`generation_defaults` 的初始白名单包括：

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `presence_penalty`
- `frequency_penalty`
- `max_tokens`
- `stop`

每个字段都有明确类型和范围。只有所选运行时实际支持的扩展字段才会保存为可应用默认值；模型卡中的其他建议以只读说明显示，不透传到网关。

`speculative` 包含：

```json
{
  "draft_model_id": "model-uuid",
  "method": "draft_model",
  "num_speculative_tokens": 5,
  "num_steps": null,
  "eagle_top_k": null,
  "num_draft_tokens": null,
  "manual_review_acknowledged": false
}
```

允许的方法由运行时能力决定，首版公共枚举为 `draft_model`、`eagle`、`eagle3` 和 `mtp`。运行时专属字段只在对应方法下有效，Pydantic 使用可辨识联合类型拒绝无效组合。

公共方法由适配器映射为运行时语法，而不是直接透传：SGLang 的普通 Draft Model 使用 `STANDALONE`，`mtp` 按镜像能力映射为 `NEXTN` 或运行时声明的方法；vLLM 使用其 `speculative_config.method`。镜像未声明对应映射时，该方法不可选择。

部署记录继续使用现有 `Deployment.config` JSON 保存以下数据，不新增独立建议表：

- 最终 `spec`；
- `generation_defaults`；
- `speculative`，包含 Draft Model ID、仓库、版本和路径快照；
- 推荐文档摘要、证据哈希、Provider ID和生成时间；
- 推荐时的资源快照；
- 用户修改过的字段列表。

不在部署记录中保存完整模型卡或完整 AI 响应，避免数据库膨胀。编辑、克隆和重新部署必须恢复上述字段。部署前重新根据 `draft_model_id` 解析当前本地路径，不信任历史路径快照。

## 7. DGX Spark 资源建议

DGX Spark 使用统一内存。资源计算以主机内存为唯一总预算，不把 `nvidia-smi` 显存数值与主机内存相加，避免重复计算。

`ResourceEstimator` 使用以下信息形成可解释估算：

- 实际权重文件总大小和量化配置；
- 模型层数、隐藏维度、KV Head、数据类型和上下文长度；
- KV Cache、运行时工作区和 CUDA Graph 保留；
- 最大并发与最大批处理 Token；
- Draft Model 权重、KV Cache 和推测验证开销；
- 当前 `available` 统一内存和已有容器的资源摘要；
- 为操作系统、管理器和突发开销保留的安全预算。

无法从配置精确计算时使用保守系数，并把置信度降为 `low`。预计超过“物理总内存减安全保留量”时阻止部署；仅超过当前可用内存时返回警告，要求停止其他部署或明确确认继续。资源状态在预览和实际创建任务开始前各采样一次；显著变化时使旧预览失效。

## 8. Draft Model 兼容性

候选模型必须是状态为可用、文件完整且位于配置模型根目录内的本地资产。基础模型本身不能作为外部 Draft Model 候选。

兼容性结论分三类：

### 8.1 `compatible`

必须同时满足运行时支持和本地文件校验，并具有足够的配对证据：

- EAGLE/EAGLE3/MTP 模型明确声明目标模型或基础模型，且与所选目标匹配；或
- 普通 Draft Model 与目标模型具有兼容的 Tokenizer、词表、特殊 Token 和模型族，并且配置或模型卡明确支持 Draft Model 用法。

### 8.2 `review`

模型没有已知冲突，但配对证据不足，或配对关系只从模型卡正文/AI 中推断。此类模型只在高级模式显示，选择后必须勾选风险确认。AI 只能把候选标记为 `review`，不能仅凭推断升级为 `compatible`。

### 8.3 `incompatible`

存在明确冲突，例如目标模型不匹配、Tokenizer 或特殊 Token 冲突、运行时不支持算法、文件不完整、上下文要求冲突或资源超过硬上限。此类模型可以在高级列表中查看原因，但不能选择部署。

首版不自动启用 vLLM 的异构词表 Draft Model。即使新版 vLLM 支持 `use_heterogeneous_vocab`，仍需后续独立设计和基准验证后再开放。

## 9. 运行时适配

### 9.1 能力探测

`RuntimeCapabilityService` 对允许列表中的镜像执行固定、无用户输入的帮助/版本探测，并按不可变镜像摘要缓存结果。探测命令属于后端常量，不接受前端或 AI 提供的命令片段。

探测失败时只使用该允许镜像随项目维护的保守能力清单；如果清单也没有推测解码能力，则禁用 Draft Model，而不是猜测参数。当前设备上的 vLLM 0.27.1 支持 `--speculative-config`，定制 SGLang 镜像支持 `--speculative-*` 参数，实施时仍以镜像实际探测结果为准。

### 9.2 vLLM

vLLM 适配器将受验证对象序列化为规范 JSON，并作为单个 `--speculative-config` 参数传入。例如：

```json
{
  "method": "draft_model",
  "model": "/models/path/to/draft",
  "num_speculative_tokens": 5
}
```

不接受前端提供的 JSON 字符串。`model` 必须由 `draft_model_id` 经模型根目录校验后转换为容器内路径。参考：[vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

### 9.3 SGLang

SGLang 适配器按方法生成独立参数：

- `--speculative-algorithm`
- `--speculative-draft-model-path`
- `--speculative-num-steps`
- `--speculative-eagle-topk`
- `--speculative-num-draft-tokens`

未被建议或用户设置的 EAGLE 调优参数保持省略，以便 SGLang 使用其模型相关自动值。需要手动设置时，相关参数必须成组验证。参考：[SGLang Speculative Decoding](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx) 和 [SGLang Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)。

### 9.4 模型挂载与预览

基础模型和 Draft Model 均以只读方式暴露给运行时。两条路径分别通过现有模型根目录校验，预览必须显示解析后的主机路径、容器路径、命令和预计资源。前端不得提交任意挂载定义。

## 10. OpenAI 网关默认生成参数

仅对 `/v1/chat/completions` 和 `/v1/completions` 合并默认生成参数，其他端点保持不变。固定优先级为：

```text
请求显式参数 > 部署保存的默认生成参数 > 推理运行时默认值
```

流式和非流式请求在转发前使用同一个纯函数完成合并。客户端显式发送的值即使为 `0`、`false` 或空数组也视为显式设置，不得被覆盖。`max_tokens` 与 `max_completion_tokens` 作为同一语义组处理：请求包含任意一个时，不注入部署默认的 `max_tokens`。

非 OpenAI 标准但由 vLLM/SGLang 支持的字段，如 `top_k` 和 `min_p`，只在目标运行时能力明确支持时注入。未知字段不保存、不注入。网关审计记录采用了哪些默认字段，但不记录完整用户 Prompt。

共享同一个 `route_alias` 的部署必须具有完全一致的 `generation_defaults`。预览和创建部署时发现同别名部署的默认值不同，应阻止保存并显示差异；这样负载均衡不会导致同一个网关模型名在不同请求中采用不同默认采样行为。

## 11. 前端交互

现有部署表单改为 Ant Design 分步向导：

1. **基础模型**：选择模型、运行时、镜像和可选 AI 推荐服务。
2. **推荐配置**：展示部署参数与默认生成参数，自动预填并标注来源、置信度和解释。
3. **Draft Model**：默认只展示兼容候选；高级模式展示待确认和不可选的不兼容候选及原因。
4. **部署预览**：展示最终参数、容器命令、模型挂载、总资源估算、网关默认值和回滚行为。

用户编辑字段后显示“已手动修改”状态。模型、运行时或镜像变化时取消旧请求并生成新建议；旧响应不得覆盖新选择。基础模型变化会清除 Draft Model 选择并重新计算候选。

AI 失败、模型卡缺失或部分字段无法确定时在对应步骤显示可恢复提示，不清空已有表单。向导允许用户在确定性校验通过后继续手动部署。

桌面端使用横向步骤；窄屏使用纵向步骤和全宽控件。高级参数放在 `Collapse` 中，来源和风险使用 Ant Design `Tag`、`Alert`、`Descriptions` 和图标表达。避免嵌套卡片，确保深色模式、长模型名和长错误文本不溢出。

## 12. 错误处理与回滚

- 模型卡或 Hub 不可用：回退本地配置和设备规则。
- AI Provider 不可用、超时或返回无效 JSON：逐项丢弃 AI 建议，返回 `partial`。
- 运行时能力探测失败：使用保守能力清单；无清单时禁用推测解码。
- Draft Model 在预览后被移动或删除：创建任务开始前重新解析并失败，不启动容器。
- 资源快照过期或变化显著：使预览失效并要求重新确认。
- 容器创建、启动、健康检查或 `/v1/models` 注册失败：删除新建的管理器容器，保留基础模型和 Draft Model 文件，任务记录失败原因。
- 默认生成参数合并异常：拒绝当前请求并记录脱敏错误，不静默发送未经验证的参数。

推荐调用、手动覆盖、风险确认、部署结果和回滚结果均写入审计。审计与日志不得包含 Provider 密钥、Authorization 请求头、完整 AI 原始响应或用户完整 Prompt。

## 13. 测试与验收

### 13.1 后端单元测试

- 模型卡、本地配置、运行时默认、DGX Spark 规则和 AI 的优先级。
- 模型卡代码块白名单提取及恶意/未知参数拒绝。
- AI 无 Provider、超时、无效 JSON、越界值和不支持字段。
- 统一内存不重复计算、KV Cache 估算、Draft 开销和硬/软资源阈值。
- Draft Model 的 `compatible`、`review`、`incompatible` 状态与原因。
- vLLM 规范 JSON 和 SGLang 参数生成，不允许路径或参数注入。
- `DeploymentSpec` 可辨识联合类型和编辑/克隆持久化。
- OpenAI 默认参数合并，覆盖流式、非流式、零值和 `max_tokens` 等价字段。
- 共享 `route_alias` 的默认生成参数一致性校验。

### 13.2 后端集成测试

- 使用假的 Hugging Face、Provider、Docker 和系统指标完成建议、预览、创建、失败回滚全流程。
- 按镜像摘要缓存能力探测，镜像变更后重新探测。
- 创建任务开始前资源和 Draft Model 二次校验。
- 网关将默认生成参数转发给正确的健康部署，客户端显式值保持不变。

### 13.3 前端测试

- 自动预填和来源展示。
- 手动修改字段不被后台刷新覆盖。
- “重新应用全部建议”恢复推荐值。
- Provider 缺失、AI 失败和部分建议状态。
- Draft 候选筛选、高级模式、风险确认和基础模型切换清理。
- 旧建议请求晚返回时不覆盖当前表单。
- 编辑和克隆已有推测解码部署。

### 13.4 真实 DGX Spark 验收

- 探测当前 vLLM 0.27.1 与定制 SGLang 镜像的实际能力。
- 对现有本地模型生成推荐并核对资源快照、模型卡证据和预览命令。
- 回归 `/v1/models`、流式和非流式 Chat Completions，确认现有网关行为不回退。
- 存在兼容 Draft Model 时执行一次真实推测解码部署和请求；不存在时验证“无兼容候选”及高级列表状态，不为验收强行选择不兼容模型。
- 使用真实浏览器检查桌面、移动端、浅色和深色模式，无遮挡、溢出或不可操作控件。

## 14. 完成标准

满足以下条件后视为功能完成：

- 新建部署会自动生成并预填针对当前 DGX Spark 的可解释建议。
- 模型卡不明确时能够使用已配置的第三方 AI 补充建议，失败时可安全降级。
- 用户修改不会被异步刷新覆盖。
- 默认生成参数按约定优先级应用到 OpenAI 兼容请求。
- 兼容 Draft Model 可以随基础模型统一预览、部署、编辑和管理。
- 已知不兼容组合、运行时不支持组合和超过硬资源上限的组合无法部署。
- vLLM 和 SGLang 的命令仅由后端受控适配器生成。
- 自动化测试通过，并在真实 DGX Spark 上完成规定的回归和界面验收。
