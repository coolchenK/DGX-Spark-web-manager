# 上游网关（Upstream Gateway）补全设计

## 目标

最新提交 `4435dee` 为网关加入了上游回退：当请求的模型不在本机时，按 `DGX_FALLBACK_BASE_URL` 转发到外部 OpenAI 兼容端点，使单个入口同时暴露本地与远程模型。该能力目前只完成了转发路径，配置与可观测性仍是半成品：客户端无法通过 `/v1/models` 发现上游模型，配置只能写在环境变量里且必须重建容器，上游流量缺少 Token 计量。

本设计补全这条链路，使"本地 + 上游"成为一个可发现、可配置、可观测、可测试的统一入口，同时保持现有 env 配置与本地推理行为完全兼容。

## 现状与缺口

以下每一项都在当前代码中核实：

| # | 缺口 | 代码证据 |
| --- | --- | --- |
| 1 | `/v1/models` 不列出上游模型 | `openai_models()` 只遍历 `_healthy_gateway_deployments(db)`，即本地健康部署 |
| 2 | `/v1/models/{model}` 对上游模型返回 404 | `openai_model()` 仅在该本地列表内查找 |
| 3 | 配置仅来自环境变量，明文且需重启 | `config.py` 的 `fallback_base_url` / `fallback_api_key`；无 `SecretSetting` 存储、无热更新 |
| 4 | 无管理界面 | `GatewayPage` 只有 API Key 表格与 SDK 示例，无上游区块 |
| 5 | 上游流量不记录 Token | `api/gateway.py` 中 `usage` 零引用，`_proxy_fallback` 调用 `record_request_metric` 时未传 `usage`；本地路径 `gateway/proxy.py` 传了 |
| 6 | 无上游健康探测 | 无对应端点，上游不可用时只能在真实请求中失败 |

缺口 5 的影响是具体的：`/api/gateway/stats` 对 `RequestMetric` 求和，上游行的 `prompt_tokens` / `completion_tokens` 为 `NULL`，因此 Token 总量与 `tokens_per_second` 漏算全部上游流量，而错误率与延迟正常计入。

## 配置存储与优先级

沿用 `SecretSetting`（`huggingface_token` 已使用同一模式），不新增数据库表，因此不需要迁移：

- `upstream_base_url`：加密存储；它不是机密，接口按原文返回以便界面显示。
- `upstream_api_key`：加密存储；接口只返回 `api_key_configured` 布尔值，任何路径都不回传原文。

解析优先级（从高到低）：

1. 数据库值（管理员在界面保存）
2. 环境变量 `DGX_FALLBACK_BASE_URL` / `DGX_FALLBACK_API_KEY`
3. 未配置

数据库存在 `upstream_base_url` 时忽略环境变量；清除数据库配置后回落到环境变量，使已通过 env 部署的实例行为不变。现有只读 `settings` 的 `_fallback_upstream(settings)` 替换为统一解析函数 `resolve_upstream_gateway(db, settings)`，转发路径与管理接口共用同一解析结果。

保存后立即生效，不重建容器，与 Hugging Face Token 的更新方式一致。

## API 契约

### `GET /api/gateway/upstream`（管理员）

```json
{
  "base_url": "https://upstream.example/v1",
  "api_key_configured": true,
  "source": "database",
  "enabled": true
}
```

`source` 为 `database` / `environment` / `unset`。`enabled` 表示当前是否启用回退（等价于 `base_url` 非空）。

### `PUT /api/gateway/upstream`（管理员 + CSRF）

请求体：

```json
{ "base_url": "https://upstream.example/v1", "api_key": "sk-..." }
```

- `base_url` 为空或 `null`：清除上游配置（同时删除密钥）。
- `api_key` 为 `null` 或省略：保留已存密钥；传空字符串则清除密钥但保留 `base_url`。
- 校验：仅接受 `http` / `https`，必须包含 host，拒绝 userinfo 与 fragment，长度受限且去除首尾空白与末尾斜杠。
- 写入审计 `gateway.upstream.update`，`details` 只含 `base_url`、`api_key_configured` 与 `source`，不含密钥。

### `POST /api/gateway/upstream/test`（管理员 + CSRF）

请求 `GET {base_url}/models`（携带配置密钥），使用有界超时，返回：

```json
{
  "status": "ok",
  "latency_ms": 128,
  "model_count": 37,
  "detail": null
}
```

`status` 为 `ok` / `unavailable` / `error` / `unset`。`detail` 必须脱敏：不返回密钥，不返回上游响应体全文，只保留有界的原因摘要。写入审计 `gateway.upstream.test`。

### `GET /v1/models` 合并上游

- 本地健康路由保持现有字段不变。
- 上游模型追加为独立条目：`owned_by: "upstream"`、`dgx_source: "upstream"`，`id` 使用上游返回的模型名。
- 未知信息不得伪造：上游条目的能力、上下文与模态无法从 `/models` 得知，因此 `capabilities` 为 `[]`、`input_modalities` 为 `[]`、上下文与输出上限类字段为 `null`。这与产品原则"未知必须明确表达"一致，客户端应据 `dgx_source` 判断条目来源。
- 命名冲突时本地优先：上游条目与本地路由同名时跳过上游条目，避免遮蔽本机实例。
- 上游不可用或未配置时，本地列表照常返回，并追加顶层字段 `upstream: {"status": "unavailable" | "unset", "detail": null}`；这是增量字段，不破坏现有客户端。
- 结果按 `base_url` 与密钥指纹缓存（默认 30 秒，可配置），避免每次 `/v1/models` 都请求上游；配置变更立即失效缓存。

### `GET /v1/models/{model}`

沿用同一合并结果查找，使上游模型可被单个查询命中；本地优先规则同样适用。

## Token 计量修正

把 usage 解析抽成共享助手，供本地与上游路径复用：

- 非流式响应：解析 JSON 正文的 `usage`。
- 流式响应：扫描 SSE `data:` 帧，取最后一个包含 `usage` 的事件（与 `gateway/proxy.py` 现有行为一致）。

`_proxy_fallback` 的三处 `record_request_metric` 调用（连接失败、非流式、流式收尾）补传该值。修正后 `/api/gateway/stats` 的 Token 总量与 `tokens_per_second` 将纳入上游流量。

## 审计边界

- 记录管理动作：`gateway.upstream.update`、`gateway.upstream.test`。
- 不记录每次转发。转发流量已由 `RequestMetric` 覆盖，与本地路径的处理一致；逐请求写审计会让审计表被推理流量淹没，也无法提供额外观测价值。

## 前端

`GatewayPage` 新增"上游网关"区块，沿用现有 `section-heading`、`Descriptions`、`Form` 与 `Popconfirm`：

- 未配置：说明上游用途，展示配置按钮。
- 已配置：显示 `base_url`、密钥状态、来源（数据库 / 环境变量），提供测试与修改入口。
- 配置弹窗：`base_url` 输入 + `api_key` 密码输入（留空表示不修改）；清除操作需二次确认。
- 测试结果：成功显示模型数量与延迟；失败显示脱敏原因。
- 复用现有布局类，保证移动端与深色主题表现一致。

## 错误与兼容性

- 未配置上游时，未知模型仍返回 404，行为与当前一致。
- 上游故障不得影响本地模型列表、本地推理与管理接口。
- `/v1/models` 新增字段为增量；上游条目是新增条目，不改变本地条目字段。
- 环境变量配置继续生效；已部署实例无需修改即可升级。
- 不新增数据库表，不需要 Alembic 迁移。

## 测试

后端：

- `/v1/models` 合并本地与上游，且同名时本地优先。
- 上游不可用 / 未配置时，本地列表完整返回并带 `upstream` 状态字段。
- usage 提取：非流式 JSON、SSE 含 usage、SSE 不含 usage（不得报错）。
- 配置优先级：数据库覆盖环境变量；清除后回落环境变量。
- 校验：非 http(s)、缺 host、含 userinfo、超长 `base_url` 被拒绝。
- 任何响应都不包含密钥原文。
- 权限：非管理员与缺失 CSRF 的请求被拒绝。
- 审计：配置变更与测试写入对应审计事件。
- 上游结果缓存命中不重复请求，配置变更后缓存失效。

前端：

- 未配置、已配置、测试成功、测试失败四种状态渲染。
- `base_url` 必填校验与清除确认流程。
- 运行完整后端测试、前端测试、类型检查与生产构建。

部署到 DGX Spark 后，使用真实上游网关验证 `/v1/models` 合并、流式转发与 Token 统计。
## 实施记录（2026-09-18）

实施期间通过测试发现并修复了已上线回退路径的两个缺陷，并做出一处契约修正：

1. **上游 URL 拼接错误（已上线缺陷）**：`_proxy_fallback` 直接拼接 `f"{base_url}{endpoint}"`，
   当运维按 README 的约定把上游配成 `https://host/v1` 时，实际请求变成
   `https://host/v1/v1/chat/completions`，必然失败。现在统一由 `upstream_api_root` /
   `upstream_request_url` 归一到唯一一个 `/v1` 前缀，两种写法都可用。
2. **上游配置不生效于转发路径**：`_fallback_upstream` 只读环境变量，面板保存的配置不会作用于
   转发。现在解析统一走 `resolve_upstream_gateway`，面板配置与转发使用同一来源。
3. **契约修正**：`resolve_upstream_gateway` 需要 `SecretBox` 才能解密存储的密钥，签名增加该参数；
   存储改为职责单一的 `set_upstream_base_url` / `clear_upstream_gateway` /
   `set_upstream_api_key` / `clear_upstream_api_key`，以便区分「字段缺省」与「显式置空」。

另修复了 `backend/tests/test_gateway.py` 中两个既有的未使用导入（`asyncio`、`httpx`），
它们会让当前 main 的 CI `ruff check` 步骤失败。
## 上游模型暴露选择（2026-09-18 追加）

上游解析出的模型默认全部在本网关提供；管理员可以在「API 网关」页逐个选择是否提供。

### 语义与默认值

存储两种模式，在「新上游模型自动对外」的不可预期性与「升级即断流」之间取折中：

- `expose_all = true`（默认）：提供上游解析出的全部模型，转发路径保持「未知模型一律转发」的既有行为。
  默认取该值是为了不破坏已经依赖上游模型的现有部署，例如 Codex 使用的远程模型。
- `expose_all = false`：只提供 `selected_models` 中的模型。

新出现的上游模型在 `expose_all = false` 时不会自动提供，需要管理员显式选择。

### 生效范围

暴露选择同时约束**发现**与**调用**，否则开关只是装饰：

- `GET /v1/models` 不列出未提供的上游模型。
- `GET /v1/models/{model}` 对未提供的模型返回 404。
- 请求未提供的上游模型**不再转发**，返回与其他隐藏部署一致的 404。

### API 契约

- `GET /api/gateway/upstream` 增加 `expose_all` 与 `selected_models`。
- `GET /api/gateway/upstream/models` 返回实时解析到的模型及其 `exposed` 状态，供界面渲染开关；
  上游不可用时返回 `status: unavailable`，不影响已保存的选择。
- `PUT /api/gateway/upstream/exposure`：`{ "expose_all": bool, "selected_models": [str] }`。
  模型名做长度与字符校验，数量有上界；写入审计 `gateway.upstream.exposure.update`。

### 前端

「上游网关」区块增加暴露选择：模式开关（提供全部 / 仅提供所选）+ 实时模型列表逐项开关 + 保存。
`expose_all` 为真时列表显示为全部提供且不允许逐项取消，避免出现「全部提供但该项关闭」的矛盾状态。

### 测试

- 默认 `expose_all` 为真，升级后 `/v1/models` 与转发行为不变。
- `expose_all = false` 时只列出且只转发所选模型，未选模型返回 404。
- 选择持久化到数据库，重新读取一致；非法模型名与超量选择被拒绝。
- 暴露变更写入审计，并使模型列表缓存失效。
