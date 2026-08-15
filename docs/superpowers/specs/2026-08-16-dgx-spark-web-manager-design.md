# DGX Spark Web Manager 系统设计

## 范围

第一版面向单台 DGX Spark，提供真实系统监控、Docker 推理服务发现、Hugging Face 模型管理、SGLang/vLLM 部署、OpenAI 兼容网关、第三方 AI 诊断和受控审批执行。多节点调度、计费和 Kubernetes 不在第一版范围内。

## 架构

系统由一个 ARM64 多阶段镜像交付。构建阶段编译 React SPA，运行阶段由 FastAPI 同时提供 REST/SSE、OpenAI 网关和静态前端。服务挂载 Docker Socket、Hugging Face 缓存、模型目录和持久化数据目录，并使用 host network 访问已有的本机推理端点。

后端按领域拆分为 system、models、deployments、tasks、gateway、providers、diagnostics、auth 和 audit。所有外部运行时行为通过 `RuntimeAdapter` 完成；AI 只能生成 `OperationPlan`，执行层只接受枚举化工具及验证后的参数。

## 数据模型

- `ModelAsset`：来源、仓库、revision、commit、本地路径、格式、量化、大小和能力。
- `Deployment`：模型、运行时、容器、端点、API 别名、参数、状态和健康信息。
- `Task`：类型、状态、进度、输入、结果、错误、取消标记和时间戳。
- `Provider`：名称、base URL、默认模型、加密凭证、超时和启用状态。
- `OperationPlan`：诊断摘要、风险、结构化步骤、状态、批准人与执行结果。
- `ApiKey`：前缀、哈希、权限、最后使用时间和吊销时间。
- `AuditEvent`：主体、动作、资源、结果、来源 IP 和脱敏详情。
- `RequestMetric`：模型、状态码、延迟、输入/输出 token 和时间戳。

## 关键数据流

### 发现

定时任务读取 Docker API、Hugging Face 缓存和模型根目录，将发现结果幂等 upsert。对容器的 `/v1/models` 进行短超时探测，成功后生成部署记录和模型能力；失败只改变健康状态，不删除历史。

### 下载

API 创建持久化任务。Worker 使用固定的 Hugging Face CLI 参数启动子进程，监控目标目录大小并更新进度。暂停或取消通过终止受管进程实现；恢复重新排队并利用 Hub 缓存断点续传。完成后校验文件并触发模型扫描。

### 部署

用户选择模型和运行时后，适配器先验证路径、镜像和参数，再生成预览。批准后 Docker Adapter 以确定性容器名创建或复用容器，等待健康检查，成功则注册网关路由，失败则保存日志并回滚新容器。

### OpenAI 网关

`/v1/models` 汇总健康部署；chat、completion 和 embedding 根据别名路由到本机端点。非流式响应记录延迟和 token；流式响应原样转发 SSE，并在断开时取消上游请求。外部调用使用哈希 API Key 鉴权，管理会话不等同于网关密钥。

### AI 诊断

系统先采集真实指标、部署状态和经截断脱敏的日志，再调用用户配置的 OpenAI 兼容 Provider。模型必须返回符合 JSON Schema 的诊断和步骤。步骤只能映射到允许的操作枚举；未知步骤保留为说明但不可执行。用户批准后逐步执行并写入审计。

## 安全边界

- 单管理员会话使用 HttpOnly、SameSite cookie，写操作要求 CSRF header。
- 管理密码来自环境变量并使用 Argon2 校验；生产环境拒绝默认密码。
- Provider 密钥使用 Fernet 加密，API Key 只存 SHA-256 哈希。
- URL 只允许 HTTP(S)，Provider 拒绝环回、链路本地和云元数据地址；本机部署端点仅由 Docker 发现或管理员显式注册。
- 模型路径必须位于配置的根目录；Docker 镜像和运行时参数使用白名单。
- 不执行 AI 生成的 Shell，不将密钥、Authorization header 或完整环境变量写入日志。

## 任务状态机

`queued -> running -> succeeded|failed|paused|cancelled`。服务重启时，`running` 任务转为 `queued` 并记录恢复事件；同一模型下载和同一部署动作使用唯一幂等键防止重复执行。

## 错误处理

API 使用统一 problem detail 响应并包含可追踪 request ID。外部调用设置连接、读取和总超时；Docker/HF/Provider 错误转换为用户可理解的错误码并保留脱敏技术详情。前端在页面内显示可恢复错误，破坏性动作失败时保留对象和历史。

## 测试策略

- 单元测试覆盖路径、SSRF、密钥、状态机、适配器参数和路由选择。
- 集成测试使用临时 SQLite、MockTransport 和伪 Docker client 覆盖 API。
- 前端使用 Vitest/Testing Library 覆盖主题、响应式数据视图和关键操作。
- Playwright 覆盖登录、发现、下载任务、部署控制、网关密钥、Provider 和审批流。
- 在真实 DGX Spark 上完成 ARM64 镜像构建、容器发现、OpenAI SDK 流式调用和桌面/移动端截图检查。
