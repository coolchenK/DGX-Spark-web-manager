# DGX Spark Web Manager

面向 NVIDIA DGX Spark 的中文模型部署与运维面板。在一个界面中查看主机资源、下载 Hugging Face 模型、管理推理容器，并通过统一的 OpenAI 兼容接口连接聊天客户端和开发工具。

项目面向 ARM64 / GB10 环境，支持 **SGLang、vLLM 和 llama.cpp**。管理服务使用 FastAPI、SQLite 和 React 19，界面提供桌面与移动端布局，以及浅色、深色和跟随系统主题。

## 功能概览

| 页面 | 主要功能 |
| --- | --- |
| 系统概览 | 查看 CPU、统一内存、磁盘、GPU、温度、功耗和驱动状态 |
| Hugging Face | 搜索模型、查看兼容性与模型卡、下载文件，支持任务暂停、恢复和取消 |
| 模型库 | 扫描本地模型与 Hugging Face 缓存，查看模型能力、历史测速结果和删除任务 |
| 部署实例 | 创建、预览、启动、停止、重启、编辑、克隆和卸载部署，查看日志、内存和 TPS |
| API 网关 | 管理 API Key、模型路由、OpenAI 兼容请求和调用指标 |
| 在线 AI 服务 | 配置、测试 OpenAI 兼容的远程 AI 服务，供部署建议和运维诊断使用 |
| AI 运维助手 | 根据主机、容器、日志和任务信息诊断问题，生成可审阅的执行计划 |
| 任务中心 / 日志与审计 | 跟踪持久化任务、执行结果与管理操作 |
| 系统设置 | 管理 Hugging Face 凭据等系统配置 |

管理器启动时会发现已有推理容器，不会因此重建或重启它们。外部容器标记为非托管实例；只有带有 `com.dgx-spark-manager.managed=true` 标记、由管理器创建的容器，才能通过面板卸载。

## 快速安装

### 环境要求

- NVIDIA DGX Spark，运行 Ubuntu / DGX OS，架构为 `aarch64` 或 `arm64`。
- Docker Engine、Docker Compose 插件和 NVIDIA Container Toolkit。
- 当前用户可以访问 Docker；安装 Host Agent 时可以使用 `sudo`。
- Git、Python 3、`curl`、`openssl` 和 systemd 等主机工具。
- 模型文件、Docker 镜像和构建缓存所需的可用磁盘空间；安装预览中的空间估计不包含模型。

### Docker Compose 安装

在 DGX Spark 上执行：

```bash
git clone https://github.com/coolchenK/DGX-Spark-web-manager.git
cd DGX-Spark-web-manager

# 预览安装目录、主机代理和执行计划
./scripts/install.sh

# 正式安装
./scripts/install.sh --apply
```

安装脚本会安装 Host Agent、生成或补全 `.env`、创建数据目录、构建管理器镜像、启动服务，并等待 `/api/health` 就绪。首次安装生成随机管理员密码，默认用户名为 `admin`；密码保存在权限为 `0600` 的 `.env` 文件中，脚本不会直接打印密码。

浏览器打开 `http://<DGX-SPARK-IP>:3000`。登录后可以先查看已有模型和部署，再配置下载或创建新实例。

需要复用现有模型目录时，在安装前指定路径：

```bash
HF_HOME_HOST=/path/to/huggingface \
MODEL_HOME_HOST=/path/to/models \
HOST_IP='<DGX-SPARK-IP>' \
./scripts/install.sh --apply
```

`HF_HOME_HOST` 指向 Hugging Face 根目录，其模型缓存位于下一级 `hub/`。默认路径分别为当前用户的 `~/.cache/huggingface` 和 `~/models`。重复运行安装脚本会按本次用户和环境重新写入目录及用户组映射，复用旧安装时应保持这些值一致。

### 手动安装

```bash
cp .env.example .env

# 安装 Compose 挂载所需的主机代理、密钥与 socket
sudo ./scripts/install-ops-agent.sh --apply

# 编辑 .env：替换密钥、密码、主机路径以及用户 / 用户组 ID
# DOCKER_GID 可用 stat -c '%g' /var/run/docker.sock 查询
# OPS_AGENT_GID 可用 getent group dgx-spark-ops 查询
mkdir -p data
chmod 600 .env

docker compose build
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:3000/api/health
```

需提前创建模型与缓存目录，并确保 `PUID` / `PGID` 对 `data/` 和下载目录有写权限。默认 Compose 使用主机网络、主机 PID 命名空间、GPU 和 Docker socket；推理容器由主机 Docker 创建，模型路径通过宿主机与管理器目录映射解析。

### 原生 systemd 安装

也可以将管理器运行在主机 Python 虚拟环境中，推理服务仍使用 Docker。此方式还需要 Python 3.11+、虚拟环境支持、Node.js 22+、Corepack 和用户级 systemd。

```bash
./scripts/install-native.sh
./scripts/install-native.sh --apply

systemctl --user status dgx-spark-web-manager.service
journalctl --user -u dgx-spark-web-manager.service -n 100 --no-pager
```

原生安装使用 `.venv-native/`、`data/native.env` 和用户级 `dgx-spark-web-manager.service`，当前脚本将服务绑定到 `0.0.0.0:3000`。与 Compose 安装不同，它不会自动安装 Host Agent；需要主机诊断和执行功能时，另行运行 `install-ops-agent.sh --apply`（需要 root 权限），为服务用户配置 `dgx-spark-ops` 组权限，并在 `data/native.env` 中将 `DGX_OPS_AGENT_KEY_FILE` 指向 `/etc/dgx-spark-manager/ops-agent.key`。组权限变更后需重新建立用户会话。

## 从下载模型到客户端调用

1. **准备模型。** 在“系统设置”保存可选的 Hugging Face Token，然后在“Hugging Face”搜索和下载；也可以扫描已有模型目录。下载和恢复进度在“任务中心”查看。
2. **创建部署。** 从“部署实例”选择模型、运行时和允许使用的镜像。管理器读取模型卡、本地配置、镜像能力和当前资源，给出附带来源与置信度的参数建议。
3. **选择加速方式。** 如模型与运行时支持，可选择 Draft Model、DSpark、DFlash 或嵌入式 MTP。以对应模型卡和镜像探测结果为依据，各模型不共用一套固定推测参数。
4. **审阅部署预览。** 核对启动命令、目录挂载、端口、内存估计和能力限制；修改表单后需重新预览。需要复核的候选项和资源提示会要求确认。
5. **等待健康检查与测速。** 部署任务在服务就绪后预热并测速。测速失败会记录原因，不会仅因此回滚已健康的服务。
6. **接入客户端。** 在“API 网关”创建 Key，设置客户端 Base URL 为 `http://<DGX-SPARK-IP>:3000/v1`，并从 `/v1/models` 返回结果中选择模型 ID。

主机端口留空时，从 `8000` 起选择最小可用端口，同时检查管理器预留和 Docker 端口绑定。已停止但未卸载的托管实例仍保留端口；克隆实例会分配新的端口。

编辑部署支持健康检查后的替换及失败回滚，任务历史保留。模型库删除任务会检查模型引用；需要先解除相关部署引用，再删除模型文件。

## 运行时与模型适配

| 运行时 | 当前适配内容 | 部署前需要确认 |
| --- | --- | --- |
| SGLang | 工具 / 思考解析器、DSpark、DFlash、MTP、总 Token 槽控制，以及专用 SSD Stream 配置 | 所选镜像实际支持该架构、量化格式和推测算法 |
| vLLM | 工具 / 思考解析器、Draft Model、部分模型的嵌入式 MTP，以及模型相关推测配置 | 镜像版本、模型权重和探测出的参数能力一致 |
| llama.cpp | GGUF、原生 ARM64/CUDA `llama-server`、上下文与并行槽配置、对应模板和 MTP 参数 | 主机已有兼容的 llama.cpp 运行文件；通用 CUDA 镜像本身不提供模型实现 |

镜像名称必须通过服务端允许列表。带 `dgx-local/` 前缀的镜像是本机构建镜像，安装管理器不会自动构建所有推理镜像。部分构建配方位于 [`deploy/images/`](deploy/images/)，允许列表与默认路径见 [`backend/app/config.py`](backend/app/config.py)。

### Qwen3.8、MiniCPM 与推测解码

- **Qwen3.8：** 使用仓库固定的 Qwen Fixed Chat Template v22.4，处理工具历史、思考模式和工具参数渲染。vLLM 使用 `qwen3_xml` / `qwen3`，常规 SGLang 使用 `qwen3_coder` / `qwen3`；SSD Stream 配置走其专用自动解析路径。已有部署的模板配置升级不会自动启动已停止模型，运行中实例在后续重启时应用。
- **MiniCPM5：** 运行时适配器可从模板标记识别 `minicpm5` 工具解析器；思考解析器同样按模板匹配。DSpark 需要兼容的 Draft checkpoint 与运行时镜像。模型名称出现在面板中，不代表任意旧版镜像都包含对应解析器。
- **Draft Model：** 校验主模型与 Draft 的兼容性，并合并估算资源。SGLang DSpark 支持从已挂载的 Hugging Face 缓存解析 Draft 仓库；vLLM 可对具有索引 `mtp.*` 权重的模型使用嵌入式 MTP，无需重复下载一份基础模型。
- **Qwen3.8 Flash Next SSD Stream：** 这是经过专门准备的 checkpoint 路径。管理器校验 `ssd-stream.json`、sidecar 文件和指定镜像，将 SSD 上的 PLE 数据从常驻权重估算中扣除，并配置插件、MTP、多模态和所需运行权限。普通 Flash、NVFP4 或 Hybrid FP8 仓库不会仅凭名称自动成为 SSD Stream 模型。

模板的来源、固定版本和许可证见 [`backend/app/chat_templates/`](backend/app/chat_templates/)。运行时支持详情见 [兼容性说明](docs/COMPATIBILITY.md)，最终应以当前代码生成的预览和镜像能力探测为准。

### 统一内存与缓存容量

DGX Spark 的系统与 GPU 共用统一内存。部署时应同时考虑模型权重、Draft、KV / Mamba 缓存、运行时工作区，以及主机和其他模型的占用。

| 配置 | 含义 |
| --- | --- |
| `context_length` | 配置的上下文长度，不等于运行时一定能分配出的缓存容量 |
| `max_concurrency` | 最大并发请求数；SGLang 映射为 `--max-running-requests` |
| `memory_fraction` | 运行时内存比例；SGLang 映射为 `--mem-fraction-static` |
| `max_total_tokens` | SGLang“运行时总 Token 槽”，用于限制整个实例的 Token 池；当前校验要求不超过 `context_length` |
| `generation_defaults.max_tokens` | 单次生成的默认输出上限，与上下文长度、实例总缓存容量不同 |

资源估算默认预留总内存的 10%，且至少 8 GiB。**这是部署预检的估算余量，不是操作系统层面的硬隔离。** 降低并发、内存比例或总 Token 槽后，需要通过部署更新让运行时重新分配资源。

面板按实例展示实际内存使用，优先使用 NVIDIA 计算进程数据，并在必要时采用有边界的 Docker 内存回退。模型发现接口同时区分配置上下文与运行时有效容量，客户端应读取当前健康路由的元数据，而不是直接照搬模型卡最大长度。

### TPS 测速口径

内置测速先执行短预热，再发送固定的数字序列任务，最多生成 `256` tokens。结果按 **`completion_tokens / 整次测速请求耗时`** 计算，因此包含请求和预填充等开销，不等同于仅统计解码阶段的 tok/s。

最新成功结果保存到部署和模型资产；卸载部署后，模型库仍可保留该模型的最近成功结果。不同上下文、思考设置、并发量、缓存状态和推测参数会影响读数，内置 TPS 也不代表回答质量评分。

## OpenAI 兼容 API

网关支持以下接口；具体能力由健康的运行实例决定：

| 接口 | 用途 |
| --- | --- |
| `GET /v1/models` | 获取运行中且健康的模型路由、上下文、输出上限和模态信息；配置上游网关后一并列出上游模型 |
| `GET /v1/models/{model}` | 获取指定路由的信息 |
| `POST /v1/chat/completions` | 聊天、工具调用和 SSE 流式响应 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | 向支持嵌入能力的实例发送请求 |

网关 Key 在创建时返回原文，数据库仅保存哈希；它与 Hugging Face Token、在线 AI 服务密钥分别管理。

本机没有的模型可以转发到上游 OpenAI 兼容网关：在「API 网关」页配置 Base URL 与密钥后，
`/v1/models` 会同时列出上游模型（标记 `dgx_source: upstream`），本机同名路由优先；上游不可用时
本地模型列表不受影响。未配置上游时，请求未知模型仍返回 404。请求显式提供的生成参数优先，部署保存的默认值仅填充未提供且运行时支持的字段。

### Python 流式调用

```bash
python -m pip install openai
export DGX_BASE_URL='http://<DGX-SPARK-IP>:3000/v1'
export DGX_API_KEY='替换为面板创建的网关密钥'
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["DGX_BASE_URL"],
    api_key=os.environ["DGX_API_KEY"],
)

models = client.models.list().data
if not models:
    raise RuntimeError("没有健康的模型路由，请先在面板启动部署")

# 实际接入时可改成列表中指定的聊天模型 ID
model_id = models[0].id
for chunk in client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "请用中文介绍你能完成的任务。"}],
    stream=True,
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### 工具调用与多模态

工具调用使用 OpenAI `tools` / `tool_calls` 协议。工具实际执行由客户端负责，网关负责路由和协议兼容。流式路径逐事件增量输出，长请求提供 15 秒 SSE 保活，上游读取超时为 30 分钟。

图像和视频需要模型、处理器资产与运行时同时支持。网关接受 OpenAI `image_url`、运行时兼容的 `video_url` 和 data URL，并规范化常见 `input_image` / `input_video` 别名；文本专用路由不会接收媒体请求。可通过 `/v1/models` 的 `input_modalities` 确认已声明能力。

更多调用示例见 [`examples/`](examples/)，管理与推理接口说明见 [`docs/API.md`](docs/API.md)。

## AI 运维助手

在“在线 AI 服务”中配置远程 OpenAI 兼容服务的 Base URL、API Key、默认模型和超时。项目的运维工作流面向 DeepSeek 等远程服务，本地推理实例作为诊断对象。连接测试分别检查服务可达性、模型列表与默认模型结构化响应。

运维助手可读取管理器库存、Docker 状态、GPU / 内存、端口、日志、任务和网关指标。涉及主机 Shell 的操作会形成持久化计划，显示具体命令、工作目录、超时、影响和回滚方式，由管理员在面板批准后交给 Host Agent 执行。

Host Agent 通过本机 Unix socket 与签名请求通信，安装位置包括：

| 路径 | 内容 |
| --- | --- |
| `/run/dgx-spark-manager/ops-agent.sock` | 通信 socket |
| `/etc/dgx-spark-manager/ops-agent.key` | 主机代理认证密钥 |
| `/usr/local/lib/dgx-spark-ops-agent/` | 代理代码 |
| `/var/lib/dgx-spark-ops-agent/jobs/` | 执行任务数据 |

## 配置与数据目录

Compose 的常用配置位于 `.env`，完整示例见 [`.env.example`](.env.example)：

| 配置项 | 用途 / 默认行为 |
| --- | --- |
| `DGX_SECRET_KEY` | 会话签名及已保存凭据的加密基础，至少 32 字符；恢复数据时应保留原值 |
| `DGX_ADMIN_USERNAME` / `DGX_ADMIN_PASSWORD` | 管理员账号，默认用户名 `admin`，密码至少 12 字符 |
| `DGX_LISTEN_HOST` / `DGX_LISTEN_PORT` | Compose 服务监听地址，默认 `0.0.0.0:3000` |
| `DGX_ALLOWED_ORIGINS` | 允许的浏览器来源，应与实际访问地址一致 |
| `DGX_COOKIE_SECURE` | HTTPS 部署可设为 `true` |
| `HF_HOME_HOST` / `MODEL_HOME_HOST` | 宿主机 Hugging Face 根目录与模型目录 |
| `LLAMA_CPP_HOME_HOST` | llama.cpp 主机目录，默认 `/opt/llamacpp`，在管理器内只读挂载 |
| `PUID` / `PGID` | 管理器进程用户与用户组 |
| `DOCKER_GID` / `OPS_AGENT_GID` | Docker socket 和 Host Agent 访问组 |
| `DGX_DEPLOYMENT_STARTUP_TIMEOUT_SECONDS` | 模型部署启动等待时间，默认 `1200` 秒 |
| `DGX_FALLBACK_BASE_URL` / `DGX_FALLBACK_API_KEY` | 上游 OpenAI 兼容网关的默认值；通常直接在面板「API 网关」页配置，面板值优先且无需重启容器 |
| `DGX_UPSTREAM_MODELS_CACHE_SECONDS` | 上游模型列表缓存时间，默认 `30` 秒 |
| `DGX_GATEWAY_THROUGHPUT_WINDOW_SECONDS` | API 网关 Token 吞吐的采样窗口，默认 `300` 秒；面板标签会显示该窗口 |

Compose 数据库位于 `./data/manager.db`，模型文件保存在配置的宿主机目录。其他后端设置见 [`backend/app/config.py`](backend/app/config.py)；新增环境配置时，也需要将其显式传给 Compose 服务，不能仅假定写入 `.env` 就会进入容器。

## 更新、备份与卸载

### 更新 Compose 安装

```bash
# 在仓库目录执行；先备份，再同步并重建
./scripts/backup.sh
git pull --ff-only
sudo ./scripts/install-ops-agent.sh --apply
./scripts/update.sh
curl -fsS http://127.0.0.1:3000/api/health
```

`update.sh` 负责拉取构建基础镜像、重新构建和启动管理器，**不会执行 `git pull`，也不会自动备份数据库或更新 Host Agent**。管理器容器启动时执行 Alembic 数据库迁移。

### 备份与恢复

```bash
./scripts/backup.sh
./scripts/restore.sh backups/dgx-manager-YYYYMMDDTHHMMSSZ.tar.gz
```

Compose 备份通过 SQLite backup API 创建一致性副本，并将副本与 `.env` 打包；不包含模型权重、Docker 镜像和 Host Agent 的密钥或任务数据。恢复脚本会停止管理器、还原数据库与环境文件，再启动服务。

原生安装使用独立更新流程，它会先备份 SQLite 数据库：

```bash
git pull --ff-only
./scripts/update-native.sh
```

原生安装的 `data/native.env` 需另行保管；Compose 的备份与恢复脚本不适用于该安装方式。

### 卸载

```bash
# Compose：立即停止并移除管理器，保留数据和模型
./scripts/uninstall.sh

# 原生安装：先预览，再执行
./scripts/uninstall-native.sh
./scripts/uninstall-native.sh --apply

# 单独卸载 Host Agent：先预览，再执行
./scripts/uninstall-ops-agent.sh
sudo ./scripts/uninstall-ops-agent.sh --apply
```

需要一并删除管理器数据库与审计数据时，Compose 使用 `uninstall.sh --purge`；原生安装使用 `uninstall-native.sh --apply --purge`。模型目录仍保留，推理实例和 Docker 镜像也不会因为卸载管理器而统一清理。

## 常见问题

**面板能打开，客户端却看不到模型。** `/v1/models` 只发布运行且健康的路由。检查实例状态、启动日志、API Key 和客户端 Base URL 中的 `/v1`，模型 ID 使用接口返回值或配置的路由别名。

**上下文设得很大，实际容量较小。** 配置上限、运行时可分配 Token 池和单次输出上限是不同指标。检查 SGLang 启动结果、总 Token 槽和网关公布的有效上下文；增加上限本身不会增加统一内存。

**模型思考或执行工具后不再输出。** 分别检查客户端的工具执行结果、`finish_reason`、实际输出上限及容器日志。模型可能返回正常 `stop` 但没有有效回答，客户端也可能遇到工具命令错误；不能仅凭界面停止判断为 OOM。网关兼容重试有触发条件和次数限制。

**删除模型后磁盘仍然占用很多。** 模型文件、Hugging Face 下载缓存、Docker 镜像层、构建缓存和容器日志是不同存储来源。删除模型任务不等于清理全部 Docker 镜像和构建缓存；可先用 `df -h`、`docker system df -v` 及任务结果核对实际占用。

**AI 运维提示 Host Agent 不可用。** 检查 `systemctl status dgx-spark-ops-agent.socket`、socket 挂载、密钥文件与访问组；原生安装还需要正确设置主机密钥路径。

更多排查步骤见 [故障排查文档](docs/TROUBLESHOOTING.md)。

## 本地开发

后端要求 Python 3.11+；容器构建使用 Python 3.12。以下命令适用于 Linux 开发环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export DGX_SECRET_KEY='development-secret-key-change-before-production'
export DGX_ADMIN_PASSWORD='Development-password-1234'
mkdir -p data
alembic upgrade head

pytest backend/tests -q
ruff check backend/app backend/tests
uvicorn app.main:app --app-dir backend --reload --port 3000
```

前端使用 Node.js 22+，pnpm 版本由 `frontend/package.json` 固定：

```bash
cd frontend
corepack enable
pnpm install --frozen-lockfile
pnpm test
pnpm lint
pnpm build
pnpm dev
```

Vite 默认监听 `5173`，将 `/api` 和 `/v1` 代理到 `127.0.0.1:3000`。Docker 和 GPU 相关集成功能需要相应主机环境。

```text
backend/app/        API、部署服务、运行时适配和网关
backend/tests/      后端测试
backend/migrations/ 数据库迁移
frontend/src/       中文管理界面
host_agent/         主机诊断与计划执行代理
deploy/             systemd 单元和专用镜像配方
scripts/            安装、更新、备份、恢复和卸载脚本
examples/           API 调用示例
docs/               架构、接口、兼容性和排障说明
```

## 文档与许可证

- [架构说明](docs/ARCHITECTURE.md)
- [API 说明](docs/API.md)
- [兼容性说明](docs/COMPATIBILITY.md)
- [故障排查](docs/TROUBLESHOOTING.md)
- [产品设计](PRODUCT.md)
- [界面设计](DESIGN.md)

项目使用 [MIT 许可证](LICENSE)。随仓库分发的第三方聊天模板保留其独立许可证，详见模板目录。
