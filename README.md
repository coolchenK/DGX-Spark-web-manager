# DGX Spark Web Manager

面向 NVIDIA DGX Spark 的中文模型部署与运维面板。在一个界面中查看主机资源、下载 Hugging Face 模型、管理推理容器，并通过统一的 OpenAI 兼容接口连接聊天客户端和开发工具。

本 README 对应 2026-10-19 的源码。项目面向 ARM64 / GB10 环境，支持 **SGLang、vLLM 和 llama.cpp**，并提供专用生图与 Laya 运行时适配。管理服务使用 FastAPI、SQLite 和 React 19，界面提供桌面与移动端布局，以及浅色、深色和跟随系统主题。

## 功能概览

| 页面 | 主要功能 |
| --- | --- |
| 系统概览 | 查看 CPU、统一内存、磁盘、GPU、温度、功耗和驱动状态；推理服务仅展示运行中的实例 |
| Hugging Face | 搜索模型、查看兼容性与模型卡、下载文件，支持任务暂停、恢复和取消 |
| 模型库 | 扫描本地模型与 Hugging Face 缓存，查看模型能力、历史测速结果和删除任务 |
| 部署实例 | 创建、预览、启动、停止、重启、编辑、克隆和卸载部署，查看日志、内存和 TPS |
| API 网关与 AI 服务 | 在一个 Magpie 界面管理 API Key、服务商、模型路由和调用记录；保留旧访问密钥的查看与吊销 |
| 图片生成 | 文生图、1–3 张参考图编辑、历史结果与下载，选中满意的结果后进行 Real-ESRGAN / SUPIR 超分 |
| AI 运维助手 | 根据主机、容器、日志和任务信息诊断问题，生成可审阅的执行计划 |
| 任务中心 / 日志与审计 | 跟踪持久化任务、执行结果与管理操作 |
| 系统设置 | 管理 Hugging Face 凭据等系统配置 |

管理器启动时会发现已有推理容器，不会因此重建或重启它们。外部容器标记为非托管实例；卸载外部实例需要确认当前容器名。停止或卸载实例与永久删除模型文件是不同操作。

原“API 网关”和“在线 AI 服务”已合并：导航只保留 **API 网关与 AI 服务**，旧 `/providers` 地址自动跳转到 `/gateway`。日志和任务时间按浏览器本地时区显示。

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

浏览器打开 `http://<DGX-SPARK-IP>:3000`。登录后可以先查看已有模型和部署，再配置下载或创建新实例。**该脚本只安装管理器与 Host Agent；合并后的网关页面还需要下面的 Magpie 接入配置。** 模型权重、专用推理镜像和生图运行时需要另外准备。

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

## 接入 Magpie

Magpie 是独立服务，管理器通过后端代理嵌入其管理页，并把公共 `/v1` 请求连接到它。安装来源和上游说明见 [Magpie 官网](https://usemagpie.ai/zh/) 与 [官方发行页](https://github.com/yetone/magpie-releases)。本仓库包含接入代码，**不包含 Magpie 二进制、安装器或自动更新服务**。

### 服务与凭据

1. 在同一主机安装支持 ARM64 的 Magpie，使用独立服务账户并持久化其配置。当前适配固定连接 API `127.0.0.1:3425`、管理页 `127.0.0.1:3430`，管理器 Compose 已使用主机网络。
2. 在 Magpie 配置固定的 `MAGPIE_WEB_KEY`（至少 16 个 URL 安全字符），开启网关的局域网共享以接受代理请求；`MAGPIE_TRUST_PROXY` 保持关闭，让 Magpie 验证每个外部 API Key。两个监听端口只需向本机开放，公网入口走管理器。
3. 在 Magpie 创建专用网关 Key，供旧 Key 兼容转发和运维助手使用；再生成一个独立随机密钥作为本地模型桥接凭据。它们都不是客户端应共用的密钥。
4. 将下表配置写入管理器 `.env`（原生安装写入 `data/native.env`），重建或重启管理器使配置生效。不要把真实密钥写入 Git。

| 管理器变量 | 配置内容 |
| --- | --- |
| `DGX_MAGPIE_ENABLED` | `true` |
| `DGX_MAGPIE_WEB_KEY` | 与 Magpie 的 `MAGPIE_WEB_KEY` 一致 |
| `DGX_MAGPIE_GATEWAY_KEY` | 上一步创建的专用 Magpie 网关 Key |
| `DGX_MAGPIE_BRIDGE_KEY` | 独立随机桥接凭据 |
| `DGX_FALLBACK_BASE_URL` | `http://127.0.0.1:3425/v1`，用于旧 Key 的上游模型发现 |
| `DGX_FALLBACK_API_KEY` | 与 `DGX_MAGPIE_GATEWAY_KEY` 一致 |

已通过管理 API 保存的上游配置优先于环境变量；升级旧安装时需同时核对 `/api/gateway/upstream`，详见 [API 文档](docs/API.md)。

### 注册本地模型

在 Magpie 创建 OpenAI 兼容服务商，**服务商 ID 固定为 `dgx-local`**：

| 项目 | 默认端口下的配置 |
| --- | --- |
| Base URL | `http://127.0.0.1:3000/_magpie/local/v1` |
| API Key | `DGX_MAGPIE_BRIDGE_KEY` 的值 |
| 模型目录 | 从此 Base URL 的 `/models` 拉取，部署启停后刷新 |
| 模型 ID 显示 | 使用带服务商前缀的 ID，关闭裸模型 ID（`bareModelIDs=false`），避免重复目录 |

管理器监听端口改变时，同步修改桥接 URL。该桥接仅访问本地健康实例，不能指回公共 `/v1`，否则会形成循环转发。当前面板接入不会安装定时刷新任务，目录刷新频率按 Magpie 配置或外部调度决定。

完成后登录面板，进入 **API 网关与 AI 服务**。新增对外 Key 在 Magpie 的网关密钥页面创建；远程厂商 API Key 在 Magpie 服务商页面配置。下方“现有访问密钥”用于查看、吊销旧 `dgx_` Key；旧客户端可以继续使用原地址和 Key。旧 Key 的 Magpie 调用记录归到兼容转发 Key，新的 Magpie Key 可分别统计。

`DGX_MAGPIE_ENABLED=false` 时原生后端网关仍可使用，但合并页面会显示 Magpie 未连接，不会恢复旧的两个管理页面。独立的第三方插件沙箱页面不通过面板管理代理开放。升级 Magpie 后应复核登录、模型目录、创建 / 吊销 Key 和流式调用；其管理页面变动可能需要调整代理适配。

## 从下载模型到客户端调用

1. **准备模型。** 在“系统设置”保存可选的 Hugging Face Token，然后在“Hugging Face”搜索和下载；也可以扫描已有模型目录。aria2 下载最多同时处理 3 个文件、每文件 16 连接，保留分片控制文件以断点续传；实际连接数和速度取决于下载源。下载和恢复进度在“任务中心”查看。
2. **创建部署。** 从“部署实例”选择模型、运行时和允许使用的镜像。管理器读取模型卡、本地配置、镜像能力和当前资源，给出附带来源与置信度的参数建议。
3. **选择加速方式。** 如模型与运行时支持，可选择 Draft Model、DSpark、DFlash 或嵌入式 MTP。以对应模型卡和镜像探测结果为依据，各模型不共用一套固定推测参数。开启选项不等于已有加速收益，应结合运行时日志和投机接受率验证。
4. **审阅部署预览。** 核对启动命令、目录挂载、端口、内存估计和能力限制；修改表单后需重新预览。需要复核的候选项和资源提示会要求确认。
5. **等待健康检查与测速。** 部署任务在服务就绪后预热并测速。测速失败会记录原因，不会仅因此回滚已健康的服务。
6. **接入客户端。** 在“API 网关与 AI 服务”的 Magpie 中创建 Key，设置客户端 Base URL 为 `http://<DGX-SPARK-IP>:3000/v1`，并从 `/v1/models` 返回结果中选择模型 ID。

主机端口留空时，从 `8000` 起选择最小可用端口，同时检查管理器预留和 Docker 端口绑定。已停止但未卸载的托管实例仍保留端口；克隆实例会分配新的端口。

编辑部署支持健康检查后的替换及失败回滚，任务历史保留。模型库删除任务会检查模型引用；需要先解除相关部署引用，再删除模型文件。

## 运行时与模型适配

| 运行时 | 当前适配内容 | 部署前需要确认 |
| --- | --- | --- |
| SGLang | 工具 / 思考解析器、DSpark、DFlash、MTP、总 Token 槽控制，以及专用 SSD Stream 配置 | 所选镜像实际支持该架构、量化格式和推测算法 |
| vLLM | 工具 / 思考解析器、Draft Model、部分模型的嵌入式 MTP，以及模型相关推测配置 | 镜像版本、模型权重和探测出的参数能力一致 |
| llama.cpp | GGUF、原生 ARM64/CUDA `llama-server`、上下文与并行槽配置、对应模板和 MTP 参数 | 主机已有兼容的 llama.cpp 运行文件；通用 CUDA 镜像本身不提供模型实现 |

另有 `viggle`（Qwen-Image-2.1 Viggle Turbo）和 `laya` 专用适配器，它们使用各自的接口，不是通用聊天运行时。

镜像名称必须通过服务端允许列表。带 `dgx-local/` 前缀的镜像是本机构建镜像，安装管理器不会自动构建所有推理镜像。部分构建配方位于 [`deploy/images/`](deploy/images/)（含 [Laya](deploy/images/laya/README.md)），允许列表与默认路径见 [`backend/app/config.py`](backend/app/config.py)。

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
| `max_concurrency` | 调度并发上限；SGLang 映射为 `--max-running-requests`，不保证每个会话同时使用完整上下文 |
| `memory_fraction` | 运行时内存比例；SGLang 映射为 `--mem-fraction-static` |
| `max_total_tokens` | SGLang“运行时总 Token 槽”，用于限制整个实例的 Token 池；当前校验要求不超过 `context_length` |
| `generation_defaults.max_tokens` | 单次生成的默认输出上限，与上下文长度、实例总缓存容量不同 |
| `disable_cuda_graph` | SGLang 可关闭 CUDA Graph 来减少捕获开销；可能影响性能，需按模型实测 |

共享 Token 池或显式 KV 预算由多个会话共用；请求输入、工具定义和预留输出均计入上下文。配置 256K / 并发 5 并不代表能够同时容纳五个 256K 请求。部分 Flash Next 专用配置还使用只读 PLE 文件映射和固定 KV 预算，必须与对应镜像及 checkpoint 匹配。

资源估算默认预留总内存的 10%，且至少 8 GiB。**这是部署预检的估算余量，不是操作系统层面的硬隔离。** 降低并发、内存比例或总 Token 槽后，需要通过部署更新让运行时重新分配资源。

面板按实例展示实际内存使用，优先使用 NVIDIA 计算进程数据，并在必要时采用有边界的 Docker 内存回退。模型发现接口同时区分配置上下文与运行时有效容量，客户端应读取当前健康路由的元数据，而不是直接照搬模型卡最大长度。

### TPS 测速口径

内置测速先执行短预热，再发送固定的**流式**数字序列任务，最多生成 `256` tokens。TPS 按 **`(总输出 tokens − 首块 tokens) / (末块时间 − 首块时间)`** 计算，排除首 token 之前的排队和预填充时间。缺少逐块 usage 时，按首块 1 token 估算，并在任务结果标记 `stream_decode_estimate`；投机解码的大块输出会影响估算精度。

任务结果另外记录首 token 延迟和整次请求吞吐。未结束、缺少 usage 或没有有效解码区间的流不会保存为成功测速。`/v1/models` 中本地模型的 `tokensPerSecond` 是**最近成功测速值，不是实时速度**；Magpie 原生目录是否展示这些扩展字段取决于其版本。

最新成功结果保存到部署和模型资产；卸载部署后，模型库仍可保留该模型的最近成功结果。不同上下文、思考设置、并发量、缓存状态和推测参数会影响读数，内置 TPS 也不代表回答质量评分。

## OpenAI 兼容 API

网关支持以下接口；具体能力由健康的运行实例决定：

| 接口 | 用途 |
| --- | --- |
| `GET /v1/models` | 获取运行中且健康的模型路由、上下文、输出上限和模态信息；配置上游网关后一并列出上游模型 |
| `GET /v1/models/{model}` | 获取指定路由的信息 |
| `POST /v1/chat/completions` | 聊天、工具调用和 SSE 流式响应 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/responses` | Responses 请求，经 Magpie 或本地适配处理 |
| `POST /v1/embeddings` | 向支持嵌入能力的实例发送请求 |

启用 Magpie 后，原生 Magpie Key 使用它的目录和路由，模型 ID 通常为 `dgx-local/<本地路由名>` 或其他服务商前缀。旧 `dgx_` Key 保留管理器的本地上下文、输出上限和测速元数据；与上游合并目录时会去掉同一本地模型的重复前缀条目。以调用 `/v1/models` 实际返回的 ID 为准，不要手工拼接显示名称。新 Key 的权限、配额和模型开放范围在 Magpie 管理。

旧网关 Key 创建时只返回一次原文，管理器数据库仅保存哈希；Magpie 原生 Key 由 Magpie 管理。它们与 Hugging Face Token、服务商密钥、内部桥接凭据分别管理。

对于已识别的原生 Flash Next（`Qwen4ExpForConditionalGeneration`）和 Qwen Fixed 模板，本地模型目录公布 `reasoning_levels` / `supported_reasoning_levels` 为 `low / medium / xhigh`；`high` 兼容映射到 `xhigh`。Chat Completions 与 Responses 均保留这些档位，`enable_thinking` 不会把此类模型降为只有开关。Magpie 可用 `magpie model efforts dgx-local/<模型ID> low,medium,xhigh` 保存相同档位；更新后需让客户端重新发现模型。此设置不适用于只有思考开关的模型。

本地聊天 / 补全请求的 **采样参数以部署保存的默认值为准**：客户端传入的 temperature、top-p、top-k、min-p、重复惩罚等采样字段会被丢弃。客户端的输出上限、stop 和思考选项保留原有处理逻辑；这些字段仍受模型与运行时能力限制。远程模型按对应服务商规则处理。

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

工具调用使用 OpenAI `tools` / `tool_calls` 协议。工具实际执行由客户端负责，网关负责路由和协议兼容。本地桥接的流式路径提供 15 秒 SSE 保活，读取超时为 30 分钟；Magpie 代理层读取超时为 10 分钟，外部反向代理也可能有自己的限制。

图像和视频需要模型、处理器资产与运行时同时支持。网关接受 OpenAI `image_url`、运行时兼容的 `video_url` 和 data URL，并规范化常见 `input_image` / `input_video` 别名；文本专用路由不会接收媒体请求。可通过 `/v1/models` 的 `input_modalities` 确认已声明能力。

更多调用示例见 [`examples/`](examples/)，管理与推理接口说明见 [`docs/API.md`](docs/API.md)。

## 图片生成与超分

“图片生成”页面对应 `viggle` 运行时。先准备 Qwen-Image-2.1 基础权重、Viggle Turbo v0.2.1 LoRA / 调度器以及兼容的专用镜像，再从“部署实例”启动服务。当前适配固定基础 revision，详见 [`image_schema.py`](backend/app/image_schema.py) 与 [`viggle.py`](backend/app/runtime/viggle.py)。**仓库仅包含面板、请求协议和运行时适配；`/opt/viggle/server.py`、量化编码器、Real-ESRGAN / SUPIR 权重及执行环境需由专用镜像提供，通用管理器镜像无法替代。**

- 文生图支持提示词、1024 / 1536 / 2048 分辨率档、宽高比、自定义尺寸、4 / 6 / 8 步和 seed。
- 图生图支持上传 1–3 张 PNG / JPEG / WebP，每张最多 16 MiB；编辑使用 1024 或 1536 档。
- 满意的历史结果可以再执行 Real-ESRGAN 快速超分或 SUPIR-F 细节重建，目标长边为 2048 / 4096；可调分块参数，SUPIR 另有步数、CFG 与约束强度。结果另存，保留原图。
- 当前队列最多接收 4 个待处理或运行中的图片任务，使用独立任务通道；输入和结果保存于 `data/image-studio/`。

超分依赖对应算法和权重，生成式细节可能改变原图，不能保证无损还原。内存拒绝启动提示来自图片运行时；其模型加载策略和系统余量与管理器的部署预检是两套配置。并存多个模型前要按实际任务测量峰值，单看空闲内存无法保证 4K 超分可运行。

## AI 运维助手

在“API 网关与 AI 服务”的 Magpie 中配置远程服务商。启用 Magpie 后，部署建议与运维助手会同步已启用的聊天模型，调用经 Magpie 转发，不在管理器中额外保存一份厂商密钥。本地 `dgx-local` 服务商不加入这份远程诊断服务列表。连接测试分别检查服务可达性、模型列表与默认模型结构化响应。

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
| `DGX_MAGPIE_*` | 合并网关页面与本地桥接配置，见上方 Magpie 接入说明 |
| `DGX_FALLBACK_BASE_URL` / `DGX_FALLBACK_API_KEY` | 旧网关上游默认值；通过管理 API 保存的配置优先 |
| `DGX_UPSTREAM_MODELS_CACHE_SECONDS` | 上游模型列表缓存时间，默认 `30` 秒 |
| `DGX_GATEWAY_THROUGHPUT_WINDOW_SECONDS` | 原生网关统计 API 的 Token 吞吐窗口，默认 `300` 秒；与模型 decode TPS 不是同一指标 |

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

Compose 备份通过 SQLite backup API 创建一致性副本，并将副本与 `.env` 打包；不包含模型权重、Docker 镜像、图片输入 / 结果、Magpie 配置和 Host Agent 的密钥或任务数据。这些目录和凭据需另行备份，尤其是 `data/image-studio/` 与独立 Magpie 的持久化目录。恢复脚本会停止管理器、还原数据库与环境文件，再启动服务。

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

**面板能打开，客户端却看不到模型。** 本地桥接只发布运行且健康的路由；Magpie 原生目录可能还需刷新。检查实例状态、网关 Key 的模型权限、`dgx-local` 的桥接地址和客户端 Base URL 中的 `/v1`，模型 ID 使用接口返回值。

**在哪里创建对外 API Key？** 进入“API 网关与 AI 服务”内的 Magpie 网关密钥管理。页面下方“现有访问密钥”只维护旧 Key；不要向客户端分发内部桥接凭据。

**合并页面提示 Magpie 加载失败。** 先确认独立 Magpie 服务已启动、3425 / 3430 端口可达，管理器启用了 `DGX_MAGPIE_ENABLED` 且 Web Key 一致。单独运行管理器安装脚本不会安装 Magpie。

**上下文设得很大，实际容量较小。** 配置上限、运行时可分配 Token 池和单次输出上限是不同指标。检查 SGLang 启动结果、总 Token 槽和网关公布的有效上下文；增加上限本身不会增加统一内存。

**模型思考或执行工具后不再输出。** 分别检查客户端的工具执行结果、`finish_reason`、实际输出上限及容器日志。模型可能返回正常 `stop` 但没有有效回答，客户端也可能遇到工具命令错误；不能仅凭界面停止判断为 OOM。网关不会通过语义重试自动改写回答。

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
ruff check backend host_agent
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

Vite 默认监听 `5173`，将 `/api` 和 `/v1` 代理到 `127.0.0.1:3000`。调试嵌入式 Magpie 时使用后端提供的构建页面（`pnpm build` 后打开 3000 端口），保证同源认证和代理路径一致。Docker 和 GPU 相关集成功能需要相应主机环境。

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
