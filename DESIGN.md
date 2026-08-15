# DGX Spark Web Manager Design System

## Direction

工作场景是设备管理员在办公室显示器或机房手机上长时间查看一台高价值计算设备。采用 restrained 色彩策略、低动效和中高信息密度；Ant Design 提供熟悉的产品交互，定制主题负责建立冷静、硬件导向的识别度。

## Theme

- 亮色背景使用纯白，主要内容面为中性浅灰。
- 深色背景使用无色相近黑，面板通过明度而非蓝灰色调分层。
- 品牌锚点采用低饱和黄绿色，来自硬件状态灯和终端磷光的联想，但仅用于主操作与当前选择。
- 青色仅用于信息状态；成功、警告、错误使用语义色，不作为装饰色。

```css
:root {
  --color-bg: oklch(1 0 0);
  --color-surface: oklch(0.97 0.004 110);
  --color-surface-raised: oklch(0.985 0.002 110);
  --color-ink: oklch(0.23 0.018 110);
  --color-muted: oklch(0.50 0.014 110);
  --color-primary: oklch(0.53 0.12 110);
  --color-primary-hover: oklch(0.46 0.13 110);
  --color-info: oklch(0.55 0.13 210);
}

[data-theme="dark"] {
  --color-bg: oklch(0.10 0 0);
  --color-surface: oklch(0.15 0.006 110);
  --color-surface-raised: oklch(0.19 0.008 110);
  --color-ink: oklch(0.93 0.008 110);
  --color-muted: oklch(0.70 0.012 110);
  --color-primary: oklch(0.75 0.09 110);
  --color-primary-hover: oklch(0.82 0.10 110);
  --color-info: oklch(0.76 0.11 210);
}
```

## Typography

- UI 字体：`Inter, ui-sans-serif, system-ui, sans-serif`，中文回退到系统无衬线字体。
- 数据与日志：`JetBrains Mono, SFMono-Regular, Consolas, monospace`。
- 页面标题 24px/32px，区块标题 16px/24px，正文 14px/22px，紧凑标签 12px/18px。
- 字距固定为 0；不使用随视口变化的字号。

## Shape And Elevation

- 页面区块不做浮动卡片；通过分隔线、留白和浅色面分组。
- 独立指标和重复对象可使用 6px 圆角卡片。
- 输入框和按钮使用 6px 圆角，标签可使用胶囊形。
- 阴影只用于 Drawer、Dropdown 和 Modal 等覆盖层。

## Layout

- 桌面端：224px 侧栏、56px 顶栏、内容最大宽度 1600px。
- 平板端：折叠侧栏，仅保留图标。
- 手机端：顶部栏加 Drawer，数据表切换为摘要列表。
- 固定格式组件使用稳定的网格轨道和最小高度，加载状态不能导致布局跳动。

## Motion

- 状态切换 160-220ms，使用 ease-out。
- 仅为 Drawer、折叠、任务状态和操作反馈添加动效。
- `prefers-reduced-motion` 下取消位移和连续动画。

## Components

- `StatusBadge`：图标、文本和语义色共同表达状态。
- `MetricStrip`：无嵌套卡片的紧凑资源指标带。
- `ResponsiveDataView`：桌面 Table，移动端 List。
- `TaskProgress`：确定或不确定进度、速度、剩余量和实时日志。
- `ApprovalPanel`：展示 AI 计划、影响、命令等价描述和回滚动作。
- `LogViewer`：等宽字体、行号、过滤、暂停自动滚动和下载。

## Content

- 使用简洁中文动词：下载、部署、停止、重启、批准、拒绝。
- 不在页面中放置功能宣传或使用教程；解释通过 Tooltip、空状态和具体错误完成。
- 危险确认必须写清对象名称与影响，不使用泛化的“确定吗”。
