# DGX Spark Web Manager Design System

## Direction

工作场景是设备管理员在办公室显示器或机房手机上长时间查看一台高价值计算设备。采用暖黑底 + 品牌橙点缀的奢华金融风（参考 Xapo Bank），配合玻璃拟态与克制的动效；Ant Design 提供熟悉的产品交互，定制主题负责建立冷静、硬件导向的识别度。信息密度保持中高，装饰不遮蔽状态。

**实现以 `frontend/src/styles.css` 为准**，本文件记录设计意图与令牌；两者不一致时先修文档。

## Theme

- 深色为主题默认值（`ThemeMode` 默认 `dark`），底色使用暖黑而非纯黑或蓝黑。
- 品牌橙仅用于主操作、当前选择与关键数值；不用于大面积填充。
- 玻璃拟态只用于承载层级（顶栏、登录面板、粘性操作条），不用于表格与正文容器。
- 成功、警告、错误使用语义色，不作为装饰色。

```css
:root {
  /* 浅色：暖白奢华 */
  --color-bg: #f9f5f1;
  --color-surface: #ffffff;
  --color-surface-raised: #fffdfb;
  --color-sidebar: #f2ece5;
  --color-ink: #221d1b;
  --color-muted: #7c7c7c;
  --color-border: #eae0d7;
  --color-primary: #c34813;
  --color-primary-strong: #ff4c00;
  --color-primary-deep: #a53505;
  --color-primary-soft: #f7e3d6;
  --color-info: #3080ff;
  --color-success: #05a352;
  --color-error: #e02b35;
  --color-warning: #d97706;
  --color-code-bg: #221d1b;
  --color-code-ink: #f3f3f3;
}

[data-theme="dark"] {
  /* 深色：暖黑奢华 */
  --color-bg: #090402;
  --color-surface: #171311;
  --color-surface-raised: #221d1b;
  --color-sidebar: #0c0806;
  --color-ink: #f9f5f1;
  --color-muted: #878787;
  --color-border: #242424;
  --color-primary: #c34813;
  --color-primary-strong: #ff4c00;
  --color-primary-deep: #a53505;
  --color-primary-soft: rgba(195, 72, 19, 0.16);
  --color-info: #3080ff;
  --color-success: #05df72;
  --color-error: #fb2c36;
  --color-warning: #ff6b00;
  --color-code-bg: #0d0a08;
  --color-code-ink: #eae0d7;
}
```

品牌橙同时通过 Ant Design token 下发（`colorPrimary` `#c34813`），保证组件库内部状态与自定义样式一致。

## Typography

- UI 字体：`Inter, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif`。
- 数据与日志：`"JetBrains Mono", ui-monospace, Consolas, monospace`；等宽同时用于 eyebrow 小标签。
- 页面标题 `clamp(26px, 3vw, 34px)`，字重 600，字距 `-0.8px`；手机端固定 24px。
- 区块标题 15px/22px 字距 `-0.2px`；正文 14px/22px；紧凑标签 11-12px。
- 大数值（指标、流量带）使用 20-22px 与 `font-variant-numeric: tabular-nums`，字距 `-0.4px` 至 `-0.5px`。
- 全站等宽小标签采用大写与 0.6-2.5px 字距，用于 eyebrow 与来源标注。

## Shape And Elevation

```css
--radius-sm: 8px;    /* 输入框、代码块、小控件 */
--radius-md: 12px;   /* 卡片、面板、表格容器 */
--radius-lg: 18px;   /* Modal、Drawer */
--radius-pill: 9999px; /* 按钮、标签、菜单项、品牌标识 */
```

- 页面区块通过 1px 边框、暖色面与留白分组，不使用浮动阴影堆叠。
- 阴影只用于覆盖层（Modal、Drawer、Dropdown）与主按钮的品牌色辉光。
- 玻璃效果使用 `backdrop-filter: blur(1rem-1.25rem)` 加半透明背景，仅见于顶栏、登录面板、粘性操作条。

## Layout

- 桌面端：224px 侧栏、64px 顶栏、内容最大宽度 1600px。
- 平板端：折叠侧栏，仅保留图标。
- 手机端：顶部栏加 Drawer，数据表切换为摘要列表。
- 固定格式组件使用稳定的网格轨道和最小高度，加载状态不能导致布局跳动。

## Motion

统一缓动为 `--xapo-ease: cubic-bezier(.4, 0, .2, 1)`。

- 状态切换 0.25-0.3s；菜单与表单控件 0.25s。
- 仅为 Drawer、折叠、任务状态、指标卡悬浮和操作反馈添加动效。
- `prefers-reduced-motion` 下取消位移和连续动画。

## Components

- `StatusBadge`：图标、文本和语义色共同表达状态。
- `MetricStrip`：紧凑资源指标带，卡片悬浮时轻微上移并描边。
- `ResponsiveDataView`：桌面 Table，移动端 List。
- `TaskProgress`：确定或不确定进度、速度、剩余量和实时日志。
- `ApprovalPanel`：展示 AI 计划、影响、命令等价描述和回滚动作。
- `LogViewer`：等宽字体、行号、过滤、暂停自动滚动和下载。

## Content

- 使用简洁中文动词：下载、部署、停止、重启、批准、拒绝。
- 不在页面中放置功能宣传或使用教程；解释通过 Tooltip、空状态和具体错误完成。
- 危险确认必须写清对象名称与影响，不使用泛化的“确定吗”。
- 指标必须标明口径：采样窗口、来源或“未知/不支持”，不得用 0 冒充未知。
