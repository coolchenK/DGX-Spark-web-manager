export function formatBytes(value: number | null | undefined, decimals = 1): string {
  if (value == null) return '不支持'
  if (value === 0) return '0 B'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  return `${(value / 1024 ** index).toFixed(index === 0 ? 0 : decimals)} ${units[index]}`
}

export function percent(used: number, total: number): number {
  return total > 0 ? (used / total) * 100 : 0
}

export function formatDuration(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return days ? `${days}天 ${hours}小时` : `${hours}小时 ${minutes}分钟`
}

export function formatDate(value: string | null | undefined): string {
  return value ? new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short', timeStyle: 'medium' }).format(new Date(value)) : '从未'
}
