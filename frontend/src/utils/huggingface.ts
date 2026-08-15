export type DownloadFileMode = 'all' | 'selected'


export function buildDownloadPatterns(mode: DownloadFileMode, selectedFiles: string[]) {
  return mode === 'selected' ? selectedFiles : []
}
