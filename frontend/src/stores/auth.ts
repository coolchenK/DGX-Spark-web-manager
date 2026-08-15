import { create } from 'zustand'

import type { User } from '../api/types'


interface AuthState {
  user: User | null
  csrfToken: string
  setSession: (user: User, csrfToken: string) => void
  clearSession: () => void
}


export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  csrfToken: sessionStorage.getItem('dgx-csrf') ?? '',
  setSession: (user, csrfToken) => {
    if (csrfToken) sessionStorage.setItem('dgx-csrf', csrfToken)
    set({ user, csrfToken: csrfToken || sessionStorage.getItem('dgx-csrf') || '' })
  },
  clearSession: () => {
    sessionStorage.removeItem('dgx-csrf')
    set({ user: null, csrfToken: '' })
  },
}))
