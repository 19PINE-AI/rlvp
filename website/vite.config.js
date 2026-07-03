import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built site works when opened from any static host / subpath.
export default defineConfig({
  plugins: [react()],
  base: './',
})
