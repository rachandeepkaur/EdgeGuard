import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// /api goes to the local analysis server (integration/analyze_server.py):
//   python -m uvicorn integration.analyze_server:app --host 127.0.0.1 --port 8000
const api = { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } }

export default defineConfig({
  plugins: [react()],
  server: { proxy: api },
  preview: { proxy: api },
})
