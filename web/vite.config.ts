import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  server: { port: 5301, strictPort: true },
  preview: { port: 5301, strictPort: true },
  build: { outDir: 'dist', target: 'es2022', chunkSizeWarningLimit: 2000 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
