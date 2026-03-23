import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

function serveOutputPlugin() {
  const serveMiddleware = (server: any) => {
    server.middlewares.use((req: any, res: any, next: any) => {
      if (req.url?.startsWith('/output/')) {
        const fileName = req.url.slice('/output/'.length).split('?')[0]
        const filePath = path.resolve('./output', fileName)
        if (fs.existsSync(filePath)) {
          const stat = fs.statSync(filePath)
          res.setHeader('Content-Type', 'text/csv; charset=utf-8')
          res.setHeader('Content-Length', stat.size)
          res.setHeader('Access-Control-Allow-Origin', '*')
          fs.createReadStream(filePath).pipe(res)
          return
        }
      }
      next()
    })
  }

  return {
    name: 'serve-output-dir',
    configureServer: serveMiddleware,
    configurePreviewServer: serveMiddleware,
    writeBundle() {
      const srcDir = path.resolve('./output')
      const dstDir = path.resolve('./dist/output')
      if (!fs.existsSync(srcDir)) return
      if (!fs.existsSync(dstDir)) fs.mkdirSync(dstDir, { recursive: true })
      for (const file of fs.readdirSync(srcDir)) {
        if (file.endsWith('.csv')) {
          fs.copyFileSync(path.resolve(srcDir, file), path.resolve(dstDir, file))
        }
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), serveOutputPlugin()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
