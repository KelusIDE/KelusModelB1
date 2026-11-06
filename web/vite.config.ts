import { rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const site = fileURLToPath(new URL('../site', import.meta.url));




const clearAssets = (): Plugin => ({
  name: 'clear-old-assets',
  apply: 'build',
  buildStart() { rmSync(`${site}/assets`, { recursive: true, force: true }); },
});

export default defineConfig(({ command }) => ({
  plugins: [react(), clearAssets()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },

  publicDir: command === 'serve' ? site : false,
  server: { fs: { allow: ['..'] } },
  build: { outDir: site, emptyOutDir: false, assetsDir: 'assets', target: 'es2022' },
}));
