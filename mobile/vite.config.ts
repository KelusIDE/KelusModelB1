import { copyFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// English practice (shared with the website from web/src and site/) fetches these beside the app.
const VOICE = ['cmudict.txt', 'whisper-tokens.json', 'CMUDICT-LICENSE'];
const voiceFiles = (): Plugin => ({
  name: 'voice-files', apply: 'build',
  async writeBundle({ dir }) {
    await mkdir(`${dir}/voice`, { recursive: true });
    for (const name of VOICE) await copyFile(fileURLToPath(new URL(`../site/voice/${name}`, import.meta.url)), `${dir}/voice/${name}`);
  },
});
export default defineConfig(({ command }) => ({


  plugins: [react(), voiceFiles(), { name: 'dev-csp', transformIndexHtml(html) {
    return command === 'serve' ? html.replace(/<meta http-equiv="Content-Security-Policy"[^>]*>/, '') : html;
  } }], base: './',
  server: { fs: { allow: ['..'] } },
  resolve: { dedupe: ['react', 'react-dom'] },
  worker: { format: 'iife' }, build: { target: 'es2022' },
}));
