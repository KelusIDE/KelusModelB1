import { useEffect, useMemo, useSyncExternalStore } from 'react';
import { App } from '@capacitor/app';
import { Capacitor, registerPlugin, type PluginListenerHandle } from '@capacitor/core';
import { createUpdateController } from './update-controller.mjs';

interface UpdaterPlugin {
  fetchManifest(options: { url: string }): Promise<{ text: string }>;
  download(options: { url: string; sha256: string; versionCode: number; size: number }): Promise<{ path: string }>;
  install(options: { path: string; sha256: string; versionCode: number; size: number }): Promise<{ started: boolean; permission: boolean }>;
  canInstall(): Promise<{ allowed: boolean }>;
  cleanup(): Promise<void>;
  addListener(event: 'progress', listener: (progress: { loaded: number; total: number }) => void): Promise<PluginListenerHandle>;
}
export interface Update { versionCode: number; versionName: string; apk: string; sha256: string; size: number; notes: string[] }
export type Phase = 'idle' | 'checking' | 'current' | 'available' | 'downloading' | 'permission' | 'installing' | 'error';
interface Snapshot {
  supported: boolean; version: { name: string; build: string }; update: Update | null;
  phase: Phase; progress: number; problem: string; hidden: boolean;
}
const native = registerPlugin<UpdaterPlugin>('Updater');
export function useUpdate() {
  const controller = useMemo(() => createUpdateController(native, App, Capacitor.getPlatform() === 'android'), []);
  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot) as Snapshot;
  useEffect(() => controller.mount(), [controller]);
  return { ...snapshot, check: () => controller.check(false), start: controller.start, install: controller.install, hide: controller.hide };
}
