import { MANIFEST_URL, isNewer, parseUpdate } from './update-manifest.mjs';
const AGAIN_AFTER = 6 * 3600_000;
const busy = ['checking', 'downloading', 'permission', 'installing'];


export function createUpdateController(native, app, supported, now = Date.now) {
  let value = { supported, version: { name: '', build: '' }, update: null, phase: 'idle', progress: 0, problem: '', hidden: false };
  let live = false, checked = 0, path = '', installing = false, hiddenFor = 0, listener;
  const subscribers = new Set();
  const set = patch => { if (live) { value = { ...value, ...patch }; subscribers.forEach(fn => fn()); } };
  const problem = (error, fallback) => error?.message || fallback;
  const controller = {
    getSnapshot: () => value,
    subscribe: callback => { subscribers.add(callback); return () => subscribers.delete(callback); },
    mount() {
      live = true;
      if (supported) {
        void (async () => {
          try {
            const info = await app.getInfo();
            if (!live) return;
            if (!/^\d+$/.test(info.build) || Number(info.build) <= 0) throw new Error('Could not read the installed app version.');
            set({ version: { name: info.version, build: info.build } });
            await native.cleanup().catch(() => {});
            if (live) await controller.check(true);
          } catch (error) { set({ phase: 'error', problem: problem(error, 'Could not read the installed app version.') }); }
        })();
        listener = app.addListener('appStateChange', ({ isActive }) => { if (isActive) void controller.resume(); });
      }
      return () => { live = false; void listener?.then(handle => handle.remove()).catch(() => {}); };
    },
    async check(quiet = false) {
      if (!live || !supported || !value.version.build || busy.includes(value.phase)) return;
      const previous = value.update;
      checked = now(); set({ phase: 'checking', problem: '' });
      try {
        const found = parseUpdate((await native.fetchManifest({ url: MANIFEST_URL })).text);
        if (!found) throw new Error('The update information is invalid. Please try again later.');
        if (!live) return;
        if (isNewer(found, value.version.build)) {
          if (previous?.sha256 !== found.sha256) path = '';
          if (!quiet) hiddenFor = 0;
          set({ update: found, phase: 'available', hidden: hiddenFor === found.versionCode });
        } else { path = ''; set({ update: null, phase: 'current', hidden: false }); }
      } catch (error) {
        set({ phase: quiet ? previous ? 'available' : 'idle' : 'error', problem: quiet ? '' : problem(error, 'Could not check for updates. Check your connection.') });
      }
    },
    async start() {
      if (!live || !value.update || ['checking', 'downloading', 'installing', 'permission'].includes(value.phase)) return;
      if (path) { await controller.install(); return; }
      const update = value.update;
      set({ phase: 'downloading', progress: 0, problem: '' });
      let progressListener;
      try {
        progressListener = await native.addListener('progress', ({ loaded, total }) => {
          if (value.phase === 'downloading') set({ progress: total > 0 ? Math.max(0, Math.min(1, loaded / total)) : 0 });
        });
        const result = await native.download({ url: update.apk, sha256: update.sha256, versionCode: update.versionCode, size: update.size });
        if (!live) return;
        path = result.path; set({ progress: 1 }); await controller.install();
      } catch (error) { set({ phase: 'error', problem: problem(error, 'The update could not be downloaded.') }); }
      finally { await progressListener?.remove().catch(() => {}); }
    },
    async install() {
      if (!live || !value.update || !path || installing) return;
      installing = true;
      try {
        const result = await native.install({ path, sha256: value.update.sha256, versionCode: value.update.versionCode, size: value.update.size });
        if (!result.permission && !result.started) throw new Error('The installer did not open. Try again.');
        set({ phase: result.permission ? 'permission' : 'installing', problem: '' });
      } catch (error) { path = ''; set({ phase: 'error', problem: problem(error, 'The installer did not open. Try again.') }); }
      finally { installing = false; }
    },
    async resume() {
      if (!live || !supported) return;
      if (value.phase === 'permission') {
        try {

          if ((await native.canInstall()).allowed) await controller.install();
          else set({ phase: 'available', problem: '' });
        } catch { set({ phase: 'available' }); }
      } else if (value.phase === 'installing') set({ phase: 'available' });
      else if (now() - checked > AGAIN_AFTER) await controller.check(true);
    },
    hide() { hiddenFor = value.update?.versionCode || 0; set({ hidden: true }); },
  };
  return controller;
}
