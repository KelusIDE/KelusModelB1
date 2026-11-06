


export const REPOSITORY = 'https://github.com/KelusIDE/KelusModelB1/releases/';
export const MANIFEST_URL = `${REPOSITORY}latest/download/kani-update.json`;
export const MAX_APK_BYTES = 1024 * 1024 * 1024;

export function isReleaseApk(value) {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && url.hostname === 'github.com' && !url.port && !url.username && !url.password
      && !url.search && !url.hash && /^\/KelusIDE\/KelusModelB1\/releases\/download\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.apk$/.test(url.pathname)
      && !url.pathname.split('/').includes('..') && url.href === value;
  } catch { return false; }
}


export function parseUpdate(text) {
  let data;
  try { data = JSON.parse(text); } catch { return null; }
  if (!data || !Number.isSafeInteger(data.versionCode) || data.versionCode <= 0 || data.versionCode > 2147483647
      || typeof data.versionName !== 'string' || !/^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/.test(data.versionName) || data.versionName.length > 40
      || !isReleaseApk(data.apk) || !/^[0-9a-f]{64}$/.test(data.sha256 ?? '')
      || !Number.isSafeInteger(data.size) || data.size <= 0 || data.size > MAX_APK_BYTES) return null;
  const notes = Array.isArray(data.notes) ? data.notes.filter(n => typeof n === 'string').slice(0, 8).map(n => n.slice(0, 300)) : [];
  return { versionCode: data.versionCode, versionName: data.versionName, apk: data.apk, sha256: data.sha256, size: data.size, notes };
}


export const isNewer = (update, build) => Boolean(update) && /^\d+$/.test(String(build))
  && Number.isSafeInteger(Number(build)) && Number(build) > 0 && update.versionCode > Number(build);

export const megabytes = bytes => `${Math.round(bytes / 1e6)} MB`;
