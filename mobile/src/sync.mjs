


import { id as newId } from './id.ts';

export const MAX_SAVED_MESSAGES = 200;
export const MAX_SYNCED_CHATS = 50;
const WEB_MODEL = { 'b1-v3': 'Kani Seal 3 · web' };


export function toRemote(chat) {
  const messages = chat.messages.filter(m => m.content || m.saved).slice(-MAX_SAVED_MESSAGES)
    .map(({ role, content, source, saved }) => ({ role, content: saved || content, ...(source && source !== 'model' ? { source } : {}) }));
  return { title: (chat.title || 'New conversation').slice(0, 120), messages, model: (chat.model || 'Kani').slice(0, 40) };
}


export function fromRemote(remote) {
  return {
    id: remote.id, title: remote.title || 'Saved chat', updated: remote.updated, model: WEB_MODEL[remote.model] || remote.model || 'Kani',
    messages: (remote.messages || []).filter(m => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .map(m => ({ id: newId(), role: m.role, content: m.content, status: 'complete', ...(m.role === 'assistant' ? { source: m.source || 'model' } : {}) })),
  };
}









export function mergeChats(local, remote, known) {
  const remoteById = new Map(remote.map(r => [r.id, r])), localIds = new Set(local.map(c => c.id));
  const oldest = remote.length >= MAX_SYNCED_CHATS ? Math.min(...remote.map(r => r.updated)) : -Infinity;
  const merged = [], upload = [], dropped = [];
  for (const chat of local) {
    const theirs = remoteById.get(chat.id);
    if (!theirs) {
      if (known.has(chat.id) && chat.updated >= oldest) dropped.push(chat.id);
      else { merged.push(chat); upload.push(chat); }
    } else if (theirs.updated > chat.updated + 1000 && !theirs.pending) merged.push(fromRemote(theirs));
    else {
      merged.push(chat);
      if (chat.updated > theirs.updated + 1000) upload.push(chat);
    }
  }
  for (const theirs of remote) if (!localIds.has(theirs.id)) merged.push(fromRemote(theirs));
  merged.sort((a, b) => b.updated - a.updated);
  return { merged: merged.slice(0, MAX_SYNCED_CHATS), upload, dropped };
}
