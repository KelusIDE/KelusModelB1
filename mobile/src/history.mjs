import { spokenAnswer } from '../../site/reading.js';
export const MAX_CHATS = 30;
export function restoreChats(value) {
  if (!value) return [];
  const chats = JSON.parse(value);
  if (!Array.isArray(chats) || chats.some(chat =>
    typeof chat.id !== 'string' || typeof chat.title !== 'string' || typeof chat.model !== 'string' ||
    !Number.isFinite(chat.updated) || !Array.isArray(chat.messages) || chat.messages.some(message =>
      typeof message.id !== 'string' || !['user', 'assistant'].includes(message.role) || typeof message.content !== 'string'))) {
    throw new Error('Saved chats could not be read. Your stored copy has been left intact.');
  }
  return chats.slice(0, MAX_CHATS);
}

// The model reads a translated turn in English; the person keeps reading it in their own language.
const inEnglish = message => message.english ? { ...message, content: message.english } : message;

export function completedContext(messages) {
  const context = [];
  for (let i = 0; i + 1 < messages.length; i += 2) {
    const user = inEnglish(messages[i]), assistant = inEnglish(messages[i + 1]);
    if (user.role === 'user' && assistant.role === 'assistant' && assistant.status === 'complete' &&
        assistant.source !== 'search_unavailable' && assistant.content) {
      // A searched answer goes back to the model as the answer alone, without the search line and the sources.
      const content = assistant.saved || (assistant.source === 'model_web' ? spokenAnswer(assistant.content) : assistant.content);
      context.push(user, { ...assistant, content });
    }
  }
  return context.slice(-14);
}

export function storable(history) {
  return history.map(chat => ({ ...chat, messages: chat.messages.map(({ saved, ...message }) => saved ? { ...message, content: saved } : message) }));
}
export function updateHistory(history, chat) {
  if (!chat.messages.length) return history;
  return [chat, ...history.filter(item => item.id !== chat.id)].slice(0, MAX_CHATS);
}
