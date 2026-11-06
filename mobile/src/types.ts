export interface Message {
  id: string; role: 'user' | 'assistant'; content: string;
  status?: 'complete' | 'stopped' | 'failed'; source?: string;
  saved?: string;
  // This message in English, when it was typed or answered in another language. The model only reads English.
  english?: string;
}
export interface Chat { id: string; title: string; updated: number; messages: Message[]; model: string }
export interface ModelInfo {
  name: string; codename: string; parameters: number; sha256: string; context: number;
  device: 'gpu' | 'cpu'; gpu?: string; note?: string;
}
export type WorkerReply =
  | { type: 'loading'; progress: number; label: string }
  | { type: 'ready'; model: ModelInfo }
  | { type: 'token'; id: string; text: string }
  | { type: 'done'; id: string; stopped: boolean; source: string; saved?: string }
  | { type: 'locate'; id: string }
  | { type: 'error'; id?: string; error: string }
  | { type: 'gpu-lost'; error: string };
