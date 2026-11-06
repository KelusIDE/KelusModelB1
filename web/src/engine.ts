

import * as b1 from '../../site/b1.js';
import * as gpu from '../../site/gpu.js';
import * as svg from '../../site/svg.js';
import * as web from '../../site/web.js';
import * as files from '../../site/attachments.js';
import * as reading from '../../site/reading.js';
import * as places from '../../site/places.js';
import * as documents from '../../site/documents.js';

export type Role = 'user' | 'assistant';
export interface Turn { role: Role; content: string }
export type Tool = [reply: string, source: string];

export interface Model {
  manifest: { name: string; codename: string; training: { steps?: number }; config: { context: number }; weights_sha256: string };
  config: { context: number };
  parameterCount: number;
  tokenizer: { encode(text: string): number[] };
  generate(messages: Turn[], options: { maxTokens: number; temperature: number }): Iterable<string> | AsyncIterable<string>;
}
export interface GpuModel extends Model { name: string; lost: Promise<unknown>; destroy(): void }

export interface WebRequest { mode: 'search' | 'read'; query: string }
export interface WebResult { text: string; engine?: string }

export interface Attachment { name: string; kind: 'text' | 'image' | 'pdf'; size: number; report: string; blob?: Blob; text?: string }
export interface PdfText { title: string | null; author: string | null; pages: string[]; total: number }
export interface PreparedFiles { helper: boolean; text: string; receipt: string; source?: string }

export interface Account {
  watchUser(callback: (user: User | null) => void): () => void;
  watchChats(uid: string, callback: (chats: SavedChat[]) => void, onError: (error: unknown) => void): () => void;
  saveChat(uid: string, id: string, messages: StoredTurn[], options: { created: boolean; model?: string }): Promise<void>;
  deleteChat(uid: string, id: string): Promise<void>;
  signInEmail(email: string, password: string): Promise<unknown>;
  createAccount(email: string, password: string): Promise<unknown>;
  resetPassword(email: string): Promise<unknown>;
  signInWith(provider: 'google' | 'github'): Promise<unknown>;
  signOutUser(): Promise<unknown>;
  friendlyError(error: unknown): string;
}
export interface User { uid: string; email?: string | null; displayName?: string | null }
export interface StoredTurn extends Turn { source?: string }
export interface SavedChat { id: string; title?: string; messages?: StoredTurn[]; updatedAt?: { toDate(): Date } | null; pending?: boolean }

export const loadModel = b1.loadModel as (base?: string) => Promise<Model>;
// The same model on WebGPU (about four times faster), checked against the CPU's results first; null without WebGPU.
export const gpuModel = gpu.gpuModel as (cpu: Model) => Promise<GpuModel | null>;
export const prepareReply = b1.prepareReply as (model: Model, messages: Turn[], maxTokens: number) =>
  { retained: Turn[]; omitted: number; tool: Tool | null };

export const svgRequest = svg.svgRequest as (text: string) => string | null;
export const svgPrompt = svg.svgPrompt as (description: string) => string;
export const extractSvg = svg.extractSvg as (reply: string) => string;

export const webRequest = web.webRequest as (text: string) => WebRequest | null;
export const webAnswer = web.webAnswer as (request: WebRequest, fetcher: typeof fetch, signal?: AbortSignal) => Promise<WebResult>;

export const SEARCH_CALL = reading.SEARCH_CALL as RegExp;
export const spokenAnswer = reading.spokenAnswer as (text: string) => string;
export const maybeSearch = reading.maybeSearch as (text: string) => boolean;
export const searchedAnswer = reading.searchedAnswer as (query: string, question: string, options: {
  generate: (messages: Turn[]) => Iterable<string> | AsyncIterable<string>; webAnswer: typeof webAnswer; signal?: AbortSignal; fits?: (prompt: string) => boolean;
}) => AsyncIterable<string>;

export interface Near { lat: number; lng: number }
export const PLACES_CALL = places.PLACES_CALL as RegExp;
export const maybePlaces = places.maybePlaces as (text: string) => boolean;
export const placesAnswer = places.placesAnswer as (call: RegExpMatchArray, options: { locate: () => Promise<Near | null>; signal?: AbortSignal }) => AsyncIterable<string>;
export const placesSummary = places.placesSummary as (what: string, where: string) => string;
export interface Place {
  name: string; url: string; address: string | null; rating: number | null; ratings: number | null; open: boolean | null; price: string | null;
  category: string | null; area: string | null; about: string | null; website: string | null; wiki: string | null; image: string | null;
}
export interface PlacesResult { what: string; where: string; status: 'found' | 'none' | 'location' | 'error'; places: Place[]; source?: string; message?: string }
export const findPlaces = places.findPlaces as (call: RegExpMatchArray, options: { locate: () => Promise<Near | null>; signal?: AbortSignal }) => Promise<PlacesResult>;
export const placesReply = places.placesReply as (result: PlacesResult) => string;
export const placesPhrase = places.phrase as (what: string, where: string) => string;
export const browserLocation = places.browserLocation as () => Promise<Near | null>;

export const ACCEPT = files.ACCEPT as string;
export const validateBatch = files.validateBatch as (batch: { name: string; size: number }[]) => void;
export const readAttachment = files.readAttachment as (file: File, options?: { readPdf?: (bytes: Uint8Array) => Promise<PdfText> }) => Promise<Attachment>;
export const MAX_PAGES = documents.MAX_PAGES as number;
export const prepareAttachments = files.prepareAttachments as (prompt: string, attached: Attachment[], tokenizer: unknown,
  context: number, maxTokens: number) => PreparedFiles;


export const loadAccount = () => import('../../site/account.js') as unknown as Promise<Account>;
