import { Capacitor } from '@capacitor/core';
import { FirebaseAuthentication } from '@capacitor-firebase/authentication';
import { initializeApp } from 'firebase/app';
import {
  GoogleAuthProvider, browserLocalPersistence, connectAuthEmulator, createUserWithEmailAndPassword, indexedDBLocalPersistence,
  initializeAuth, onAuthStateChanged, reload, sendEmailVerification, sendPasswordResetEmail, signInWithCredential,
  signInWithEmailAndPassword, signOut, type User,
} from 'firebase/auth';
import {
  collection, connectFirestoreEmulator, deleteDoc, doc, initializeFirestore, limit, onSnapshot, orderBy, query, serverTimestamp, setDoc,
  type Timestamp,
} from 'firebase/firestore';
import { MAX_SYNCED_CHATS, toRemote } from './sync.mjs';
import type { Chat } from './types';





const app = initializeApp({
  apiKey: 'AIzaSyBakQH8Zk7_Wkh-r7fqEW6FEjqLb-hQfdI', authDomain: 'kelus-ide.web.app', projectId: 'kelus-ide',
  appId: '1:335934652437:web:9fa724e71738b9a9690e1c', messagingSenderId: '335934652437',
});
const auth = initializeAuth(app, { persistence: [indexedDBLocalPersistence, browserLocalPersistence] });
const db = initializeFirestore(app, { experimentalAutoDetectLongPolling: true });

if (location.hostname === '127.0.0.1' && new URLSearchParams(location.search).has('emulators')) {
  connectAuthEmulator(auth, 'http://127.0.0.1:9099', { disableWarnings: true });
  connectFirestoreEmulator(db, '127.0.0.1', 8080);
}
const AFTER_VERIFYING = { url: 'https://ide.kelus.org/' };

export interface Person { uid: string; email: string; name: string; verified: boolean; google: boolean }
const person = (user: User): Person => ({
  uid: user.uid, email: user.email ?? '', name: user.displayName || user.email?.split('@')[0] || 'You',
  verified: user.emailVerified, google: user.providerData.some(p => p.providerId === 'google.com'),
});

export const watchPerson = (callback: (person: Person | null) => void) => onAuthStateChanged(auth, user => callback(user && person(user)));
export const canUseGoogle = () => Capacitor.isNativePlatform();

export async function signInGoogle(): Promise<Person> {
  if (!canUseGoogle()) throw new Error('Google sign-in works in the Kani app. Use email here.');
  const result = await FirebaseAuthentication.signInWithGoogle({ skipNativeAuth: true });
  const idToken = result.credential?.idToken;
  if (!idToken) throw new Error('Google did not sign you in. Try again.');
  return person((await signInWithCredential(auth, GoogleAuthProvider.credential(idToken))).user);
}
export async function signInEmail(email: string, password: string) { return person((await signInWithEmailAndPassword(auth, email.trim(), password)).user); }

export async function createAccount(email: string, password: string) {
  const { user } = await createUserWithEmailAndPassword(auth, email.trim(), password);
  await sendEmailVerification(user, AFTER_VERIFYING);
  return person(user);
}
export async function resendVerification() { if (auth.currentUser) await sendEmailVerification(auth.currentUser, AFTER_VERIFYING); }

export async function checkVerified(): Promise<Person | null> {
  if (!auth.currentUser) return null;
  await reload(auth.currentUser);
  if (auth.currentUser.emailVerified) await auth.currentUser.getIdToken(true);
  return person(auth.currentUser);
}
export const resetPassword = (email: string) => sendPasswordResetEmail(auth, email.trim(), AFTER_VERIFYING);
export async function signOutEverywhere() {
  await signOut(auth);
  if (canUseGoogle()) await FirebaseAuthentication.signOut().catch(() => {});
}

export interface RemoteChat { id: string; title: string; updated: number; model: string; messages: { role: string; content: string; source?: string; english?: string }[]; pending: boolean }
const chats = (uid: string) => collection(db, 'users', uid, 'b1Chats');


export function watchChats(uid: string, callback: (chats: RemoteChat[]) => void, onError: (error: unknown) => void) {
  return onSnapshot(query(chats(uid), orderBy('updatedAt', 'desc'), limit(MAX_SYNCED_CHATS)), { includeMetadataChanges: false }, snapshot => {
    callback(snapshot.docs.map(item => {
      const data = item.data({ serverTimestamps: 'estimate' });
      return { id: item.id, title: data.title, model: data.model, messages: data.messages ?? [], pending: item.metadata.hasPendingWrites,
        updated: (data.updatedAt as Timestamp | null)?.toMillis() ?? Date.now() };
    }));
  }, onError);
}


export function saveChat(uid: string, chat: Chat, created: boolean) {
  return setDoc(doc(chats(uid), chat.id), { ...toRemote(chat), updatedAt: serverTimestamp(), ...(created ? {} : { createdAt: serverTimestamp() }) }, { merge: true });
}
export const deleteChat = (uid: string, id: string) => deleteDoc(doc(chats(uid), id));

const MESSAGES: Record<string, string> = {
  'auth/invalid-credential': 'Email or password is incorrect.',
  'auth/invalid-email': 'Enter a valid email address.',
  'auth/missing-password': 'Enter your password.',
  'auth/weak-password': 'Use a password with at least 6 characters.',
  'auth/email-already-in-use': 'An account with this email already exists. Sign in instead.',
  'auth/too-many-requests': 'Too many attempts. Wait a moment and try again.',
  'auth/account-exists-with-different-credential': 'This email already uses another sign-in method.',
  'auth/network-request-failed': 'No connection. Check your internet and try again.',
  'auth/user-disabled': 'This account has been turned off.',
  'permission-denied': 'Saving was refused. Sign out and in again.',
  unavailable: 'Saved chats are offline for now. They sync when you are back online.',
};
export function friendlyError(error: unknown): string {
  const code = (error as { code?: string })?.code ?? '', text = error instanceof Error ? error.message : String(error ?? '');
  if (/cancel|canceled|cancelled|16:/i.test(text) && !code.startsWith('auth/')) return '';
  if (/no credentials|NoCredential/i.test(text)) return 'No Google account on this phone. Add one in the phone settings, or use email.';
  return MESSAGES[code] || text || 'Something went wrong.';
}
