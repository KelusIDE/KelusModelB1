// Kelus accounts (Firebase project kelus-ide, shared with the Kelus IDE) and saved B1 chats.
// The model still runs in the browser; only the transcript of a signed-in chat is stored,
// under users/{uid}/b1Chats/{chatId}. Firestore rules allow the owner only.
import {
  initializeApp, getAuth, onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword,
  sendPasswordResetEmail, signInWithPopup, signOut, GoogleAuthProvider, GithubAuthProvider, connectAuthEmulator,
  getFirestore, collection, doc, setDoc, deleteDoc, onSnapshot, query, orderBy, limit, serverTimestamp, connectFirestoreEmulator,
} from './vendor/firebase.js';

const app = initializeApp({ apiKey: 'AIzaSyBakQH8Zk7_Wkh-r7fqEW6FEjqLb-hQfdI', authDomain: 'kelus-ide.firebaseapp.com', projectId: 'kelus-ide' });
const auth = getAuth(app), db = getFirestore(app);
// Local testing only: http://127.0.0.1:<port>/?emulators uses the Firebase emulators instead of real accounts.
if (location.hostname === '127.0.0.1' && new URLSearchParams(location.search).has('emulators')) {
  connectAuthEmulator(auth, 'http://127.0.0.1:9099', { disableWarnings: true });
  connectFirestoreEmulator(db, '127.0.0.1', 8080);
}
export const MAX_SAVED_MESSAGES = 200;

export const watchUser = callback => onAuthStateChanged(auth, callback);
export const signInEmail = (email, password) => signInWithEmailAndPassword(auth, email, password);
export const createAccount = (email, password) => createUserWithEmailAndPassword(auth, email, password);
export const resetPassword = email => sendPasswordResetEmail(auth, email);
export const signInWith = provider => signInWithPopup(auth, provider === 'github' ? new GithubAuthProvider() : new GoogleAuthProvider());
export const signOutUser = () => signOut(auth);

const chats = uid => collection(db, 'users', uid, 'b1Chats');

/** Live list of the 50 most recent chats; the callback runs again on every change, from any device. */
export function watchChats(uid, callback, onError) {
  return onSnapshot(query(chats(uid), orderBy('updatedAt', 'desc'), limit(50)), snapshot => {
    callback(snapshot.docs.map(item => ({ id: item.id, pending: item.metadata.hasPendingWrites, ...item.data({ serverTimestamps: 'estimate' }) })));
  }, onError);
}

export function saveChat(uid, id, messages, { created }) {
  const first = messages.find(message => message.role === 'user')?.content.trim().replace(/\s+/g, ' ') || 'New chat';
  const title = first.length > 60 ? first.slice(0, 59) + '…' : first;
  const stored = messages.slice(-MAX_SAVED_MESSAGES).map(({ role, content, source }) => source ? { role, content, source } : { role, content });
  return setDoc(doc(chats(uid), id), {
    title, messages: stored, model: 'b1-v3', updatedAt: serverTimestamp(), ...(created ? {} : { createdAt: serverTimestamp() }),
  }, { merge: true });
}

export const deleteChat = (uid, id) => deleteDoc(doc(chats(uid), id));

const MESSAGES = {
  'auth/invalid-credential': 'Email or password is incorrect.',
  'auth/invalid-email': 'Enter a valid email address.',
  'auth/missing-password': 'Enter your password.',
  'auth/weak-password': 'Use a password with at least 6 characters.',
  'auth/email-already-in-use': 'An account with this email already exists. Sign in instead.',
  'auth/too-many-requests': 'Too many attempts. Wait a moment and try again.',
  'auth/popup-closed-by-user': 'Sign-in window was closed.',
  'auth/popup-blocked': 'The browser blocked the sign-in window. Allow pop-ups for this site.',
  'auth/unauthorized-domain': 'Google and GitHub sign-in are not enabled for this site yet. Use email instead.',
  'auth/account-exists-with-different-credential': 'This email already uses another sign-in method.',
  'auth/network-request-failed': 'Network error. Check your connection.',
  'permission-denied': 'Saving was refused. Sign in again.',
};
export const friendlyError = error => MESSAGES[error?.code] || error?.message || 'Something went wrong.';
