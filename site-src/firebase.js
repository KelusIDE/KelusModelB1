


export { initializeApp } from 'firebase/app';
export {
  getAuth, onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword, sendPasswordResetEmail,
  signInWithPopup, signOut, GoogleAuthProvider, GithubAuthProvider, connectAuthEmulator,
} from 'firebase/auth';
export {
  getFirestore, collection, doc, setDoc, deleteDoc, onSnapshot, query, orderBy, limit, serverTimestamp, connectFirestoreEmulator,
} from 'firebase/firestore';
