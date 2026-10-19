// Firebase Web SDK entry for the B1 site, bundled to site/vendor/firebase.js:
//   cd ../Kelus && NODE_PATH=$PWD/node_modules node_modules/.bin/esbuild ../KelusModelB1/site-src/firebase.js \
//     --bundle --format=esm --minify --legal-comments=eof --outfile=../KelusModelB1/site/vendor/firebase.js
export { initializeApp } from 'firebase/app';
export {
  getAuth, onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword, sendPasswordResetEmail,
  signInWithPopup, signOut, GoogleAuthProvider, GithubAuthProvider, connectAuthEmulator,
} from 'firebase/auth';
export {
  getFirestore, collection, doc, setDoc, deleteDoc, onSnapshot, query, orderBy, limit, serverTimestamp, connectFirestoreEmulator,
} from 'firebase/firestore';
