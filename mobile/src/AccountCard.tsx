import { useState, type FormEvent } from 'react';
import { Icon } from './Icon';
import type { useChat } from './useChat';



type Account = ReturnType<typeof useChat>['account'];
type Note = { text: string; good?: boolean } | null;

export function AccountCard({ account }: { account: Account }) {
  const [sheet, setSheet] = useState(false), [leaving, setLeaving] = useState(false);
  const [busy, setBusy] = useState(false), [note, setNote] = useState<Note>(null);
  const run = async (action: () => Promise<string>, done?: string) => {
    setBusy(true); setNote(null);
    const problem = await action();
    setBusy(false);
    setNote(problem ? { text: problem } : done ? { text: done, good: true } : null);
  };
  const who = account.person;
  return <section className="settings-section account-card" aria-labelledby="account-title">
    <h2 id="account-title"><Icon name="user" size={19} />Kelus account</h2>
    {!account.ready ? <p>Getting accounts ready…</p> : !who ? <>
      <p>Sign in to keep your chats in your Kelus account too, and find them on kani.kelus.org. Optional: Kani works without it.</p>
      {account.canUseGoogle() && <button className="auth-button google" disabled={busy} onClick={() => void run(account.google)}><GoogleMark />Continue with Google</button>}
      <button className="auth-button email" disabled={busy} onClick={() => { setNote(null); setSheet(true); }}><Icon name="mail" size={18} />Continue with email</button>
    </> : !who.verified ? <>
      <div className="account-who"><span className="avatar" aria-hidden="true">{who.name[0]?.toUpperCase()}</span><span><strong>Verify your email</strong><small>{who.email}</small></span></div>
      <p>We sent a link to {who.email}. Open it, then come back and tap below. Chats start saving to your account once it's verified.</p>
      <button className="auth-button main" disabled={busy} onClick={() => void run(account.verified, 'Verified. Your chats now save to your account.')}>I've verified my email</button>
      <button className="auth-button quiet" disabled={busy} onClick={() => void run(account.resend, 'Sent again. Check your inbox and spam folder.')}>Send the link again</button>
      <button className="auth-button quiet" disabled={busy} onClick={() => void run(account.signOut)}>Use a different account</button>
    </> : <>
      <div className="account-who"><span className="avatar" aria-hidden="true">{who.name[0]?.toUpperCase()}</span><span><strong>{who.name}</strong><small>{who.email}</small></span></div>
      <p>{account.sync === 'synced' ? 'Your chats are saved to your account and sync with kani.kelus.org.'
        : account.sync === 'offline' ? 'Offline for now. Chats sync when you are back online.' : 'Syncing your chats…'}</p>
      <button className="auth-button quiet" disabled={busy} onClick={() => setLeaving(true)}>Sign out</button>
    </>}
    {note && <p className={`auth-note ${note.good ? 'good' : 'bad'}`} role="status">{note.text}</p>}
    {sheet && <EmailSheet account={account} close={() => setSheet(false)} />}
    {leaving && <div className="modal-backdrop" onClick={() => setLeaving(false)}><section className="modal" role="alertdialog" aria-modal="true" aria-labelledby="leave-title" onClick={e => e.stopPropagation()}>
      <h2 id="leave-title">Sign out?</h2><p>Chats already on this phone stay here. New chats stop saving to your account.</p>
      <div><button className="secondary" autoFocus onClick={() => setLeaving(false)}>Stay signed in</button>
        <button className="danger" onClick={() => { setLeaving(false); void run(account.signOut); }}>Sign out</button></div>
    </section></div>}
  </section>;
}

function EmailSheet({ account, close }: { account: Account; close: () => void }) {
  const [mode, setMode] = useState<'in' | 'new'>('in');
  const [email, setEmail] = useState(''), [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false), [note, setNote] = useState<Note>(null);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setNote(null);
    const problem = mode === 'in' ? await account.signIn(email, password) : await account.create(email, password);
    setBusy(false);
    if (problem) setNote({ text: problem }); else close();
  };
  const forgot = async () => {
    if (!email.trim()) { setNote({ text: 'Enter your email first.' }); return; }
    setBusy(true);
    const problem = await account.reset(email);
    setBusy(false);
    setNote(problem ? { text: problem } : { text: `If ${email.trim()} has an account, a reset link is on its way.`, good: true });
  };
  return <div className="modal-backdrop" onClick={close}><section className="modal sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-title" onClick={e => e.stopPropagation()}>
    <div className="close-row"><h2 id="sheet-title">{mode === 'in' ? 'Welcome back' : 'Create your account'}</h2>
      <button onClick={close} aria-label="Close"><Icon name="close" size={18} /></button></div>
    <p>{mode === 'in' ? 'Sign in with your Kelus account (the same one as the Kelus IDE and kani.kelus.org).'
      : 'One account for Kani, the Kelus IDE and kani.kelus.org. We send a link to verify your email.'}</p>
    <div className="segmented" role="group" aria-label="Account">{(['in', 'new'] as const).map(m =>
      <button key={m} type="button" aria-pressed={mode === m} onClick={() => { setMode(m); setNote(null); }}>{m === 'in' ? 'Sign in' : 'Create account'}</button>)}</div>
    <form onSubmit={e => void submit(e)}>
      <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" autoComplete="email" inputMode="email" autoCapitalize="off" aria-label="Email" required />
      <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder={mode === 'in' ? 'Password' : 'Password (6 or more characters)'}
        autoComplete={mode === 'in' ? 'current-password' : 'new-password'} minLength={6} aria-label="Password" required />
      <button className="auth-button main" disabled={busy || !email.trim() || password.length < 6}>{busy ? 'One moment…' : mode === 'in' ? 'Sign in' : 'Create account'}</button>
      {mode === 'in' && <button type="button" className="link" disabled={busy} onClick={() => void forgot()}>Forgot your password?</button>}
      {note && <p className={`auth-note ${note.good ? 'good' : 'bad'}`} role="status">{note.text}</p>}
    </form>
  </section></div>;
}


function GoogleMark() {
  return <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
    <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.6 5.4 2.7 13.2l7.9 6.2C12.5 13.6 17.8 9.5 24 9.5z" />
    <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.4c-.5 2.9-2.2 5.3-4.6 6.9l7.4 5.8c4.3-4 6.9-9.9 6.9-17.2z" />
    <path fill="#FBBC05" d="M10.6 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-7.9-6.2C1 16.6 0 20.2 0 24s1 7.4 2.7 10.8l7.9-6.2z" />
    <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.4-5.8c-2.1 1.4-4.8 2.3-8.5 2.3-6.2 0-11.5-4.1-13.4-9.9l-7.9 6.2C6.6 42.6 14.6 48 24 48z" />
  </svg>;
}
