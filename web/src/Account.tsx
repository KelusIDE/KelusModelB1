import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { Account, SavedChat, User } from './engine';

function relativeTime(timestamp: SavedChat['updatedAt']) {
  if (!timestamp?.toDate) return 'saving…';
  const seconds = (Date.now() - timestamp.toDate().getTime()) / 1000;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return timestamp.toDate().toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function ChatList({ chats, current, onOpen, onDelete, className = '' }: {
  chats: SavedChat[]; current: string; onOpen: (chat: SavedChat) => void; onDelete: (chat: SavedChat) => void; className?: string;
}) {
  return (
    <ul className={`chat-list ${className}`}>
      {!chats.length && <li className="chat-empty">No saved chats yet. Send a message to start one.</li>}
      {chats.map(chat => (
        <li key={chat.id} aria-current={chat.id === current}>
          <button className="chat-item" onClick={() => onOpen(chat)}>
            <span>{chat.title || 'Untitled chat'}</span><small>{relativeTime(chat.updatedAt)}</small>
          </button>
          <button className="chat-delete" aria-label={`Delete chat: ${chat.title || 'Untitled chat'}`} onClick={() => onDelete(chat)}>×</button>
        </li>
      ))}
    </ul>
  );
}


export function AccountDialog({ open, onClose, account, user, chats, current, onOpenChat, onDeleteChat, onNewChat }: {
  open: boolean; onClose: () => void; account: Account | null; user: User | null; chats: SavedChat[]; current: string;
  onOpenChat: (chat: SavedChat) => void; onDeleteChat: (chat: SavedChat) => void; onNewChat: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState({ text: '', error: false });

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) { setStatus({ text: '', error: false }); element.showModal(); }
    if (!open && element.open) element.close();
  }, [open]);

  async function act(action: () => Promise<unknown>, success = '') {
    if (!account) return;
    setStatus({ text: 'Working…', error: false });
    try {
      await action();
      setStatus({ text: success, error: false });
      if (!success) onClose();
    } catch (error) { setStatus({ text: account.friendlyError(error), error: true }); }
  }

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const create = ((event.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null)?.dataset.mode === 'create';
    void act(() => create ? account!.createAccount(email.trim(), password) : account!.signInEmail(email.trim(), password));
  };
  const forgot = () => {
    if (!email.trim()) { setStatus({ text: 'Enter your email first.', error: true }); return; }
    void act(() => account!.resetPassword(email.trim()), 'If an account exists for that email, a reset link is on its way.');
  };

  return (
    <dialog id="account-dialog" ref={dialog} aria-labelledby="account-title" onClose={onClose}
      onClick={event => { if (event.target === dialog.current) onClose(); }}>
      <div className="dialog-head"><h2 id="account-title">Kelus account</h2><button id="close-account" className="icon-button" aria-label="Close" onClick={onClose}>×</button></div>
      {!user ? (
        <div id="signed-out-view">
          <p className="dialog-copy">One account for the Kelus IDE and Kani. Signed-in chats are saved to your account and stay up to date on every device. The model itself always runs on this device.</p>
          <form id="email-form" onSubmit={submit}>
            <label>Email<input id="email" type="email" autoComplete="email" required value={email} onChange={event => setEmail(event.target.value)} /></label>
            <label>Password<input id="password" type="password" autoComplete="current-password" minLength={6} required value={password} onChange={event => setPassword(event.target.value)} /></label>
            <div className="dialog-actions"><button type="submit" className="primary" data-mode="signin">Sign in</button><button type="submit" data-mode="create">Create account</button></div>
            <button type="button" id="forgot" className="link" onClick={forgot}>Forgot password?</button>
          </form>
          <div className="divider"><span>or</span></div>
          <div className="providers">
            <button type="button" onClick={() => void act(() => account!.signInWith('google'))}>Continue with Google</button>
            <button type="button" onClick={() => void act(() => account!.signInWith('github'))}>Continue with GitHub</button>
          </div>
        </div>
      ) : (
        <div id="signed-in-view">
          <p className="dialog-copy">Signed in as <strong id="account-email">{user.email || user.displayName || 'your account'}</strong></p>
          <div className="eyebrow">Saved chats</div>
          <ChatList className="dialog-list" chats={chats} current={current} onOpen={chat => { onOpenChat(chat); onClose(); }} onDelete={onDeleteChat} />
          <div className="dialog-actions">
            <button type="button" id="dialog-new-chat" onClick={() => { onNewChat(); onClose(); }}>New conversation</button>
            <button type="button" id="sign-out" onClick={() => void act(() => account!.signOutUser())}>Sign out</button>
          </div>
        </div>
      )}
      <p id="auth-status" role="status" className={status.error ? 'error' : undefined}>{status.text}</p>
    </dialog>
  );
}
