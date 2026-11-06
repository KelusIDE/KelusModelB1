import { createRoot } from 'react-dom/client';
import App from './App';
import { ThemeProvider, applyStoredMode } from './theme';
import './styles.css';

applyStoredMode();
createRoot(document.getElementById('root')!).render(<ThemeProvider><App /></ThemeProvider>);
