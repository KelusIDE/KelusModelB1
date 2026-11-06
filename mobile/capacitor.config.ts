import type { CapacitorConfig } from '@capacitor/cli';
const config: CapacitorConfig = {
  appId: 'com.kelus.kani', appName: 'Kani', webDir: process.env.KANI_IOS_WEB_DIR || 'dist',
  backgroundColor: '#f5f3ec',
  ios: { contentInset: 'never' },
  plugins: {
    Keyboard: { resize: 'native', resizeOnFullScreen: true },

    FirebaseAuthentication: { skipNativeAuth: true, providers: ['google.com'] },
  },
};
export default config;
