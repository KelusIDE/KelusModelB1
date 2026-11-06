import { TextToSpeech } from '@capacitor-community/text-to-speech';



export async function speak(text: string, lang: string, rate = 0.8): Promise<boolean> {
  try {
    await TextToSpeech.stop().catch(() => {});
    await TextToSpeech.speak({ text, lang, rate, pitch: 1, volume: 1, category: 'playback' });
    return true;
  } catch {
    return false;
  }
}

export async function canSpeak(lang: string): Promise<boolean> {
  try {
    return (await TextToSpeech.isLanguageSupported({ lang })).supported;
  } catch {
    return false;
  }
}
