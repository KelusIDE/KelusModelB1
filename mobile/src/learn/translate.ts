import { Capacitor } from '@capacitor/core';
import { Language, Translation } from '@capacitor-mlkit/translation';
import type { LanguageId } from './content.ts';




export const canTranslate = () => Capacitor.isNativePlatform();

const code = (id: LanguageId | string) => (id.startsWith('zh') ? 'zh' : id) as Language;


export async function translate(text: string, from: LanguageId | string, to: LanguageId | string): Promise<string> {
  if (!canTranslate()) throw new Error('Translation works in the Kani app on Android.');
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error('The translation pack (about 30 MB) is still downloading. It downloads over Wi-Fi; try again in a minute.')), 120_000);
  });
  try {
    const result = await Promise.race([Translation.translate({ text, sourceLanguage: code(from), targetLanguage: code(to) }), late]);
    return result.text.trim();
  } catch (error) {
    throw error instanceof Error && error.message.startsWith('The translation pack') ? error : new Error('Translation failed. Try again.');
  } finally {
    clearTimeout(timer);
  }
}
