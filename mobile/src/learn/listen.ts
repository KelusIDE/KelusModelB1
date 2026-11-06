import { SpeechRecognition } from '@capacitor-community/speech-recognition';



export async function canListen(): Promise<boolean> {
  try {
    return (await SpeechRecognition.available()).available;
  } catch {
    return false;
  }
}

export async function listen(lang: string): Promise<string[]> {
  if ((await SpeechRecognition.checkPermissions()).speechRecognition !== 'granted'
      && (await SpeechRecognition.requestPermissions()).speechRecognition !== 'granted') {
    throw new Error('Kani needs the microphone to hear you. Allow it in the phone settings.');
  }
  const result = await SpeechRecognition.start({ language: lang, maxResults: 3, partialResults: false, popup: false });
  return result.matches ?? [];
}
