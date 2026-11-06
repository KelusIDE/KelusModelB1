import { Capacitor } from '@capacitor/core';
import { Camera, MediaTypeSelection } from '@capacitor/camera';
import { Script, TextRecognition } from '@capacitor-mlkit/text-recognition';
import { ImageLabeling } from '@capacitor-mlkit/image-labeling';
import type { LanguageId } from './content.ts';



export const canScan = () => Capacitor.isNativePlatform();
export interface Photo { uri: string; shown: string }


export async function photo(from: 'camera' | 'gallery'): Promise<Photo | null> {
  try {
    const result = from === 'camera'
      ? await Camera.takePhoto({ quality: 85, targetWidth: 2048, targetHeight: 2048, correctOrientation: true, editable: 'no' })
      : (await Camera.chooseFromGallery({ mediaType: MediaTypeSelection.Photo, limit: 1, quality: 85, targetWidth: 2048, targetHeight: 2048, correctOrientation: true })).results[0];
    if (!result?.uri) return null;
    const uri = result.uri.startsWith('/') ? `file://${result.uri}` : result.uri;
    return { uri, shown: result.webPath ?? Capacitor.convertFileSrc(uri) };
  } catch (error) {
    if (/cancel/i.test(String(error))) return null;
    throw new Error(from === 'camera' ? 'The camera could not take a photo.' : 'That photo could not be opened.');
  }
}

const SCRIPTS: Partial<Record<LanguageId, Script>> = { ko: Script.Korean, ja: Script.Japanese, 'zh-Hans': Script.Chinese, 'zh-Hant': Script.Chinese };


export async function readText(uri: string, id: LanguageId): Promise<string[]> {
  const { blocks } = await TextRecognition.processImage({ path: uri, script: SCRIPTS[id] ?? Script.Latin });
  return blocks.flatMap(block => block.lines.map(line => line.text.trim())).filter(line => /\p{L}/u.test(line));
}


export async function things(uri: string): Promise<string[]> {
  const { labels } = await ImageLabeling.processImage({ path: uri, confidenceThreshold: 0.6 });
  return labels.sort((a, b) => b.confidence - a.confidence).map(label => label.text).slice(0, 8);
}
