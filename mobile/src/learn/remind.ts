import { LocalNotifications } from '@capacitor/local-notifications';



const ID = 7001;
export async function remind(hour: number | null, languageName: string): Promise<boolean> {
  try {
    await LocalNotifications.cancel({ notifications: [{ id: ID }] });
    if (hour === null) return true;
    if ((await LocalNotifications.requestPermissions()).display !== 'granted') return false;
    await LocalNotifications.schedule({ notifications: [{ id: ID, title: 'Time to practice',
      body: `A few minutes of ${languageName} keeps your streak going.`, schedule: { on: { hour, minute: 0 } }, isExactNotification: false }] });
    return true;
  } catch {
    return false;
  }
}
