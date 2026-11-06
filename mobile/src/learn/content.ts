



export type LanguageId = 'ko' | 'de' | 'fr' | 'es' | 'pt' | 'ja' | 'zh-Hans' | 'zh-Hant';
export interface Language { id: LanguageId; name: string; native: string; speech: string; romanized: boolean }
export const LANGUAGES: Language[] = [
  { id: 'ko', name: 'Korean', native: '한국어', speech: 'ko-KR', romanized: true },
  { id: 'ja', name: 'Japanese', native: '日本語', speech: 'ja-JP', romanized: true },
  { id: 'zh-Hans', name: 'Chinese (Simplified)', native: '中文（简体）', speech: 'zh-CN', romanized: true },
  { id: 'zh-Hant', name: 'Chinese (Traditional)', native: '中文（繁體）', speech: 'zh-TW', romanized: true },
  { id: 'es', name: 'Spanish', native: 'Español', speech: 'es-MX', romanized: false },
  { id: 'fr', name: 'French', native: 'Français', speech: 'fr-FR', romanized: false },
  { id: 'de', name: 'German', native: 'Deutsch', speech: 'de-DE', romanized: false },
  { id: 'pt', name: 'Portuguese (Brazil)', native: 'Português', speech: 'pt-BR', romanized: false },
];
export const language = (id: LanguageId) => LANGUAGES.find(l => l.id === id)!;

type Romanized = [text: string, romanization: string];
export interface Phrase { id: string; en: string; forms: Record<LanguageId, { text: string; say: string; romanization?: string }> }


type Row = [string, Romanized, string, string, string, string, Romanized, Romanized, Romanized];
const same = (text: string, romanization: string): [Romanized, Romanized] => [[text, romanization], [text, romanization]];
const form = (value: string, romanization?: string) => {
  const [text, say] = value.split('|');
  return { text, say: say ?? text, ...(romanization ? { romanization } : {}) };
};

const TOPIC_ROWS: [id: string, title: string, rows: Row[]][] = [
  ['greetings', 'Greetings', [
    ['Hello', ['안녕하세요', 'annyeonghaseyo'], 'Hallo', 'Bonjour', 'Hola', 'Olá', ['こんにちは', 'konnichiwa'], ...same('你好', 'nǐ hǎo')],
    ['Good morning', ['좋은 아침이에요', 'joeun achimieyo'], 'Guten Morgen', 'Bonjour', 'Buenos días', 'Bom dia', ['おはようございます', 'ohayō gozaimasu'], ...same('早上好', 'zǎoshang hǎo')],
    ['Good evening', ['좋은 저녁이에요', 'joeun jeonyeogieyo'], 'Guten Abend', 'Bonsoir', 'Buenas noches', 'Boa noite', ['こんばんは', 'konbanwa'], ...same('晚上好', 'wǎnshang hǎo')],
    ['Goodbye', ['안녕히 가세요', 'annyeonghi gaseyo'], 'Auf Wiedersehen', 'Au revoir', 'Adiós', 'Tchau', ['さようなら', 'sayōnara'], ['再见', 'zàijiàn'], ['再見', 'zàijiàn']],
    ['Nice to meet you', ['만나서 반가워요', 'mannaseo bangawoyo'], 'Freut mich', 'Enchanté(e)|Enchanté', 'Mucho gusto', 'Muito prazer', ['はじめまして', 'hajimemashite'], ['很高兴认识你', 'hěn gāoxìng rènshi nǐ'], ['很高興認識你', 'hěn gāoxìng rènshi nǐ']],
    ['How are you?', ['잘 지내세요?', 'jal jinaeseyo?'], 'Wie geht es Ihnen?', 'Comment allez-vous ?', '¿Cómo está?', 'Como vai?', ['お元気ですか？', 'o-genki desu ka?'], ['你好吗？', 'nǐ hǎo ma?'], ['你好嗎？', 'nǐ hǎo ma?']],
    ["I'm fine, thank you", ['잘 지내요, 고마워요', 'jal jinaeyo, gomawoyo'], 'Mir geht es gut, danke', 'Je vais bien, merci', 'Estoy bien, gracias', 'Estou bem, obrigado(a)|Estou bem, obrigado', ['元気です、ありがとう', 'genki desu, arigatō'], ['我很好，谢谢', 'wǒ hěn hǎo, xièxie'], ['我很好，謝謝', 'wǒ hěn hǎo, xièxie']],
    ["What's your name?", ['이름이 뭐예요?', 'ireumi mwoyeyo?'], 'Wie heißen Sie?', 'Comment vous appelez-vous ?', '¿Cómo se llama?', 'Qual é o seu nome?', ['お名前は何ですか？', 'o-namae wa nan desu ka?'], ['你叫什么名字？', 'nǐ jiào shénme míngzi?'], ['你叫什麼名字？', 'nǐ jiào shénme míngzi?']],
    ['My name is Anna', ['제 이름은 안나예요', 'je ireumeun annayeyo'], 'Ich heiße Anna', "Je m'appelle Anna", 'Me llamo Anna', 'Meu nome é Anna', ['私の名前はアンナです', 'watashi no namae wa Anna desu'], ...same('我叫安娜', 'wǒ jiào Ānnà')],
    ['See you tomorrow', ['내일 봐요', 'naeil bwayo'], 'Bis morgen', 'À demain', 'Hasta mañana', 'Até amanhã', ['また明日', 'mata ashita'], ['明天见', 'míngtiān jiàn'], ['明天見', 'míngtiān jiàn']],
  ]],
  ['polite', 'Being polite', [
    ['Thank you', ['감사합니다', 'gamsahamnida'], 'Danke', 'Merci', 'Gracias', 'Obrigado(a)|Obrigado', ['ありがとう', 'arigatō'], ['谢谢', 'xièxie'], ['謝謝', 'xièxie']],
    ['Thank you very much', ['정말 감사합니다', 'jeongmal gamsahamnida'], 'Vielen Dank', 'Merci beaucoup', 'Muchas gracias', 'Muito obrigado(a)|Muito obrigado', ['どうもありがとうございます', 'dōmo arigatō gozaimasu'], ['非常感谢', 'fēicháng gǎnxiè'], ['非常感謝', 'fēicháng gǎnxiè']],
    ["You're welcome", ['천만에요', 'cheonmaneyo'], 'Bitte schön', 'De rien', 'De nada', 'De nada', ['どういたしまして', 'dō itashimashite'], ['不客气', 'bú kèqi'], ['不客氣', 'bú kèqi']],
    ['Please', ['부탁해요', 'butakaeyo'], 'Bitte', "S'il vous plaît", 'Por favor', 'Por favor', ['お願いします', 'onegai shimasu'], ['请', 'qǐng'], ['請', 'qǐng']],
    ['Excuse me', ['실례합니다', 'sillyehamnida'], 'Entschuldigung', 'Excusez-moi', 'Disculpe', 'Com licença', ['すみません', 'sumimasen'], ['打扰一下', 'dǎrǎo yíxià'], ['打擾一下', 'dǎrǎo yíxià']],
    ["I'm sorry", ['죄송합니다', 'joesonghamnida'], 'Es tut mir leid', 'Je suis désolé(e)|Je suis désolé', 'Lo siento', 'Desculpe', ['ごめんなさい', 'gomen nasai'], ['对不起', 'duìbuqǐ'], ['對不起', 'duìbuqǐ']],
    ['Yes', ['네', 'ne'], 'Ja', 'Oui', 'Sí', 'Sim', ['はい', 'hai'], ...same('是的', 'shì de')],
    ['No', ['아니요', 'aniyo'], 'Nein', 'Non', 'No', 'Não', ['いいえ', 'iie'], ...same('不是', 'bú shì')],
    ["It's okay", ['괜찮아요', 'gwaenchanayo'], 'Schon gut', "Ce n'est pas grave", 'Está bien', 'Tudo bem', ['大丈夫です', 'daijōbu desu'], ['没关系', 'méi guānxi'], ['沒關係', 'méi guānxi']],
    ["I don't understand", ['이해가 안 돼요', 'ihaega an dwaeyo'], 'Ich verstehe nicht', 'Je ne comprends pas', 'No entiendo', 'Não entendo', ['わかりません', 'wakarimasen'], ...same('我不明白', 'wǒ bù míngbai')],
  ]],
  ['conversation', 'Keeping a conversation going', [
    ['Do you speak English?', ['영어 하세요?', 'yeongeo haseyo?'], 'Sprechen Sie Englisch?', 'Parlez-vous anglais ?', '¿Habla inglés?', 'Você fala inglês?', ['英語を話せますか？', 'eigo o hanasemasu ka?'], ['你会说英语吗？', 'nǐ huì shuō Yīngyǔ ma?'], ['你會說英語嗎？', 'nǐ huì shuō Yīngyǔ ma?']],
    ['I speak a little {language}', ['한국어를 조금 해요', 'hangugeoreul jogeum haeyo'], 'Ich spreche ein bisschen Deutsch', 'Je parle un peu français', 'Hablo un poco de español', 'Falo um pouco de português', ['日本語が少し話せます', 'nihongo ga sukoshi hanasemasu'], ['我会说一点中文', 'wǒ huì shuō yìdiǎn Zhōngwén'], ['我會說一點中文', 'wǒ huì shuō yìdiǎn Zhōngwén']],
    ['Please speak slowly', ['천천히 말해 주세요', 'cheoncheonhi malhae juseyo'], 'Bitte sprechen Sie langsam', "Parlez lentement, s'il vous plaît", 'Hable despacio, por favor', 'Fale devagar, por favor', ['ゆっくり話してください', 'yukkuri hanashite kudasai'], ['请说慢一点', 'qǐng shuō màn yìdiǎn'], ['請說慢一點', 'qǐng shuō màn yìdiǎn']],
    ['Can you say that again?', ['다시 말해 주시겠어요?', 'dasi malhae jusigesseoyo?'], 'Können Sie das wiederholen?', 'Pouvez-vous répéter ?', '¿Puede repetirlo?', 'Pode repetir?', ['もう一度言ってください', 'mō ichido itte kudasai'], ['请再说一遍', 'qǐng zài shuō yí biàn'], ['請再說一遍', 'qǐng zài shuō yí biàn']],
    ['What does this mean?', ['이게 무슨 뜻이에요?', 'ige museun tteusieyo?'], 'Was bedeutet das?', "Qu'est-ce que ça veut dire ?", '¿Qué significa esto?', 'O que isso significa?', ['これはどういう意味ですか？', 'kore wa dō iu imi desu ka?'], ['这是什么意思？', 'zhè shì shénme yìsi?'], ['這是什麼意思？', 'zhè shì shénme yìsi?']],
    ['How do you say this?', ['이건 어떻게 말해요?', 'igeon eotteoke malhaeyo?'], 'Wie sagt man das?', 'Comment dit-on ça ?', '¿Cómo se dice esto?', 'Como se diz isso?', ['これは何と言いますか？', 'kore wa nan to iimasu ka?'], ['这个怎么说？', 'zhège zěnme shuō?'], ['這個怎麼說？', 'zhège zěnme shuō?']],
    ["I'm learning {language}", ['한국어를 배우고 있어요', 'hangugeoreul baeugo isseoyo'], 'Ich lerne Deutsch', "J'apprends le français", 'Estoy aprendiendo español', 'Estou aprendendo português', ['日本語を勉強しています', 'nihongo o benkyō shite imasu'], ['我在学中文', 'wǒ zài xué Zhōngwén'], ['我在學中文', 'wǒ zài xué Zhōngwén']],
    ['Please write it down', ['써 주세요', 'sseo juseyo'], 'Bitte schreiben Sie es auf', "Pouvez-vous l'écrire, s'il vous plaît ?", 'Escríbalo, por favor', 'Escreva, por favor', ['書いてください', 'kaite kudasai'], ['请写下来', 'qǐng xiě xiàlái'], ['請寫下來', 'qǐng xiě xiàlái']],
    ['I understand', ['알겠어요', 'algesseoyo'], 'Ich verstehe', 'Je comprends', 'Entiendo', 'Entendo', ['わかりました', 'wakarimashita'], ...same('我明白了', 'wǒ míngbai le')],
    ["I don't know", ['몰라요', 'mollayo'], 'Ich weiß nicht', 'Je ne sais pas', 'No sé', 'Não sei', ['知りません', 'shirimasen'], ...same('我不知道', 'wǒ bù zhīdào')],
  ]],
  ['food', 'Food and drink', [
    ["I'm hungry", ['배고파요', 'baegopayo'], 'Ich habe Hunger', "J'ai faim", 'Tengo hambre', 'Estou com fome', ['お腹がすきました', 'onaka ga sukimashita'], ['我饿了', 'wǒ è le'], ['我餓了', 'wǒ è le']],
    ['Water, please', ['물 주세요', 'mul juseyo'], 'Wasser, bitte', "De l'eau, s'il vous plaît", 'Agua, por favor', 'Água, por favor', ['水をください', 'mizu o kudasai'], ['请给我水', 'qǐng gěi wǒ shuǐ'], ['請給我水', 'qǐng gěi wǒ shuǐ']],
    ['The menu, please', ['메뉴판 주세요', 'menyupan juseyo'], 'Die Speisekarte, bitte', "La carte, s'il vous plaît", 'El menú, por favor', 'O cardápio, por favor', ['メニューをください', 'menyū o kudasai'], ['请给我菜单', 'qǐng gěi wǒ càidān'], ['請給我菜單', 'qǐng gěi wǒ càidān']],
    ['What do you recommend?', ['뭐 추천해 주시겠어요?', 'mwo chucheonhae jusigesseoyo?'], 'Was empfehlen Sie?', "Qu'est-ce que vous recommandez ?", '¿Qué me recomienda?', 'O que você recomenda?', ['おすすめは何ですか？', 'osusume wa nan desu ka?'], ['你推荐什么？', 'nǐ tuījiàn shénme?'], ['你推薦什麼？', 'nǐ tuījiàn shénme?']],
    ['This is delicious', ['맛있어요', 'masisseoyo'], 'Das ist lecker', "C'est délicieux", 'Está delicioso', 'Está delicioso', ['おいしいです', 'oishii desu'], ...same('很好吃', 'hěn hǎochī')],
    ["I don't eat meat", ['저는 고기를 안 먹어요', 'jeoneun gogireul an meogeoyo'], 'Ich esse kein Fleisch', 'Je ne mange pas de viande', 'No como carne', 'Eu não como carne', ['肉は食べません', 'niku wa tabemasen'], ...same('我不吃肉', 'wǒ bù chī ròu')],
    ["I'm allergic to nuts", ['견과류 알레르기가 있어요', 'gyeongwaryu allereugiga isseoyo'], 'Ich bin allergisch gegen Nüsse', "J'ai une allergie aux noix", 'Tengo alergia a los frutos secos', 'Tenho alergia a nozes', ['ナッツアレルギーがあります', 'nattsu arerugī ga arimasu'], ['我对坚果过敏', 'wǒ duì jiānguǒ guòmǐn'], ['我對堅果過敏', 'wǒ duì jiānguǒ guòmǐn']],
    ['The check, please', ['계산서 주세요', 'gyesanseo juseyo'], 'Die Rechnung, bitte', "L'addition, s'il vous plaît", 'La cuenta, por favor', 'A conta, por favor', ['お会計をお願いします', 'o-kaikei o onegai shimasu'], ['请结账', 'qǐng jiézhàng'], ['請結帳', 'qǐng jiézhàng']],
    ['A coffee, please', ['커피 한 잔 주세요', 'keopi han jan juseyo'], 'Einen Kaffee, bitte', "Un café, s'il vous plaît", 'Un café, por favor', 'Um café, por favor', ['コーヒーを一つください', 'kōhī o hitotsu kudasai'], ['请给我一杯咖啡', 'qǐng gěi wǒ yì bēi kāfēi'], ['請給我一杯咖啡', 'qǐng gěi wǒ yì bēi kāfēi']],
    ['Cheers!', ['건배!', 'geonbae!'], 'Prost!', 'Santé !', '¡Salud!', 'Saúde!', ['乾杯！', 'kanpai!'], ['干杯！', 'gānbēi!'], ['乾杯！', 'gānbēi!']],
  ]],
  ['shopping', 'Shopping', [
    ['How much is this?', ['이거 얼마예요?', 'igeo eolmayeyo?'], 'Wie viel kostet das?', 'Combien ça coûte ?', '¿Cuánto cuesta?', 'Quanto custa?', ['これはいくらですか？', 'kore wa ikura desu ka?'], ['这个多少钱？', 'zhège duōshao qián?'], ['這個多少錢？', 'zhège duōshao qián?']],
    ["That's too expensive", ['너무 비싸요', 'neomu bissayo'], 'Das ist zu teuer', "C'est trop cher", 'Es demasiado caro', 'É muito caro', ['高すぎます', 'takasugimasu'], ['太贵了', 'tài guì le'], ['太貴了', 'tài guì le']],
    ["I'll take it", ['이걸로 할게요', 'igeollo halgeyo'], 'Ich nehme es', 'Je le prends', 'Me lo llevo', 'Vou levar', ['これをください', 'kore o kudasai'], ['我要这个', 'wǒ yào zhège'], ['我要這個', 'wǒ yào zhège']],
    ['Can I pay by card?', ['카드로 계산해도 돼요?', 'kadeuro gyesanhaedo dwaeyo?'], 'Kann ich mit Karte bezahlen?', 'Je peux payer par carte ?', '¿Puedo pagar con tarjeta?', 'Posso pagar com cartão?', ['カードで払えますか？', 'kādo de haraemasu ka?'], ['可以刷卡吗？', 'kěyǐ shuākǎ ma?'], ['可以刷卡嗎？', 'kěyǐ shuākǎ ma?']],
    ["I'm just looking", ['그냥 구경하는 거예요', 'geunyang gugyeonghaneun geoyeyo'], 'Ich schaue nur', 'Je regarde seulement', 'Solo estoy mirando', 'Só estou olhando', ['見ているだけです', 'mite iru dake desu'], ...same('我只是看看', 'wǒ zhǐshì kànkan')],
    ['Do you have this in another color?', ['다른 색깔 있어요?', 'dareun saekkal isseoyo?'], 'Haben Sie das in einer anderen Farbe?', "Vous l'avez dans une autre couleur ?", '¿Lo tiene en otro color?', 'Tem em outra cor?', ['ほかの色はありますか？', 'hoka no iro wa arimasu ka?'], ['有别的颜色吗？', 'yǒu bié de yánsè ma?'], ['有別的顏色嗎？', 'yǒu bié de yánsè ma?']],
    ['Can I try it on?', ['입어 봐도 돼요?', 'ibeo bwado dwaeyo?'], 'Kann ich das anprobieren?', "Je peux l'essayer ?", '¿Me lo puedo probar?', 'Posso experimentar?', ['試着してもいいですか？', 'shichaku shite mo ii desu ka?'], ['我可以试穿吗？', 'wǒ kěyǐ shìchuān ma?'], ['我可以試穿嗎？', 'wǒ kěyǐ shìchuān ma?']],
    ['A bag, please', ['봉투 주세요', 'bongtu juseyo'], 'Eine Tüte, bitte', "Un sac, s'il vous plaît", 'Una bolsa, por favor', 'Uma sacola, por favor', ['袋をください', 'fukuro o kudasai'], ['请给我一个袋子', 'qǐng gěi wǒ yí ge dàizi'], ['請給我一個袋子', 'qǐng gěi wǒ yí ge dàizi']],
    ['Do you have a smaller size?', ['더 작은 사이즈 있어요?', 'deo jageun saijeu isseoyo?'], 'Haben Sie eine kleinere Größe?', 'Vous avez une taille plus petite ?', '¿Tiene una talla más pequeña?', 'Tem um tamanho menor?', ['もっと小さいサイズはありますか？', 'motto chiisai saizu wa arimasu ka?'], ['有小一点的吗？', 'yǒu xiǎo yìdiǎn de ma?'], ['有小一點的嗎？', 'yǒu xiǎo yìdiǎn de ma?']],
    ['Where can I pay?', ['어디서 계산해요?', 'eodiseo gyesanhaeyo?'], 'Wo kann ich bezahlen?', 'Où est-ce que je peux payer ?', '¿Dónde puedo pagar?', 'Onde posso pagar?', ['どこで払えますか？', 'doko de haraemasu ka?'], ['在哪里付钱？', 'zài nǎlǐ fù qián?'], ['在哪裡付錢？', 'zài nǎlǐ fù qián?']],
  ]],
  ['directions', 'Getting around', [
    ['Where is the bathroom?', ['화장실이 어디예요?', 'hwajangsiri eodiyeyo?'], 'Wo ist die Toilette?', 'Où sont les toilettes ?', '¿Dónde está el baño?', 'Onde fica o banheiro?', ['トイレはどこですか？', 'toire wa doko desu ka?'], ['洗手间在哪里？', 'xǐshǒujiān zài nǎlǐ?'], ['洗手間在哪裡？', 'xǐshǒujiān zài nǎlǐ?']],
    ['Where is the station?', ['역이 어디예요?', 'yeogi eodiyeyo?'], 'Wo ist der Bahnhof?', 'Où est la gare ?', '¿Dónde está la estación?', 'Onde fica a estação?', ['駅はどこですか？', 'eki wa doko desu ka?'], ['车站在哪里？', 'chēzhàn zài nǎlǐ?'], ['車站在哪裡？', 'chēzhàn zài nǎlǐ?']],
    ['How do I get to the airport?', ['공항에 어떻게 가요?', 'gonghange eotteoke gayo?'], 'Wie komme ich zum Flughafen?', "Comment aller à l'aéroport ?", '¿Cómo llego al aeropuerto?', 'Como chego ao aeroporto?', ['空港へはどう行けばいいですか？', 'kūkō e wa dō ikeba ii desu ka?'], ['去机场怎么走？', 'qù jīchǎng zěnme zǒu?'], ['去機場怎麼走？', 'qù jīchǎng zěnme zǒu?']],
    ['Turn left', ['왼쪽으로 가세요', 'oenjjogeuro gaseyo'], 'Biegen Sie links ab', 'Tournez à gauche', 'Gire a la izquierda', 'Vire à esquerda', ['左に曲がってください', 'hidari ni magatte kudasai'], ['向左转', 'xiàng zuǒ zhuǎn'], ['向左轉', 'xiàng zuǒ zhuǎn']],
    ['Turn right', ['오른쪽으로 가세요', 'oreunjjogeuro gaseyo'], 'Biegen Sie rechts ab', 'Tournez à droite', 'Gire a la derecha', 'Vire à direita', ['右に曲がってください', 'migi ni magatte kudasai'], ['向右转', 'xiàng yòu zhuǎn'], ['向右轉', 'xiàng yòu zhuǎn']],
    ['Go straight', ['쭉 가세요', 'jjuk gaseyo'], 'Gehen Sie geradeaus', 'Allez tout droit', 'Siga derecho', 'Siga em frente', ['まっすぐ行ってください', 'massugu itte kudasai'], ...same('一直走', 'yìzhí zǒu')],
    ['Is it far?', ['멀어요?', 'meoreoyo?'], 'Ist es weit?', "C'est loin ?", '¿Está lejos?', 'É longe?', ['遠いですか？', 'tōi desu ka?'], ['远吗？', 'yuǎn ma?'], ['遠嗎？', 'yuǎn ma?']],
    ["It's near here", ['여기서 가까워요', 'yeogiseo gakkawoyo'], 'Es ist hier in der Nähe', "C'est près d'ici", 'Está cerca de aquí', 'Fica perto daqui', ['この近くです', 'kono chikaku desu'], ...same('就在附近', 'jiù zài fùjìn')],
    ['One ticket, please', ['표 한 장 주세요', 'pyo han jang juseyo'], 'Eine Fahrkarte, bitte', "Un billet, s'il vous plaît", 'Un boleto, por favor', 'Uma passagem, por favor', ['切符を一枚ください', 'kippu o ichimai kudasai'], ['请给我一张票', 'qǐng gěi wǒ yì zhāng piào'], ['請給我一張票', 'qǐng gěi wǒ yì zhāng piào']],
    ['Where is the hotel?', ['호텔이 어디예요?', 'hoteri eodiyeyo?'], 'Wo ist das Hotel?', "Où est l'hôtel ?", '¿Dónde está el hotel?', 'Onde fica o hotel?', ['ホテルはどこですか？', 'hoteru wa doko desu ka?'], ['酒店在哪里？', 'jiǔdiàn zài nǎlǐ?'], ['飯店在哪裡？', 'fàndiàn zài nǎlǐ?']],
  ]],
  ['numbers', 'Numbers', [
    ['One', ['하나', 'hana'], 'eins', 'un', 'uno', 'um', ['いち', 'ichi'], ...same('一', 'yī')],
    ['Two', ['둘', 'dul'], 'zwei', 'deux', 'dos', 'dois', ['に', 'ni'], ...same('二', 'èr')],
    ['Three', ['셋', 'set'], 'drei', 'trois', 'tres', 'três', ['さん', 'san'], ...same('三', 'sān')],
    ['Four', ['넷', 'net'], 'vier', 'quatre', 'cuatro', 'quatro', ['よん', 'yon'], ...same('四', 'sì')],
    ['Five', ['다섯', 'daseot'], 'fünf', 'cinq', 'cinco', 'cinco', ['ご', 'go'], ...same('五', 'wǔ')],
    ['Six', ['여섯', 'yeoseot'], 'sechs', 'six', 'seis', 'seis', ['ろく', 'roku'], ...same('六', 'liù')],
    ['Seven', ['일곱', 'ilgop'], 'sieben', 'sept', 'siete', 'sete', ['なな', 'nana'], ...same('七', 'qī')],
    ['Eight', ['여덟', 'yeodeol'], 'acht', 'huit', 'ocho', 'oito', ['はち', 'hachi'], ...same('八', 'bā')],
    ['Nine', ['아홉', 'ahop'], 'neun', 'neuf', 'nueve', 'nove', ['きゅう', 'kyū'], ...same('九', 'jiǔ')],
    ['Ten', ['열', 'yeol'], 'zehn', 'dix', 'diez', 'dez', ['じゅう', 'jū'], ...same('十', 'shí')],
  ]],
  ['time', 'Days and time', [
    ['Today', ['오늘', 'oneul'], 'heute', "aujourd'hui", 'hoy', 'hoje', ['今日', 'kyō'], ...same('今天', 'jīntiān')],
    ['Tomorrow', ['내일', 'naeil'], 'morgen', 'demain', 'mañana', 'amanhã', ['明日', 'ashita'], ...same('明天', 'míngtiān')],
    ['Yesterday', ['어제', 'eoje'], 'gestern', 'hier', 'ayer', 'ontem', ['昨日', 'kinō'], ...same('昨天', 'zuótiān')],
    ['What time is it?', ['지금 몇 시예요?', 'jigeum myeot siyeyo?'], 'Wie spät ist es?', 'Quelle heure est-il ?', '¿Qué hora es?', 'Que horas são?', ['今何時ですか？', 'ima nanji desu ka?'], ['现在几点？', 'xiànzài jǐ diǎn?'], ['現在幾點？', 'xiànzài jǐ diǎn?']],
    ['Monday', ['월요일', 'woryoil'], 'Montag', 'lundi', 'lunes', 'segunda-feira', ['月曜日', 'getsuyōbi'], ...same('星期一', 'xīngqīyī')],
    ['Friday', ['금요일', 'geumyoil'], 'Freitag', 'vendredi', 'viernes', 'sexta-feira', ['金曜日', "kin'yōbi"], ...same('星期五', 'xīngqīwǔ')],
    ['Sunday', ['일요일', 'iryoil'], 'Sonntag', 'dimanche', 'domingo', 'domingo', ['日曜日', 'nichiyōbi'], ...same('星期天', 'xīngqītiān')],
    ['Morning', ['아침', 'achim'], 'der Morgen', 'le matin', 'la mañana', 'a manhã', ['朝', 'asa'], ...same('早上', 'zǎoshang')],
    ['Night', ['밤', 'bam'], 'die Nacht', 'la nuit', 'la noche', 'a noite', ['夜', 'yoru'], ...same('晚上', 'wǎnshang')],
    ['Now', ['지금', 'jigeum'], 'jetzt', 'maintenant', 'ahora', 'agora', ['今', 'ima'], ['现在', 'xiànzài'], ['現在', 'xiànzài']],
  ]],
  ['family', 'Family and people', [
    ['Mother', ['어머니', 'eomeoni'], 'die Mutter', 'la mère', 'la madre', 'a mãe', ['お母さん', 'okāsan'], ['妈妈', 'māma'], ['媽媽', 'māma']],
    ['Father', ['아버지', 'abeoji'], 'der Vater', 'le père', 'el padre', 'o pai', ['お父さん', 'otōsan'], ...same('爸爸', 'bàba')],
    ['My family', ['우리 가족', 'uri gajok'], 'meine Familie', 'ma famille', 'mi familia', 'minha família', ['私の家族', 'watashi no kazoku'], ...same('我的家人', 'wǒ de jiārén')],
    ['Son', ['아들', 'adeul'], 'der Sohn', 'le fils', 'el hijo', 'o filho', ['息子', 'musuko'], ['儿子', 'érzi'], ['兒子', 'érzi']],
    ['Daughter', ['딸', 'ttal'], 'die Tochter', 'la fille', 'la hija', 'a filha', ['娘', 'musume'], ['女儿', "nǚ'ér"], ['女兒', "nǚ'ér"]],
    ['Friend', ['친구', 'chingu'], 'ein Freund', 'un ami', 'un amigo', 'um amigo', ['友達', 'tomodachi'], ...same('朋友', 'péngyou')],
    ['Child', ['아이', 'ai'], 'das Kind', "l'enfant", 'el niño', 'a criança', ['子供', 'kodomo'], ...same('孩子', 'háizi')],
    ['I have two children', ['아이가 두 명 있어요', 'aiga du myeong isseoyo'], 'Ich habe zwei Kinder', "J'ai deux enfants", 'Tengo dos hijos', 'Tenho dois filhos', ['子供が二人います', 'kodomo ga futari imasu'], ['我有两个孩子', 'wǒ yǒu liǎng ge háizi'], ['我有兩個孩子', 'wǒ yǒu liǎng ge háizi']],
    ['This is my husband', ['제 남편이에요', 'je nampyeonieyo'], 'Das ist mein Mann', 'Voici mon mari', 'Este es mi esposo', 'Este é meu marido', ['夫です', 'otto desu'], ['这是我丈夫', 'zhè shì wǒ zhàngfu'], ['這是我丈夫', 'zhè shì wǒ zhàngfu']],
    ['This is my wife', ['제 아내예요', 'je anaeyeyo'], 'Das ist meine Frau', 'Voici ma femme', 'Esta es mi esposa', 'Esta é minha esposa', ['妻です', 'tsuma desu'], ['这是我妻子', 'zhè shì wǒ qīzi'], ['這是我妻子', 'zhè shì wǒ qīzi']],
  ]],
  ['help', 'Help and emergencies', [
    ['Help!', ['도와주세요!', 'dowajuseyo!'], 'Hilfe!', 'Au secours !', '¡Ayuda!', 'Socorro!', ['助けて！', 'tasukete!'], ...same('救命！', 'jiùmìng!')],
    ['Call the police', ['경찰을 불러 주세요', 'gyeongchareul bulleo juseyo'], 'Rufen Sie die Polizei!', 'Appelez la police !', '¡Llame a la policía!', 'Chame a polícia!', ['警察を呼んでください', 'keisatsu o yonde kudasai'], ['请叫警察', 'qǐng jiào jǐngchá'], ['請叫警察', 'qǐng jiào jǐngchá']],
    ['Call an ambulance', ['구급차를 불러 주세요', 'gugeupchareul bulleo juseyo'], 'Rufen Sie einen Krankenwagen!', 'Appelez une ambulance !', '¡Llame a una ambulancia!', 'Chame uma ambulância!', ['救急車を呼んでください', 'kyūkyūsha o yonde kudasai'], ['请叫救护车', 'qǐng jiào jiùhùchē'], ['請叫救護車', 'qǐng jiào jiùhùchē']],
    ['I need a doctor', ['의사가 필요해요', 'uisaga pillyohaeyo'], 'Ich brauche einen Arzt', "J'ai besoin d'un médecin", 'Necesito un médico', 'Preciso de um médico', ['医者が必要です', 'isha ga hitsuyō desu'], ['我需要医生', 'wǒ xūyào yīshēng'], ['我需要醫生', 'wǒ xūyào yīshēng']],
    ["I'm lost", ['길을 잃었어요', 'gireul ireosseoyo'], 'Ich habe mich verlaufen', 'Je me suis perdu(e)|Je me suis perdu', 'Me perdí', 'Me perdi', ['道に迷いました', 'michi ni mayoimashita'], ...same('我迷路了', 'wǒ mílù le')],
    ['Where is the hospital?', ['병원이 어디예요?', 'byeongwoni eodiyeyo?'], 'Wo ist das Krankenhaus?', "Où est l'hôpital ?", '¿Dónde está el hospital?', 'Onde fica o hospital?', ['病院はどこですか？', 'byōin wa doko desu ka?'], ['医院在哪里？', 'yīyuàn zài nǎlǐ?'], ['醫院在哪裡？', 'yīyuàn zài nǎlǐ?']],
    ['It hurts here', ['여기가 아파요', 'yeogiga apayo'], 'Es tut hier weh', "J'ai mal ici", 'Me duele aquí', 'Dói aqui', ['ここが痛いです', 'koko ga itai desu'], ['这里疼', 'zhèlǐ téng'], ['這裡疼', 'zhèlǐ téng']],
    ['I lost my bag', ['가방을 잃어버렸어요', 'gabangeul ireobeoryeosseoyo'], 'Ich habe meine Tasche verloren', "J'ai perdu mon sac", 'Perdí mi bolsa', 'Perdi minha bolsa', ['かばんをなくしました', 'kaban o nakushimashita'], ['我的包丢了', 'wǒ de bāo diū le'], ['我的包丟了', 'wǒ de bāo diū le']],
    ['Can you help me?', ['좀 도와주시겠어요?', 'jom dowajusigesseoyo?'], 'Können Sie mir helfen?', "Pouvez-vous m'aider ?", '¿Me puede ayudar?', 'Pode me ajudar?', ['手伝ってくれませんか？', 'tetsudatte kuremasen ka?'], ['你能帮我吗？', 'nǐ néng bāng wǒ ma?'], ['你能幫我嗎？', 'nǐ néng bāng wǒ ma?']],
    ['Be careful!', ['조심하세요!', 'josimhaseyo!'], 'Vorsicht!', 'Attention !', '¡Cuidado!', 'Cuidado!', ['気をつけて！', 'ki o tsukete!'], ...same('小心！', 'xiǎoxīn!')],
  ]],
];

export interface Topic { id: string; title: string; phrases: Phrase[] }
export const TOPICS: Topic[] = TOPIC_ROWS.map(([id, title, rows]) => ({
  id, title,
  phrases: rows.map(([en, ko, de, fr, es, pt, ja, zhHans, zhHant], i) => ({
    id: `${id}.${i + 1}`, en,
    forms: { ko: form(ko[0], ko[1]), de: form(de), fr: form(fr), es: form(es), pt: form(pt), ja: form(ja[0], ja[1]),
             'zh-Hans': form(zhHans[0], zhHans[1]), 'zh-Hant': form(zhHant[0], zhHant[1]) },
  })),
}));
export const PHRASES = TOPICS.flatMap(t => t.phrases);


export const english = (phrase: Phrase, id: LanguageId) =>
  phrase.en.replace('{language}', id.startsWith('zh') ? 'Chinese' : language(id).name.replace(/ \(.*\)$/, ''));


export interface Letter { letter: string; sound: string; say: string }
export interface Script { title: string; note: string; letters: Letter[] }
const kana = (chars: string, romaji: string): Letter[] => [...chars].map((letter, i) => ({ letter, sound: romaji.split(' ')[i], say: letter }));
const GOJUON = 'a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho ma mi mu me mo ya yu yo ra ri ru re ro wa wo n';
export const SCRIPTS: Partial<Record<LanguageId, Script[]>> = {
  ko: [
    { title: 'Hangul consonants', note: 'Each consonant shown with ㅏ (a) to make a syllable you can hear.',
      letters: [['ㄱ', 'g/k', '가'], ['ㄴ', 'n', '나'], ['ㄷ', 'd/t', '다'], ['ㄹ', 'r/l', '라'], ['ㅁ', 'm', '마'], ['ㅂ', 'b/p', '바'], ['ㅅ', 's', '사'],
                ['ㅇ', 'silent / ng', '아'], ['ㅈ', 'j', '자'], ['ㅊ', 'ch', '차'], ['ㅋ', 'k', '카'], ['ㅌ', 't', '타'], ['ㅍ', 'p', '파'], ['ㅎ', 'h', '하']]
        .map(([letter, sound, say]) => ({ letter, sound, say })) },
    { title: 'Hangul vowels', note: 'Written with the silent ㅇ in front, the way they appear on their own.',
      letters: [['ㅏ', 'a', '아'], ['ㅑ', 'ya', '야'], ['ㅓ', 'eo', '어'], ['ㅕ', 'yeo', '여'], ['ㅗ', 'o', '오'], ['ㅛ', 'yo', '요'], ['ㅜ', 'u', '우'],
                ['ㅠ', 'yu', '유'], ['ㅡ', 'eu', '으'], ['ㅣ', 'i', '이']].map(([letter, sound, say]) => ({ letter, sound, say })) },
  ],
  ja: [
    { title: 'Hiragana', note: 'For Japanese words and grammar.', letters: kana('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん', GOJUON) },
    { title: 'Katakana', note: 'The same sounds, for foreign words and names.', letters: kana('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン', GOJUON) },
  ],
  'zh-Hans': [{ title: 'The four tones', note: 'The same syllable "ma" means different things with each tone.',
    letters: [['妈', 'mā · high and level · mother'], ['麻', 'má · rising · hemp'], ['马', 'mǎ · low, dipping · horse'], ['骂', 'mà · falling · to scold'], ['吗', 'ma · light · question word']]
      .map(([letter, sound]) => ({ letter, sound, say: letter })) }],
  'zh-Hant': [{ title: 'The four tones', note: 'The same syllable "ma" means different things with each tone.',
    letters: [['媽', 'mā · high and level · mother'], ['麻', 'má · rising · hemp'], ['馬', 'mǎ · low, dipping · horse'], ['罵', 'mà · falling · to scold'], ['嗎', 'ma · light · question word']]
      .map(([letter, sound]) => ({ letter, sound, say: letter })) }],
  de: [{ title: 'Special letters', note: 'Listen for the sound in each word.',
    letters: [['ä', 'Mädchen (girl)', 'Mädchen'], ['ö', 'schön (beautiful)', 'schön'], ['ü', 'über (over)', 'über'], ['ß', 'Straße (street), like ss', 'Straße'], ['ch', 'ich (I)', 'ich'], ['ei', 'nein (no), like "eye"', 'nein']]
      .map(([letter, sound, say]) => ({ letter, sound, say })) }],
  fr: [{ title: 'Special letters', note: 'Listen for the sound in each word.',
    letters: [['é', 'café', 'café'], ['è', 'très (very)', 'très'], ['ê', 'forêt (forest)', 'forêt'], ['ç', 'garçon (boy), like s', 'garçon'], ['on', 'bonjour, through the nose', 'bonjour'], ['r', 'merci, at the back of the throat', 'merci']]
      .map(([letter, sound, say]) => ({ letter, sound, say })) }],
  es: [{ title: 'Special letters', note: 'Listen for the sound in each word.',
    letters: [['ñ', 'niño (child), like ny', 'niño'], ['ll', 'llamar (to call), like y', 'llamar'], ['rr', 'perro (dog), rolled', 'perro'], ['j', 'jamón (ham), like a strong h', 'jamón'], ['h', 'hola, silent', 'hola']]
      .map(([letter, sound, say]) => ({ letter, sound, say })) }],
  pt: [{ title: 'Special letters', note: 'Listen for the sound in each word.',
    letters: [['ã', 'mão (hand), through the nose', 'mão'], ['ç', 'coração (heart), like s', 'coração'], ['lh', 'filho (son), like lli in million', 'filho'], ['nh', 'vinho (wine), like ny', 'vinho'], ['r', 'Rio, like h at the start', 'Rio']]
      .map(([letter, sound, say]) => ({ letter, sound, say })) }],
};
