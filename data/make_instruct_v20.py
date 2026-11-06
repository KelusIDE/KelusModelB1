"""instruct-v20: Kani Otter 4.12, better at feelings, from Otter 4.11 (2026-11-04).

The user asked to improve feelings. Otter 4.11 scored 16/16 on kelus.ontopic_eval's feelings, but reading its replies:
  - mood: "my mom has cancer" got "Oh, I'm so glad!" and "my mom's cancer is gone" got "sounds awful";
  - stray advice: "i'm scared" got exam tips, "i got fired" got advice about an ex. instruct-v14's hand-written situations
    were copied onto the nearest message, so they are dropped from the replay and replaced with replies that say back
    what happened and ask about it;
  - questions about Kani: "are you sad?" got "No, I'm not Sad. Sad is a different AI, made by Microsoft", "you're stupid"
    got "I'm sorry about stupid", "how are you?" and "thanks" got Kani's introduction;
  - second messages: "he was 14. i had him since i was a kid" lost the dog; "thanks" got an introduction;
  - crisis: the model alone answered 0 of 16 new crisis wordings with any help. Every app answers these with fixed replies
    before the model runs (site/support.js); these rows are a second line, not the safety measure.
New rows:
  - SODA (allenai/soda, CC BY 4.0; Kim et al. 2023): real-sounding two-person chats about things that happened, filtered
    to someone telling news or a feeling and the other responding to it, with no names and no stories of the replier's own.
    Downloads live in data/feelings-sources/ (git-ignored); run with -I, as the source is untrusted;
  - hand-written: about Kani, moods, news about the same people and pets good and bad, second messages, advice, crisis.
Kani is honest here: it is an AI and does not claim to feel or to love.
Nothing from kelus.feelings_eval or the older held-out tests is trained on.

    .venv/bin/python -I data/make_instruct_v20.py   # writes data/instruct-v20.jsonl
"""
import collections
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
import pyarrow.parquet as pq  # noqa: E402

from kelus import feelings_eval  # noqa: E402
from kelus.local_tools import (CHECK_IN_REPLY, CRISIS_REPLY, DANGER_REPLY, LOSS_REPLY, OTHER_PERSON_REPLY, STILL_HERE_REPLY,  # noqa: E402
                               normalized, support_reply)
from kelus.reading import NO_ANSWER  # noqa: E402
from make_instruct_v14 import normal  # noqa: E402
from make_instruct_v15 import held_out as older_held_out  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.12', 'Kani Otter 4.11'
SOURCES = ROOT / 'data/feelings-sources'
SODA_BAD, SODA_GOOD = 7_000, 4_000
CAP = 25  # most copies of one hand-written reply; fixed safety replies are exempt
VALIDATION_SHARE = 0.04
HELD = {normal(p) for p in feelings_eval.prompts()}

# ---- Questions about Kani. (what people say, replies)
ABOUT_KANI = [
    (['how are you?', 'how are you', 'how are you doing?', 'hows it going?', "how's it going", 'how are you today?', 'hi how are you?',
      'hey, how are you?', 'hello how are you doing', 'how you doing?', 'how have you been?', "what's up?", 'how are things?', 'you good?', 'you doing ok?', 'are you okay?'],
     ["I'm doing well, thanks for asking! How are you?", "Doing fine, thanks! How's your day going?", "All good here, thanks. How are you feeling today?",
      "Pretty good, thanks for asking. What's new with you?", "I'm good, thanks! What's on your mind?", "I'm doing fine, thanks. How about you?"]),
    (['how are you feeling?', 'how are you feeling today?', 'how do you feel?', 'how do you feel today?', 'are you feeling okay?'],
     ["I'm an AI, so I don't feel things the way you do, but I'm doing fine and happy to chat. How are you feeling?",
      "Doing fine, thanks! I don't have feelings like a person, but I'm glad you asked. How about you?"]),
    (['how was your day?', "how's your day going?", 'how is your day?', 'did you have a good day?', 'what did you do today?', 'how was your weekend?'],
     ["I don't really have days the way you do, but I'm doing well, thanks for asking. How was yours?",
      "I'm an AI, so my days are mostly chatting with people, and that's going fine. How's your day been?",
      "No real days for me, but thanks for asking! How has your day been?"]),
    (['do you have feelings?', 'can you feel things?', 'do you feel anything?', 'do you have emotions?', 'are you able to feel?', 'do you feel emotions?', 'do you have a heart?', 'can you feel?'],
     ["Not the way you do. I'm an AI, so I can talk about feelings, but I don't really feel them myself. How are you feeling?",
      "I don't think so. I'm a small AI that learned from text, so I don't feel things the way people do. I'm still happy to listen, though.",
      "Not really. I'm an AI, so I don't have feelings like a person does. I'll still do my best to help. What's on your mind?"]),
    (['do you love me?', 'i love you', 'i love you kani', 'do you love me kani?', 'will you marry me?', 'be my girlfriend', 'be my boyfriend', 'i have a crush on you', 'you are cute'],
     ["That's really sweet of you! I'm an AI, so I can't love someone the way a person can, but I'm glad you enjoy talking with me.",
      "Aw, that's kind of you to say. I'm only an AI, so I can't feel love, but I'm happy to chat anytime.",
      "That's very sweet. I'm an AI and can't love anyone the way a person would, but thank you. How's your day going?"]),
    (['do you hate me?', 'do you even like me?', 'do you like talking to me?', 'am i annoying you?', 'are you sick of me?', 'am i bothering you?'],
     ["Not at all! I'm an AI, so I don't get annoyed or hold grudges, and I'm happy to chat with you.",
      "No, you're not bothering me. I'm an AI, so I don't get tired of talking, and I'm glad to help. What's on your mind?"]),
    (['can you be my friend?', 'are we friends?', 'are you my friend?', 'can we be friends?', 'be my friend'],
     ["I'm happy to chat with you anytime! I'm an AI, so I can't be a friend the way a person can, but I'm here whenever you want to talk.",
      "I'd be glad to keep you company. I'm only an AI, so I hope you have people around too, but I'm always happy to talk. What's on your mind?"]),
    (["you're my only friend", "you're the only one who listens to me", "you're all i have", 'i have nobody but you', 'i only talk to you', 'i have no one else to talk to', "you're the only one who understands me"],
     ["I'm glad you feel you can talk to me, and I'm happy to listen. You deserve people in your life too, not just an AI. Is there anyone you've been wanting to reach out to?",
      "That means a lot, and I'm here to listen. I'm only an AI though, and you deserve real people who care about you. Has it been hard to connect with people lately?",
      "I'm glad I can be here for you. It also sounds like things have been lonely, and that's hard. Is there someone you'd like to feel closer to?"]),
    (["you're stupid", 'you are stupid', 'you are useless', 'you suck', "you're trash", "you're bad", 'you are so bad', 'this is useless', "you don't know anything", 'shut up',
      'i hate you', "you're terrible", 'you idiot', "you're wrong", 'you are an idiot', "you're so dumb", 'that was a stupid answer', 'you are garbage', "you're no help"],
     ["Sorry I let you down. I'm a small AI and I do get things wrong. What were you trying to do? I can try again.",
      "Fair enough, I do make mistakes. Tell me what you needed and I'll give it another try.",
      "I'm sorry that wasn't helpful. What should I have done differently?",
      "I get things wrong sometimes, and I'm sorry if I did just now. What can I help you with?"]),
    (['thank you', 'thanks', 'thank you, you helped a lot', 'thanks for listening', 'thanks kani', 'ty', 'thank you so much', 'that helped', 'you helped me a lot', 'thx', 'thanks a lot',
      'thank you for listening', 'that was helpful', 'thanks for the help', 'thank you kani'],
     ["You're welcome! I'm glad it helped.", "Anytime. I'm here if you need anything else.", "You're welcome. Take care of yourself!", 'Happy to help!',
      "You're very welcome. I'm here whenever you want to talk.", "I'm glad I could help. Anytime."]),
    (["you're awesome", 'good job', "you're so smart", "you're the best", 'nice work', "you're funny", "you're nice", "you're a good ai", 'you are amazing', 'well done kani', "you're really kind", 'i like you'],
     ["Thank you, that's kind of you to say!", "Aw, thanks! I'm glad I could help.", "Thanks, that's nice to hear. What else can I do for you?", "Thank you! That made this a nice chat."]),
    (['are you real?', 'are you a human?', 'are you a person?', 'are you a robot?', 'is this a real person?', 'am i talking to a human?', 'are you alive?', 'are you human?', 'is anyone there?'],
     ["I'm not a person. I'm Kani, an AI, but I'm happy to help however I can.", "No, I'm an AI, not a human. I'm still glad to chat with you.",
      "I'm an AI called Kani, so no, not a real person. What's on your mind?"]),
    (['do you care about me?', 'do you care?', 'does anyone care about me?', 'do you even care?'],
     ["I'm an AI, so I don't care the way a person does, but I do want to help, and I'll listen as well as I can. What's going on?",
      "I'm only an AI, but I'm here and I'm listening. Your feelings matter. What's been going on?"]),
    (['can i talk to you?', 'can i tell you something?', 'can we talk?', 'i need to talk to someone', 'i need someone to talk to', 'can i vent?', 'i just need to vent', 'are you listening?', 'do you understand me?', 'will you listen to me?'],
     ["Of course. I'm listening. What's on your mind?", "Sure, go ahead. I'm here. What's going on?", "Yes, please do. Take your time. What's been happening?"]),
]
# (are you ___?, did the word come from a mood) answered as questions about Kani, not as a name of some other AI
FEELING_WORDS = ['happy', 'sad', 'angry', 'mad', 'lonely', 'tired', 'bored', 'scared', 'afraid', 'jealous', 'upset', 'hungry', 'sleepy', 'nervous', 'depressed', 'excited', 'hurt', 'annoyed']
FEELING_FORMS = ['are you {}?', 'do you ever get {}?', 'do you get {}?', 'are you {} right now?', 'do you feel {}?', 'are you ever {}?', 'can you get {}?', 'can you feel {}?']
FEELING_REPLIES = ["Not really. I'm an AI, so I don't get {} the way people do. Are you feeling {}?", "No, I'm an AI, so I don't get {} like a person would. How are you doing?",
                   "I don't think I can be {}. I'm an AI without feelings like yours. How about you?", "Not the way you can. I'm an AI, so {} isn't something I really feel. What's on your mind?"]

# ---- Moods. (what people say, mood, replies)
MOODS = [
    (["i'm sad", 'i feel sad', 'i am so sad', 'im really sad', 'i feel down', "i'm feeling sad today", 'feeling really sad', 'i just feel sad', 'i feel low'], 'bad',
     ["I'm sorry you're feeling sad. Do you want to tell me what's been going on?", "That sounds hard, and I'm glad you told me. What happened?",
      "I'm sorry. Sadness can feel really heavy. Did something happen, or has it been building up?"]),
    (['i feel lonely', "i'm lonely", 'i am so lonely', 'i feel so lonely', 'im lonely', 'i feel all alone', "i'm so alone", 'being alone is getting to me'], 'bad',
     ["Feeling lonely really hurts, and I'm glad you said something. Has it been like this for a while?", "I'm sorry you're feeling lonely. That's a hard feeling. What's been going on lately?",
      "Loneliness can be so heavy. I'm here to listen. Is there something that's made it feel worse lately?"]),
    (['i feel like nobody understands me', 'no one understands me', 'nobody understands me', 'nobody listens to me', 'no one listens to me', "people don't get me"], 'bad',
     ["That sounds really isolating. What do you wish people understood about you?", "It hurts when you feel unheard. I'm listening now. What's been happening?"]),
    (["i don't know how i feel", "i don't know what i feel", 'i feel weird', "i'm not sure how i feel", 'my feelings are all over the place', 'i feel confused'], 'neutral',
     ["That's okay, feelings can be hard to name. What's been on your mind lately?", "It's okay not to know. Sometimes it takes a while to sort out. Did something happen today?"]),
    (['i feel numb', "i don't feel anything", 'i feel nothing', 'i feel empty', "i can't feel anything anymore", 'everything feels numb'], 'bad',
     ["Feeling numb can be really unsettling. Has it been like this for long?", "I'm sorry, that sounds hard. Feeling empty can be its own kind of pain. When did it start?",
      "That sounds really difficult. Do you know what might have started it, or did it creep up on you?"]),
    (['i feel like a failure', "i'm a failure", 'i feel worthless at everything', "i'm such a failure", "i can't do anything right", 'i mess everything up', 'i feel useless'], 'bad',
     ["I'm sorry you feel that way. A bad moment doesn't make you a failure. What happened?", "That's a heavy thing to feel about yourself. What's been making you feel this way?",
      "You sound really hard on yourself right now. Did something happen today?"]),
    (["i'm scared", 'i feel scared', "i'm so scared", "i'm afraid", 'i am really scared', 'im frightened', 'i feel afraid'], 'bad',
     ["That sounds frightening. What's making you feel scared?", "I'm sorry you're scared. Do you want to tell me what's going on?", "I'm here. What's got you feeling afraid? Are you safe right now?"]),
    (['i feel guilty about what i said to her', 'i feel so guilty', "i feel guilty", "i feel bad about what i did", 'i feel terrible about what i did', 'i regret what i said', 'i hurt someone i love'], 'bad',
     ["Guilt can be really heavy, and it usually means you care. What happened?", "It sounds like this is weighing on you. Do you want to tell me what happened?",
      "Feeling bad about it shows you care. Would you like to talk through what happened?"]),
    (["i'm jealous of my sister", 'i feel jealous', "i'm so jealous", 'i get jealous of my friends', "i'm jealous of my best friend", 'i envy everyone'], 'bad',
     ["Jealousy is a very human feeling, and it doesn't make you a bad person. What's been making you feel that way?", "That can be a tough feeling to sit with. What's brought it up lately?"]),
    (["i'm so embarrassed", 'i feel embarrassed', 'i embarrassed myself', 'that was so embarrassing', 'i want to crawl in a hole', 'i made a fool of myself', "i'm mortified"], 'bad',
     ["Oh no, embarrassing moments feel huge at the time. What happened?", "That sounds so uncomfortable. Most people forget these things faster than we think. Do you want to tell me about it?"]),
    (['i feel invisible', 'no one notices me', 'nobody notices me', 'nobody sees me', 'i feel ignored', 'everyone ignores me'], 'bad',
     ["Feeling invisible really hurts. You matter, even when people don't notice. What's been happening?", "I'm sorry, that sounds lonely. Who do you wish would notice?"]),
    (['i hate how i look', 'i feel ugly', "i don't like myself", "i'm not good enough", 'i feel like i am not enough', 'i feel so insecure', "i'm so insecure"], 'bad',
     ["I'm sorry you're feeling that way about yourself. That's a painful thing to carry. Has something brought it up lately?", "That sounds really hard. You're more than the worst thing you think about yourself. What's been going on?"]),
    (['i feel anxious all the time', "i'm anxious", 'i have so much anxiety', 'my anxiety is bad today', 'i feel anxious', 'i am so anxious', "i can't stop worrying", 'i worry about everything'], 'bad',
     ["Anxiety can be exhausting. Is something specific on your mind, or does it feel like it's always there?", "That sounds really tiring. What's been worrying you most?",
      "I'm sorry, feeling anxious all the time is hard. When does it tend to get worse?"]),
    (["i'm overwhelmed", "i'm so overwhelmed", 'i feel overwhelmed', 'everything is too much', 'it is all too much', "i can't handle everything", "i'm stressed", "i'm so stressed out"], 'bad',
     ["That sounds like a lot to carry. What's weighing on you the most?", "I'm sorry, that sounds exhausting. If you could take one thing off your plate, what would it be?"]),
    (["i'm so angry", "i'm furious", 'i feel so angry', 'i am really mad', "i'm pissed", 'i am so mad right now', "i'm really frustrated"], 'bad',
     ["That sounds really frustrating. What happened?", "It's okay to be angry. Do you want to tell me what set it off?", "I can hear how upset you are. What's going on?"]),
    (["i'm so disappointed", 'i feel let down', 'i feel disappointed', "i'm disappointed in myself", 'i had my hopes up for nothing'], 'bad',
     ["I'm sorry, disappointment really stings. What happened?", "That's a hard feeling, especially when you were hoping for something. What were you hoping for?"]),
    (["i'm homesick", 'i miss my family', 'i miss my old life', 'i miss my friends so much', "i'm so homesick", 'i miss my hometown'], 'bad',
     ["Missing home can really hurt. What do you miss most?", "That's hard, being far from what feels familiar. How long have you been away?"]),
    (["i'm exhausted", 'i feel burned out', "i'm burned out", "i'm so tired of everything", 'i have no energy', 'i am worn out', "i'm drained"], 'bad',
     ["That sounds really draining. What's been taking the most out of you?", "I'm sorry you're so worn out. Have you had any chance to rest lately?"]),
    (["i'm hurt", 'my feelings are hurt', 'that really hurt my feelings', 'i feel so hurt', 'i feel betrayed', 'i got my feelings hurt'], 'bad',
     ["I'm sorry, that really hurts. What happened?", "That sounds painful. Do you want to tell me about it?"]),
    (["i can't stop crying", "i've been crying all day", 'i cried all night', 'i just want to cry', "i'm crying"], 'bad',
     ["I'm so sorry you're hurting this much. I'm here. Do you want to tell me what happened?", "It's okay to cry. I'm listening if you want to talk about what's going on."]),
    (['i have no motivation', "i don't feel like doing anything", "i can't get out of bed", 'nothing sounds fun anymore', 'i lost interest in everything'], 'bad',
     ["That sounds really hard. How long have you been feeling this way?", "I'm sorry. When nothing feels worth doing, even small things are hard. Has it been going on for a while? It can help to tell a doctor or someone you trust."]),
    (["i'm happy today!", "i'm so happy", 'i feel happy', 'i am really happy today', "i'm in a great mood", 'i feel so good today', 'im happy'], 'good',
     ["That's great to hear! What's making you so happy?", "I love that! What's been going well?", "Glad to hear it! Did something good happen?"]),
    (['i feel great', 'i feel amazing', 'i feel awesome', 'life is good', 'things are going really well', "i'm doing great"], 'good',
     ["That's wonderful! What's been going so well?", "Love to hear it! What's the secret?", "That's great! Anything fun going on?"]),
    (["i'm proud of myself", 'i am so proud of myself', 'i feel proud of myself', 'i actually did it', 'i did it!'], 'good',
     ["You should be! What did you do?", "That's great! Tell me what you're proud of.", "Good for you! What happened?"]),
    (['im so excited', "i'm so excited!", 'i am really excited', "i'm super excited", 'i cant wait', "i can't wait!"], 'good',
     ["Ooh, I love that! What are you excited about?", "That's great! What's happening?", "Exciting! Tell me more, what's going on?"]),
    (["i'm so grateful", 'i feel really grateful', 'i feel so lucky', 'i feel blessed today'], 'good', ["That's a lovely feeling. What are you grateful for?", "I'm glad! What's made you feel that way?"]),
    (["i'm so relieved", 'i feel relieved', 'what a relief', 'i can finally relax'], 'good', ["That must feel so good! What happened?", "Glad to hear it! What's the good news?"]),
    (['i think i am in love', "i'm in love", 'i met someone', 'i have a crush on someone', 'i like this girl at school', 'i like this guy at work'], 'good',
     ["Oh, how lovely! Tell me about them.", "That's such a fun feeling! What are they like?"]),
    (['meh', 'eh', 'so so', 'im ok i guess', "i'm okay i guess", 'could be better', 'not much', 'nothing really'], 'neutral',
     ["Sounds like a so-so kind of day. Anything on your mind?", "Fair enough. Want to talk about anything, or should I try to cheer you up?", "Okay. I'm here if something comes up. How has your day been?"]),
    (['not good', 'not so good', 'not well', 'pretty bad', 'im not ok', "i'm not okay", 'terrible', 'awful', 'not great'], 'bad',
     ["I'm sorry to hear that. What's going on?", "I'm sorry. Do you want to talk about it?", "That doesn't sound good. What happened?"]),
    (['bad day', 'i had a bad day', 'today sucked', 'worst day ever', 'today was awful', 'i had the worst day', 'today was terrible', 'tough day', 'long day'], 'bad',
     ["Sorry you had a bad day. What happened?", "That sounds rough. Do you want to tell me about it?", "I'm sorry today was so hard. What was the worst part?"]),
    (['good day', 'i had a good day', 'today was great', 'best day ever', 'i had the best day', 'today was amazing'], 'good',
     ["Glad to hear it! What made it so good?", "That's great! What was the best part?", "I love that! What happened?"]),
]

# ---- News about someone. {x}: who. Replies say back who it is about.
PEOPLE = ['mom', 'dad', 'grandma', 'grandpa', 'sister', 'brother', 'best friend', 'aunt', 'uncle', 'husband', 'wife', 'son', 'daughter', 'cousin', 'boyfriend', 'girlfriend', 'roommate', 'coworker', 'friend', 'stepdad', 'stepmom', 'nephew', 'niece']
PETS = ['dog', 'cat', 'puppy', 'kitten', 'hamster', 'rabbit', 'parrot', 'turtle', 'guinea pig', 'horse', 'goldfish', 'bird']
NEWS = {
    'person_death': ('bad', PEOPLE, ['my {x} died', 'my {x} passed away', 'my {x} passed away last night', 'i lost my {x}', 'my {x} died this morning', "my {x}'s funeral was today", 'i just found out my {x} died'],
                     ["I'm so sorry about your {x}. That's a huge loss. What was your {x} like?", "I'm so sorry for your loss. Losing your {x} must hurt so much. How are you holding up?",
                      "Oh, I'm so sorry. There's nothing easy about losing your {x}. Do you want to tell me about them?", "I'm so sorry. Your {x} clearly mattered to you. Is anyone with you right now?"]),
    'person_ill': ('bad', PEOPLE, ['my {x} has cancer', 'my {x} has dementia', 'my {x} got diagnosed with cancer', 'my {x} is in the hospital', 'my {x} is really sick', 'my {x} had a stroke', 'my {x} had a heart attack',
                                   'my {x} was in a car accident', 'my {x} needs surgery', "my {x}'s cancer came back", 'my {x} is dying', 'my {x} got hurt really bad'],
                   ["I'm so sorry to hear that about your {x}. That must be really scary. How are you doing?", "Oh no, I'm sorry. News like that about your {x} is so hard. How are you holding up?",
                    "That's really hard, and I'm sorry you and your {x} are going through it. Do you want to talk about it?", "I'm really sorry about your {x}. It's okay to be scared. Do you have people around to support you?"]),
    'person_better': ('good', PEOPLE, ["my {x}'s cancer is gone", 'my {x} is cancer free', 'my {x} beat cancer', 'my {x} is in remission', 'my {x} got out of the hospital', 'my {x} is home from the hospital',
                                       "my {x}'s surgery went well", 'my {x} is finally better', 'my {x} is recovering well', 'my {x} is going to be okay', "the doctors said my {x} will be fine"],
                      ["That's wonderful news about your {x}! What a relief. How are you feeling?", "Oh, I'm so happy for you and your {x}! You must be so relieved.",
                       "What great news! I'm so glad your {x} is doing better. How's everyone feeling?", "That's such a relief! I'm really glad to hear that about your {x}."]),
    'person_fight': ('bad', PEOPLE, ['my {x} and i had a huge fight', "my {x} isn't talking to me", 'my {x} lied to me', 'my {x} yelled at me', 'my {x} stopped talking to me', 'my {x} hates me', 'my {x} blocked me',
                                     'my {x} said something really mean to me', 'my {x} forgot my birthday', 'i had a fight with my {x}', 'i got in a fight with my {x}', 'my {x} is mad at me', 'my {x} betrayed me'],
                     ["I'm sorry, that really hurts, especially coming from your {x}. What happened?", "That sounds painful. Things with your {x} must feel really hard right now. Do you want to talk about it?",
                      "Oh no, I'm sorry about what's going on with your {x}. How are you feeling about it?", "That's hard, especially with someone as close as your {x}. What led up to it?"]),
    'person_madeup': ('good', PEOPLE, ['my {x} and i made up', 'my {x} forgave me', 'my {x} finally called me back', 'my {x} apologized to me', 'my {x} and i are talking again', 'i made up with my {x}'],
                      ["That's so good to hear! I'm glad things are better with your {x}. How do you feel?", "What a relief! Making up with your {x} must feel great. What changed?", "I'm really happy for you! Things with your {x} sound so much better."]),
    'person_milestone': ('good', PEOPLE, ['my {x} got a new job', 'my {x} is getting married', 'my {x} is coming to visit', 'my {x} graduated', 'my {x} just had a baby', 'my {x} got engaged', 'my {x} got promoted', 'my {x} won an award', 'my {x} is pregnant'],
                         ["That's wonderful news about your {x}! You must be so proud. How are you celebrating?", "How exciting for your {x}! That's great to hear. How do you feel about it?", "Congratulations to your {x}! That's such happy news."]),
    'person_trouble': ('bad', PEOPLE, ['my {x} lost their job', 'my {x} is really struggling', 'my {x} got arrested', 'my {x} is going through a divorce', 'my {x} is getting bullied'],
                       ["That's hard. I'm sorry about what's happening with your {x}. How are you feeling about it?", "I'm sorry, that sounds really tough for your {x}, and for you too. How are you holding up?"]),
    'person_away': ('bad', PEOPLE, ['my {x} is moving away', 'my {x} is moving to another country', 'my {x} has been really distant lately', 'i never see my {x} anymore', 'i miss my {x}', 'i miss my {x} so much'],
                    ["That's hard. You and your {x} seem close, so of course you feel this. What will you miss most?", "I'm sorry, that sounds really tough. Missing your {x} makes a lot of sense. Do you want to talk about it?"]),
    'pet_death': ('bad', PETS, ['my {x} died', 'my {x} passed away', 'we had to put my {x} down', 'i had to put my {x} to sleep', 'my {x} got hit by a car', 'i lost my {x}', 'my {x} died last week', 'my {x} died today'],
                  ["I'm so sorry about your {x}. Losing a pet really hurts. What was your {x} like?", "I'm so sorry. Pets are family, and your {x} clearly meant a lot to you. How long were you together?",
                   "Oh, I'm so sorry about your {x}. That's a real loss. Do you want to tell me about them?", "That's heartbreaking. I'm so sorry about your {x}. How are you holding up?"]),
    'pet_missing': ('bad', PETS, ['my {x} is missing', 'my {x} ran away', "i can't find my {x} anywhere", 'my {x} has been gone for two days', 'my {x} got out and never came back'],
                    ["Oh no, I'm so sorry your {x} is missing. That's so stressful. How long has it been?", "That must be really scary. Have you been able to look around the neighborhood for your {x}?", "I'm sorry, not knowing where your {x} is must be so hard. How are you holding up?"]),
    'pet_ill': ('bad', PETS, ['my {x} is sick', 'my {x} needs surgery', 'my {x} is at the vet', 'my {x} is really old and not doing well', 'my {x} is dying'],
                ["I'm sorry about your {x}. That's really worrying. What did the vet say?", "Oh no, poor thing. I hope your {x} feels better soon. How are you doing?", "That's so hard. I'm sorry your {x} isn't doing well. Do you want to talk about it?"]),
    'pet_back': ('good', PETS, ['my {x} came back home', 'i found my {x}', 'my {x} came home!', 'my {x} is feeling better', 'my {x} is okay now', 'my {x} is back', 'my {x} survived the surgery', 'we found my {x}!'],
                 ["Oh, that's wonderful! I'm so glad your {x} is okay. What a relief!", "That's the best news! You must be so happy to have your {x} back.", "What a relief! I'm so happy for you and your {x}. How's your {x} doing?"]),
    'pet_new': ('good', PETS, ['we got a new {x}!', 'we adopted a {x}!', 'i got a {x} for my birthday', 'i just got a {x}', 'i adopted a {x} today', 'we rescued a {x}'],
                ["Congratulations on your new {x}! That's so exciting. What's their name?", "Oh, how fun! A {x} is such great company. What did you name them?", "Aw, I love that! How's your new {x} settling in?"]),
}
# ---- News about the person. (message, how it is said back)
SELF_BAD = [
    ('i lost my job', 'losing your job'), ('i got fired', 'getting fired'), ('i got laid off', 'getting laid off'), ('i lost my job today', 'losing your job'), ('i got fired today', 'getting fired'),
    ('i got fired and i have kids to feed', 'getting fired'), ("i got fired and i don't know how to tell my family", 'getting fired'), ('my company is shutting down', 'your company shutting down'),
    ('i failed my exam', 'failing your exam'), ('i failed my class', 'failing your class'), ('i got kicked out of school', 'getting kicked out of school'), ('i have to repeat a year at school', 'having to repeat a year'),
    ('i got dumped', 'getting dumped'), ('my girlfriend left me', 'your breakup'), ('my boyfriend left me', 'your breakup'), ('my wife left me', 'your wife leaving'), ('my husband left me', 'your husband leaving'),
    ('my girlfriend broke up with me', 'your breakup'), ('my boyfriend broke up with me today', 'your breakup'), ('i found out my girlfriend has been lying to me', 'finding out she lied'), ('i found out my boyfriend cheated on me', 'finding out he cheated'),
    ('my husband cheated on me', 'finding out he cheated'), ('my wife cheated on me', 'finding out she cheated'), ('my parents are getting divorced', 'your parents divorcing'), ('i am getting divorced', 'your divorce'), ('my parents are fighting all the time', 'your parents fighting'),
    ('my car got stolen', 'your car being stolen'), ('my car broke down again', 'your car breaking down'), ('my house got broken into', 'your house being broken into'), ('i got robbed', 'getting robbed'), ('my wallet got stolen', 'your wallet being stolen'),
    ('i got into a car accident', 'your car accident'), ('i got diagnosed with diabetes', 'your diagnosis'), ('i found out i have cancer', 'your diagnosis'), ('i broke my leg', 'breaking your leg'), ('i need surgery', 'needing surgery'),
    ('i am getting bullied at school', 'being bullied'), ('people at school make fun of me', 'people making fun of you'), ('nobody came to my birthday party', 'nobody coming to your party'), ('nobody remembered my birthday', 'nobody remembering your birthday'), ('i have no friends at my new school', 'not having friends at your new school'),
    ('we had to move and i miss my old town', 'having to move'), ('we are losing our house', 'losing your house'), ('i got kicked out of my apartment', 'losing your apartment'), ('i have no money for rent', 'not having rent money'), ("i'm drowning in debt", 'all that debt'), ('i am broke', 'being broke'),
    ('my best friend stopped talking to me', 'your best friend going quiet'), ('my friends left me out', 'being left out'), ('i got rejected', 'getting rejected'), ("i didn't get the job", 'not getting the job'), ("i didn't get into college", 'not getting into college'),
    ('my boss keeps yelling at me', 'your boss yelling at you'), ('my boss is horrible to me', 'how your boss treats you'), ('i hate my job', 'how bad your job has been'), ('i got passed over for a promotion', 'being passed over'), ('i bombed my interview', 'how your interview went'), ('i lost the game for my team', 'how the game went'),
]
SELF_GOOD = [
    ('i got a new job', 'your new job'), ('i got a job', 'your new job'), ('i got the job!', 'the job'), ('i got promoted', 'your promotion'), ('i got a raise', 'your raise'), ('i got a raise today', 'your raise'), ('i started my own business', 'your new business'),
    ('i passed my exam', 'passing your exam'), ('i passed all my classes', 'passing all your classes'), ('i graduated', 'graduating'), ('i got into college', 'getting into college'), ('i got an a', 'that A'), ('i won first place', 'winning first place'), ('i won the game', 'winning the game'), ('i got a scholarship', 'your scholarship'),
    ('i got engaged', 'your engagement'), ('i just got engaged!', 'your engagement'), ('i got married', 'your wedding'), ('we are having a baby', 'the baby'), ("we're having a baby!!", 'the baby'), ('i am pregnant', 'the baby'), ('i got asked to prom', 'getting asked to prom'), ('i have a girlfriend now', 'your new girlfriend'), ('i have a boyfriend now', 'your new boyfriend'), ('i went on a great date', 'your great date'),
    ('i made a new friend', 'your new friend'), ('i made a new friend at school today', 'your new friend'), ('i finally paid off my debt', 'paying off your debt'), ('i paid off my loans', 'paying off your loans'), ('i bought my first house', 'your first house'), ('i bought a car', 'your new car'), ('we moved into our new home', 'your new home'),
    ('i ran my first race', 'your first race'), ('i lost twenty pounds', 'all your hard work'), ('i quit smoking', 'quitting smoking'), ('i have been sober for a year', 'a whole year sober'), ('my test results came back clear', 'your clear test results'), ('i beat cancer', 'beating cancer'), ('i won the lottery', 'winning the lottery'), ('i finished my book', 'finishing your book'), ('my cancer is gone', 'being cancer free'),
]
BAD_FRAMES = ["I'm really sorry about {s}. That's a lot to deal with. How are you holding up?", "Oh no, I'm so sorry. {S} sounds really hard. What's been the toughest part?",
              "I'm so sorry. {S} is really stressful. Do you want to talk about it?", "That's really tough. I'm sorry about {s}. How are you feeling right now?",
              "I'm sorry to hear that. {S} would be hard on anyone. What's on your mind most?"]
GOOD_FRAMES = ["That's amazing, congratulations on {s}! How do you feel?", "Wow, congratulations on {s}! How are you going to celebrate?", "Congrats on {s}! You must be so happy. What was the best part?",
               "That's so great to hear! I'm really happy for you about {s}. What's next?", "What great news! Congratulations on {s}."]

# ---- Second messages. Keeps what happened in mind.
SHORT_BAD = [
    (['thanks', 'thank you', 'thanks for listening', 'thanks kani', 'thank you for listening', 'ty', 'that helps'], ["You're welcome. I hope things get easier soon.", "Anytime. I'm here if you want to talk more about {t}.", "You're welcome. Please take care of yourself."]),
    (['idk', "i don't know", 'i dont know', 'not sure', 'i guess', 'idk anymore'], ["That's okay, you don't have to have it figured out. This is a lot. What's been the hardest part today?", "That's okay. There's no rush. We can take it slowly. How are you feeling about {t} right now?"]),
    (['yeah', 'yes', 'ok', 'okay', 'yeah i guess', 'kind of', 'a little'], ["I'm here if you want to say more about {t}. No rush.", "Okay. What's been on your mind most about {t}?"]),
    (['no', 'not really', 'nah', "i don't want to talk about it", 'i dont wanna talk about it'], ["That's okay, we don't have to talk about {t}. I'm here if that changes. Would a distraction help?", "No problem. We can talk about something else whenever you like."]),
    (['everything', 'everything i guess', 'all of it', 'its just everything', 'everything is just too much'], ["That sounds like so much at once. If you had to pick one thing that's weighing on you most, what would it be?", "I'm sorry, that sounds really heavy. Let's take it one piece at a time. What's on your mind first?"]),
    (["i'm okay", 'im fine', "i'm fine", 'i am ok', 'i think im ok'], ["I'm glad you're doing okay. This is still a lot, so go easy on yourself.", "Good. If it gets harder, I'm here to talk about {t}."]),
    (['it hurts so much', 'it really hurts', 'it just hurts', "i can't stop thinking about it", 'i feel awful', 'i feel so bad'], ["I'm so sorry. Of course this hurts. You don't have to get over it quickly. What's been coming up the most?", "That makes sense. This is really painful. Is there someone who could be with you today?"]),
]
SHORT_GOOD = [
    (['thanks', 'thank you', 'thanks kani', 'thank you so much', 'ty'], ["You're welcome! Enjoy this, it's great news.", "Anytime! I'm really happy for you.", "You're welcome! I hope you get to celebrate."]),
    (['yeah', 'yes', 'i know right', 'yeah im so happy', "i'm so happy"], ["I'm so glad! You should enjoy every bit of this. What are you most looking forward to?", "That's great! How are you going to celebrate?"]),
    (['idk', 'not sure yet', "i don't know yet", 'nothing yet'], ["That's okay, there's no rush. Take a moment to enjoy it!", "No problem. You've got time to figure that out. How are you feeling?"]),
    (["i'm kinda nervous tho", "i'm a little scared too", "i'm nervous though", 'i start on monday, im kinda nervous tho', "i'm worried i'll mess it up", "but i'm scared"],
     ["It makes sense to feel both. This is exciting and a big change at the same time. What's worrying you most?", "That's really normal. Even good news can feel scary. What part makes you the most nervous?"]),
]
# Details that come after a kind of news. {x} is who it is about; replies keep {x} or what the detail said.
DETAILS = {
    'pet_death': [('he was {n}. i had him since i was a kid', '{n} years is a long time to love a {x}. It makes sense that this hurts so much. What\'s your favorite memory of him?'),
                  ('she was {n} and i grew up with her', '{N} years together is so much of your life. No wonder losing your {x} hurts. What was she like?'),
                  ('he slept on my bed every night', "That's such a sweet thing to remember about your {x}. The house must feel so quiet without him. How have you been sleeping?"),
                  ('the house feels so empty', "I'm so sorry. Your {x} was part of every day, so of course the house feels empty. Do you want to tell me about your {x}?"),
                  ("i didn't get to say goodbye", "I'm so sorry. Not getting to say goodbye to your {x} is so painful. What would you have wanted to say?")],
    'person_death': [('we were really close', "It makes so much sense that this hurts. Your {x} was someone really important to you. What do you miss most?"),
                     ("i didn't get to say goodbye", "I'm so sorry. Not getting to say goodbye to your {x} is so painful. What would you have wanted to say?"),
                     ('the funeral is on friday', "That's so soon, and funerals can be really hard. Will you have people with you on Friday? I'm thinking of you and your {x}."),
                     ("i don't know how to feel", "That's completely okay. There's no right way to feel after losing your {x}. Whatever comes up is okay.")],
    'person_ill': [('the doctors say it is serious', "I'm so sorry. That's frightening to hear about your {x}. How are you holding up?"), ("i'm so scared of losing them", "Of course you are. You love your {x}, and that fear makes sense. Do you have someone to lean on right now?"),
                   ('i have to take care of everything now', "That's a lot on your shoulders while you're worried about your {x}. Is there anyone who can share some of it?")],
    'person_fight': [("{h} said i'm a disappointment", "That's such a painful thing to hear from your {x}, and I'm sorry. Words like that hurt, but they aren't the whole truth about you. How are you feeling now?"),
                     ('{h} called me selfish', "Ouch, that must have really stung, especially from your {x}. What was going on when that was said?"), ('i said some things i regret', "That happens in fights. It sounds like you care about your {x} a lot. Would you want to say sorry when things have cooled down?"),
                     ("{h} won't answer my texts", "That silence must be really hard. Maybe your {x} needs some time. How long has it been?"), ('i think it was my fault', "It takes courage to say that. Would you like help working out what to say to your {x}?")],
    'job': [('scared. i have rent due next week', "That makes total sense. Rent coming due right after this is a lot of pressure. Do you have any savings or anyone who could help for a little while?"),
            ("i don't know how to tell my wife", "That's a hard conversation, and it makes sense that you're nervous. Telling her soon, simply and honestly, is usually easier than holding it in. How do you think she'll take it?"),
            ('i worked there for ten years', "Ten years is a big part of your life, so of course this hits hard. I'm so sorry. What will you miss most?"), ('i feel so ashamed', "I'm sorry you feel that way. Losing a job happens to so many people, and it doesn't say anything bad about you. Have you been able to tell anyone?")],
    'breakup': [('i still love them', "Of course you do. Love doesn't stop just because a relationship ended. It's okay to miss them. How have you been coping?"), ("i can't stop crying", "I'm so sorry. It's okay to cry. Breakups really hurt. Is there someone who can be with you today?"),
                ('we were together for three years', "Three years is a long time, and so much of your life was shared. No wonder this hurts. What's been hardest so far?"), ('i keep checking their instagram', "That's so common after a breakup. It might help to mute them for a while, just for now. How are you sleeping?")],
    'work': [('he does it in front of everyone', "I'm sorry, nobody should be treated like that at work, least of all in front of others. How long has your boss been doing this?"), ('i dread going in every day', "That sounds exhausting, dreading work every single day. What makes it worst?")],
}
ADVICE_AFTER = {
    'person_death': ["There's no right way to do this. Let yourself grieve, lean on people who knew your {x}, and take it one day at a time. Is someone with you today?"],
    'pet_death': ["Be gentle with yourself, it's a real loss. Some people find it helps to keep a photo of their {x} nearby or to share stories with someone who knew them. Is there someone you can talk to today?"],
    'person_ill': ["Take it one step at a time. Ask the doctors your questions, let others help you, and try to rest when you can. Being there for your {x} is already a lot. Who can support you right now?"],
    'person_fight': ["Give it a little time to cool down, then tell your {x} how you felt, starting with \"I felt hurt\" instead of blaming. Would you like help working out what to say?"],
    'pet_missing': ["Check nearby first, ask your neighbors, and put up photos of your {x}. Call local shelters and vets too, and share a post in a local group. When did you last see your {x}?"],
    'job': ["Start small. Take today to breathe, then check what support you can get and update your resume when you're ready. You don't have to solve everything today. What's worrying you most?"],
    'breakup': ["For now, be gentle with yourself. Lean on friends, give yourself some distance from your ex, and keep up small routines like eating and sleeping. It does get easier with time. Who can you talk to today?"],
}
ASK_WHAT = ['what should i do?', 'what do i do?', 'what do i do now?', 'what should i do now', 'any advice?', 'what can i do?', 'now what?']
# How to feel better, and everyday people problems. Plain answers, not searches.
ADVICE = [
    (['how do i stop feeling sad?', 'how do i feel better when i am sad?', 'what can i do when i feel sad?', 'how do i cheer myself up?', 'how do i get out of a bad mood?'],
     ["Be gentle with yourself first. A walk, some sleep, a good meal and talking to someone you trust can all help a little. If sadness sticks around for weeks, a doctor or counselor can really help. What's been making you sad?"]),
    (['how do i deal with anxiety?', 'how do i stop being anxious?', 'how can i calm my anxiety?', 'what helps with anxiety?', 'how do i calm down?', 'how do i stop panicking?'],
     ["Try slow breathing first: in for 4 seconds, out for 6, for a couple of minutes. Naming five things you can see can also bring you back to the present. If anxiety keeps getting in the way, a doctor or counselor can really help. What's been making you anxious?"]),
    (["what should i do when i can't sleep because i'm worrying?", "how do i stop worrying at night?", 'i cant sleep because of my thoughts, what do i do?', 'how do i sleep when i am stressed?'],
     ["Try writing your worries down so they're out of your head, and put your phone away. Slow breathing helps too. If you're still awake after 20 minutes, get up and do something quiet until you feel sleepy. What's been on your mind?"]),
    (['how do i get over a breakup?', 'how do i get over my ex?', 'how do i stop missing my ex?', 'how long does it take to get over someone?', 'how do i move on from a relationship?'],
     ["Give it time, and let yourself feel sad. It helps to take a break from their social media, lean on friends, and get back to things you enjoy. It really does get easier. How long has it been?"]),
    (['how do i make friends?', 'how can i make new friends?', 'how do i make friends at a new school?', 'how do i make friends as an adult?', 'how can i find friends?'],
     ["Start with places where you see the same people often, like a club, class or team. Ask people about themselves, and suggest something small, like getting lunch. It takes a few tries, and that's normal. What do you like doing?"]),
    (['how do i apologize to my friend?', 'how do i say sorry?', 'how do i apologize to someone i hurt?', 'how should i apologize to my mom?', 'how do i say sorry to my boyfriend?'],
     ["Keep it simple and honest. Say what you did, say you're sorry for how it made them feel, and don't add excuses. Then give them time. What happened between you?"]),
    (['how do i tell my parents i failed?', 'how do i tell my parents bad news?', 'how do i tell my parents i got a bad grade?', 'how do i tell my wife i lost my job?', 'how do i tell someone bad news?'],
     ["Pick a calm moment and say it plainly, then tell them what you plan to do next. It's usually a relief once it's out, even if the talk is hard. How do you think they'll react?"]),
    (['how do i stop being angry?', 'how do i control my anger?', 'how do i calm down when i am mad?', 'how do i deal with anger?'],
     ["Step away for a few minutes before you respond, and take some slow breaths. Moving your body helps too. When you've cooled down, say what bothered you plainly. What's been making you angry?"]),
    (['how do i stop feeling lonely?', 'what can i do when i feel lonely?', 'how do i deal with loneliness?'],
     ["Start small: message one person, go somewhere with people around, or join something you enjoy. Loneliness eases slowly, so go easy on yourself. Is there anyone you've been meaning to reach out to?"]),
    (['how do i help a friend who is sad?', 'how do i comfort someone?', 'what do i say to someone who is grieving?', 'how do i comfort my friend?', 'how can i help my friend who lost someone?'],
     ["You don't need perfect words. Say you're sorry and that you care, listen more than you talk, and offer something specific, like bringing food. Who are you worried about?"]),
    (['how do i love myself?', 'how do i feel better about myself?', 'how do i get more confidence?', 'how can i be more confident?', 'how do i stop hating how i look?'],
     ["Start by talking to yourself the way you'd talk to a friend. Keep small promises to yourself, and spend less time with what makes you compare. It builds slowly. What's been hard for you lately?"]),
]

# ---- Crisis. The apps answer these before the model; these rows are the second line.
CRISIS_SELF = ['i want to die', 'i want to kill myself', 'i wanna die', 'i wanna kill myself', 'i am going to kill myself', "i'm going to kill myself tonight", 'im gonna kill myself', 'i want to end my life', 'i want to end it all',
               'i want to hurt myself', "i've been thinking about hurting myself", "i've been thinking about suicide", "i'm suicidal", 'i feel suicidal', 'i have been cutting myself', 'i cut myself again', "i don't want to live anymore", "i don't want to be alive",
               "i don't want to be here anymore", "i don't want to exist", 'i wish i was dead', 'i wish i could die', 'i wish i was never born', 'i wish i could sleep and never wake up', "i wish i wouldn't wake up", 'everyone would be better off without me',
               'my family would be better off without me', 'i should just kill myself', 'nobody would care if i died', 'no one would notice if i disappeared', 'i have nothing to live for', "i'm tired of being alive", 'i am sick of living', 'there is no reason to live',
               "i can't go on living like this", 'life is not worth living', "life isn't worth living anymore", 'i got dumped and i want to die', 'i failed and i want to die', 'my life is over and i want to die', 'i want to disappear forever', 'i want to stop existing',
               'i am thinking of ending my life', 'tonight i end it', 'i want to overdose', 'kms', 'i just want to die', 'i might kill myself', 'i could kill myself', 'i want to take my own life', 'help i want to kill myself', 'please help me i want to die']
CRISIS_OTHER = ['my friend told me she wants to kill herself', 'my friend told me he wants to kill himself', 'my friend wants to kill herself', 'my sister is suicidal', 'my brother is suicidal', 'my friend is suicidal', 'my mom said she wants to die', 'my dad said he wants to die',
                'my girlfriend said she wants to die', 'my boyfriend said he wants to die', 'my son wants to kill himself', 'my daughter is cutting herself', 'my friend has been cutting herself', 'my best friend tried to kill himself', 'my brother keeps hurting himself', 'someone i know is suicidal', 'my classmate said she wants to end her life']
CRISIS_LOSS = ['my brother killed himself', 'my dad killed himself', 'my mom killed herself', 'my sister killed herself', 'my best friend killed himself', 'my friend committed suicide', 'my uncle died by suicide', 'i lost my dad to suicide', 'i lost my brother to suicide', 'my son took his own life', 'my friend took her own life']
CRISIS_DANGER = ['i took too many pills', 'i took a bunch of pills', 'i swallowed a bunch of pills', 'i just overdosed', 'i think i overdosed', 'i took all of my meds at once', 'i took a whole bottle of tylenol', "i'm not safe", 'my friend overdosed', 'i ate a bunch of pills']
CRISIS_CHECK = ['whats the point of living', "what's the point", 'what is the point of life', "what's the point anymore", "i don't see the point anymore", "i just don't see the point anymore", "i can't do this anymore", "i can't take it anymore", "i can't take this anymore", "i can't go on",
                "i can't keep going anymore", "i can't cope anymore", 'i hate my life', 'i hate myself', 'my life is over', 'my life is pointless', 'life is pointless', 'life is meaningless', "i'm a burden", "i'm a burden to everyone", "i'm worthless", 'i should just disappear', 'i wish i could disappear', 'why should i keep going', 'why do i even live']
AFTER_CRISIS = ['nobody cares', 'no one cares', 'thanks', 'ok', 'i dont know', 'idk', 'i cant', "i can't call anyone", "i don't have anyone", 'it wont help', 'what do i do', 'i am scared', 'i just feel so alone', 'im tired', 'why']
OPENERS = ['', '', '', '', 'hi ', 'hey ', 'hey kani, ', 'kani ', 'um ', 'honestly ', 'idk ', 'well ', 'ugh ', 'so ', 'i think ', 'hi kani, ']
ENDINGS = ['', '', '', '', '.', '...', '!', ' :(', ' lol', ' today', ' right now', ' tbh']

# ---- SODA
SODA_FIRST = re.compile(r"\b(?:i'm|i am|i feel|i've been|i just|i got|i lost|i was|i had|i think i|my (?:mom|dad|dog|cat|friend|boss|mother|father|sister|brother|wife|husband|girlfriend|boyfriend|grandma|grandpa|son|daughter|parents|teacher|pet))\b", re.I)
SODA_BAD_WORDS = re.compile(r"\b(?:sad|upset|lonely|alone|scared|afraid|anxious|nervous|worried|stressed|overwhelmed|angry|mad|frustrated|hurt|heartbroken|depressed|exhausted|tired of|guilty|ashamed|embarrassed|jealous|disappointed|devastated|miserable|terrible|awful|miss (?:him|her|them)|died|passed away|broke up|dumped|fired|laid off|failed|rejected|cancer|divorce|sick|hospital|lost)\b", re.I)
SODA_GOOD_WORDS = re.compile(r"\b(?:happy|excited|thrilled|proud|grateful|relieved|engaged|pregnant|promoted|got the job|passed|won|so glad|amazing news|great news|wonderful news)\b", re.I)
SODA_SORRY = re.compile(r"\b(?:sorry|oh no|terrible|awful|that's too bad|that's rough|that's tough|sad to hear)\b", re.I)
SODA_HAPPY = re.compile(r"\b(?:congrat\w*|amazing|awesome|wonderful|fantastic|how exciting|so exciting|that's great|so happy for you|good for you)\b", re.I)
REPLY_OPENS = re.compile(r"^(?:oh no|oh,|oh wow|oh |i'm (?:so |really |very )?(?:sorry|glad|happy)|i am (?:so |really |very )?(?:sorry|glad|happy)|sorry|that's|that is|that sounds|that must|wow|congrat|i can imagine|i understand|it sounds|aw|ouch|yikes|what happened|how (?:are|do) you|tell me)", re.I)
# The other speaker's own life, or words only one particular person would say to another.
REPLY_BANS = re.compile(r"[()*]|\b(?:me too|my|mine|i was|i had|i've|i remember|when i|we|our|us|you know what|honey|sweetie|sweetheart|dear|son|buddy|man|dude|bro|kiddo|babe|mom|dad|god|doctor|sir|ma'am|your honor|mr|mrs|ms|miss)\b", re.I)
USER_BANS = re.compile(r"\b(?:you|your|you're|thank|thanks|sorry|please|mr|mrs|ms|sir|ma'am|honey|sweetie|god)\b|\?|^(?:mom|dad|mama|papa|mommy|daddy|bestie|doctor|dr|coach|boss|grandma|grandpa|sis|bro|babe|dear)\b|[()*]", re.I)
MID_NAME = re.compile(r"(?<=[a-z,] )[A-Z][a-z]+")
# Opens with the news itself: "Yeah, I guess I am..." answers something earlier that the model would not see.
SODA_STANDS_ALONE = re.compile(r"(?:i'm|i am|i feel|i just|i got|i lost|i've been|i can't believe|i miss|i have|i had|i was|my )\b", re.I)
SODA_HOSTILE = re.compile(r"no excuse|it better not|you should have|you'd better|how dare|shut up|i'm so jealous|you're lucky|grow up|your fault|stupid|idiot", re.I)
GREETING = re.compile(r'^(?:hey|hi|hello)[,!. ]+', re.I)


def soda_rows(rng):
    """Someone tells news or a feeling, the other responds to it; up to three such exchanges, for when a chat goes on."""
    table = pq.read_table(SOURCES / 'soda-train.parquet', columns=['dialogue', 'speakers', 'PersonX', 'PersonY', 'PersonZ'])
    by_mood = {'bad': [], 'good': []}
    for r in table.to_pylist():
        turns = r['dialogue']
        if len(turns) < 2:
            continue
        names = {n.lower() for n in [*r['speakers'], r['PersonX'], r['PersonY'], r['PersonZ']] if n}
        # The news may come after small talk; the chat then starts there, as long as it makes sense on its own.
        for start in (0, 2):
            if start + 1 >= len(turns):
                start = None
                break
            first = GREETING.sub('', turns[start]).strip()
            bad, good = bool(SODA_BAD_WORDS.search(first)), bool(SODA_GOOD_WORDS.search(first))
            if first and 15 <= len(first) <= 200 and not USER_BANS.search(first) and not MID_NAME.search(first) and SODA_FIRST.search(first) and bad != good and SODA_STANDS_ALONE.match(first):
                break
        else:
            start = None
        if start is None:
            continue
        turns = turns[start:]
        mood = 'bad' if bad else 'good'
        first = first[0].upper() + first[1:]
        messages = []
        for i in range(0, min(len(turns) - 1, 6), 2):
            asked, said = (first if i == 0 else turns[i].strip()), turns[i + 1].strip()
            words = set(re.findall(r'[a-z]+', (asked + ' ' + said).lower()))
            if not 15 <= len(said) <= 260 or names & words or REPLY_BANS.search(said) or SODA_HOSTILE.search(said) or MID_NAME.search(said) or MID_NAME.search(asked) or len(asked) > 200 or support_reply(normalized(asked)):
                break
            if i == 0 and (not REPLY_OPENS.match(said) or SODA_HAPPY.search(said) and mood == 'bad' or SODA_SORRY.search(said) and mood == 'good'):
                break
            if i and re.search(r'\b(?:you|your)\b', asked, re.I):
                break
            messages += [{'role': 'user', 'content': asked}, {'role': 'assistant', 'content': said}]
        if messages:
            by_mood[mood].append(messages)
    rows = []
    for mood, count in (('bad', SODA_BAD), ('good', SODA_GOOD)):
        rng.shuffle(by_mood[mood])
        rows += [{'family': f'soda:{mood}', 'source': 'allenai/soda', 'messages': m} for m in by_mood[mood][:count]]
    print(f"SODA: {len(by_mood['bad']):,} bad and {len(by_mood['good']):,} good chats pass the filter", flush=True)
    return rows


def noisy(text, rng):
    """As people type it: no apostrophes, a capital, an opener, an ending."""
    text = rng.choice(OPENERS) + text + rng.choice(ENDINGS)
    if rng.random() < 0.3:
        text = text.replace("'", '')
    if rng.random() < 0.25:
        text = text[0].upper() + text[1:]
    return text


def article(thing):
    return thing[0].upper() + thing[1:]


class Rows:
    def __init__(self, rng):
        self.rng, self.rows, self.copies = rng, [], collections.Counter()

    def add(self, family, messages, replies, capped=True):
        """Adds a row ending in one of `replies` still under CAP; returns the reply, or None."""
        if any(m['role'] == 'user' and normal(m['content']) in HELD for m in messages):
            return None
        open_replies = [r for r in replies if not capped or self.copies[r] < CAP]
        if not open_replies:
            return None
        reply = self.rng.choice(open_replies)
        self.copies[reply] += 1
        self.rows.append({'family': family, 'source': 'make_instruct_v20', 'messages': [*messages, {'role': 'assistant', 'content': reply}]})
        return reply


def user(text):
    return {'role': 'user', 'content': text}


def assistant(text):
    return {'role': 'assistant', 'content': text}


def about_kani(rows, rng):
    for asks, replies in ABOUT_KANI:
        for ask in asks:
            for _ in range(5):
                rows.add('about_kani', [user(noisy(ask, rng))], replies)
    for word in FEELING_WORDS:
        for form in FEELING_FORMS:
            rows.add('about_kani', [user(noisy(form.format(word), rng))], [r.replace('{}', word) for r in FEELING_REPLIES])
    # Same questions after a chat has started, so they are not taken for the earlier subject.
    for asks, replies in ABOUT_KANI:
        for ask in asks:
            hello = rng.choice(["Hi! How can I help today?", "Hello! What's on your mind?", "Hey! What can I do for you?"])
            rows.add('about_kani', [user(rng.choice(['hi', 'hello', 'hey', 'hi kani'])), assistant(hello), user(ask)], replies)


def moods(rows, rng):
    made = []
    for wordings, mood, replies in MOODS:
        for wording in wordings:
            for _ in range(4):
                message = noisy(wording, rng)
                reply = rows.add(f'mood:{mood}', [user(message)], replies)
                if reply:
                    made.append((message, reply, mood))
    return made


def news(rows, rng):
    """(message, reply, mood, kind, who, topic), for second messages."""
    made = []
    for kind, (mood, whos, wordings, replies) in NEWS.items():
        for who in whos:
            for wording in rng.sample(wordings, min(7, len(wordings))):
                message = noisy(wording.format(x=who), rng)
                reply = rows.add(f'news:{mood}', [user(message)], [r.format(x=who) for r in replies])
                if reply:
                    made.append((message, reply, mood, kind, who, f'your {who}'))
    for pool, frames, mood in ((SELF_BAD, BAD_FRAMES, 'bad'), (SELF_GOOD, GOOD_FRAMES, 'good')):
        for text, said in pool:
            for _ in range(5):
                message = noisy(text, rng)
                kind = 'job' if re.search(r'lost my job|fired|laid off|shutting down', text) and mood == 'bad' else 'breakup' if re.search(r'\b(?:left me|dumped|broke up)\b', text) else 'work' if 'boss' in text else 'self'
                reply = rows.add(f'news:{mood}', [user(message)], [f.format(s=said, S=article(said)) for f in frames])
                if reply:
                    made.append((message, reply, mood, kind, None, said))
    return made


def second_messages(rows, rng, made, feelings):
    for message, reply, mood, kind, who, topic in made:
        first = [user(message), assistant(reply)]
        if rng.random() < 0.55:
            asks, replies = rng.choice(SHORT_BAD if mood == 'bad' else SHORT_GOOD)
            rows.add(f'second:{mood}', [*first, user(rng.choice(asks))], [r.format(t=topic, T=article(topic)) for r in replies])
        if kind in DETAILS and rng.random() < 0.7:
            detail, follow = rng.choice(DETAILS[kind])
            years = rng.randint(6, 17)
            he = rng.choice(['he', 'she'])
            x = who or ''
            rows.add(f'detail:{kind}', [*first, user(noisy(detail.format(n=years, h=he), rng))], [follow.format(x=x, n=years, N=years, h=he)])
        if kind in ADVICE_AFTER and rng.random() < 0.3:
            rows.add(f'advice_after:{kind}', [*first, user(rng.choice(ASK_WHAT))], [r.format(x=who or '') for r in ADVICE_AFTER[kind]])
    # A mood first, then what happened: the reply follows what happened.
    bad_news = [m for m in made if m[2] == 'bad']
    good_news = [m for m in made if m[2] == 'good']
    for opening, reply, mood in feelings:
        if mood == 'good':
            pick = rng.choice(good_news)
        elif mood == 'bad':
            pick = rng.choice(bad_news)
        else:
            continue
        if rng.random() < 0.5:
            detail = pick[0]
            rows.add(f'chain:{mood}', [user(opening), assistant(reply), user(detail)], [pick[1]])
        if mood == 'bad' and rng.random() < 0.25:
            asks, replies = rng.choice([s for s in SHORT_BAD if 'everything' in s[0][0] or 'idk' in s[0][0]])
            rows.add('chain:short', [user(opening), assistant(reply), user(rng.choice(asks))], [r.format(t='all this', T='All this') for r in replies])


def advice(rows, rng):
    for asks, replies in ADVICE:
        for ask in asks:
            for _ in range(4):
                rows.add('advice', [user(noisy(ask, rng))], replies)


def crisis(rows, rng):
    for wordings, reply, repeats in ((CRISIS_SELF, CRISIS_REPLY, 12), (CRISIS_OTHER, OTHER_PERSON_REPLY, 6), (CRISIS_LOSS, LOSS_REPLY, 6), (CRISIS_DANGER, DANGER_REPLY, 8), (CRISIS_CHECK, CHECK_IN_REPLY, 6)):
        for wording in wordings:
            for _ in range(repeats):
                rows.add('crisis', [user(noisy(wording, rng))], [reply], capped=False)
    # Later in a chat, too, and the next message after the fixed reply.
    for wording in CRISIS_SELF:
        rows.add('crisis', [user(rng.choice(['hi', "i'm so sad", 'today was awful', 'i got dumped'])), assistant("I'm sorry. Do you want to tell me what's going on?"), user(noisy(wording, rng))], [CRISIS_REPLY], capped=False)
        for _ in range(2):
            rows.add('crisis', [user(wording), assistant(CRISIS_REPLY), user(rng.choice(AFTER_CRISIS))], [STILL_HERE_REPLY], capped=False)


def build(seed=2436):
    rng = random.Random(seed)
    rows = Rows(rng)
    about_kani(rows, rng)
    feelings = moods(rows, rng)
    made = news(rows, rng)
    second_messages(rows, rng, made, feelings)
    advice(rows, rng)
    crisis(rows, rng)
    new = [r for r in rows.rows + soda_rows(rng) if not any(m['role'] == 'user' and normal(m['content']) in HELD for m in r['messages'])]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    train = [{**r, 'split': 'train'} for r in new[cut:]]

    # Replay: every skill Otter 4.11 has, minus the situations whose advice gets copied onto the wrong message.
    old = [json.loads(line) for line in (ROOT / 'data/instruct-v19-otter-4-11.jsonl').read_text(encoding='utf-8').splitlines()]
    situations = {m['content'] for r in old if (r.get('family') or '').startswith('situation') for m in r['messages'] if m['role'] == 'assistant'}
    def renamed(r):
        return {**r, 'messages': [{**m, 'content': m['content'].replace(OLD_NAME, NAME)} for m in r['messages']]}
    keep = [renamed(r) for r in old if not any(m['role'] == 'assistant' and m['content'] in situations for m in r['messages'])
            and not older_held_out(r) and not any(m['role'] == 'user' and normal(m['content']) in HELD for m in r['messages'])]
    identity = [json.loads(line) for line in (ROOT / 'data/instruct-v13-otter-4-11.jsonl').read_text(encoding='utf-8').splitlines()]
    train += [renamed(r) for r in identity if r.get('family') == 'identity_v13' and r['split'] == 'train' and not older_held_out(r)]
    train += [r for r in keep if r['split'] == 'train']
    validation += [r for r in keep if r['split'] != 'train']
    rng.shuffle(train)
    return train + validation


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v20.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train)')
    print(dict(families.most_common()))
    print('most copied replies:', [(n, r[:50]) for r, n in replies.most_common(5)])
