"""New, disclosed repair curriculum; retain original coding practice.

Final-reply training includes wrong prior answers as context, never as targets.
Held-out wording never appears in training histories.
"""
import json
from pathlib import Path
import random
from make_conversations import build

def build_repair():
    rng = random.Random(812)
    base = build()
    groups = [
        (['can you code?', 'can you code', 'do you code', 'can u code', 'you can code?', 'are you able to code', 'can you program', 'can you help me code?', 'do you write programs', 'can you write a program', 'can u write code', 'help me with coding', 'can you make code', 'write some code', 'i need code', 'code please'],
         ['are you able to write code for me?', 'could you help with a little programming?'],
         'Yes. I can write small Python and JavaScript examples. Tell me what you want to build and which language to use.'),
        (['no', 'no!', 'nooo', 'noooo', 'nope', 'nah', 'wrong', 'that is wrong', 'you are wrong', "that's wrong", 'not what i asked', 'you misunderstood', 'stop repeating yourself', 'stop saying hello', 'answer my question', 'that was unrelated', 'you did not answer', "that's not it", 'bro what', 'gng what', 'bruh', 'try again', 'i meant something else'],
         ['no, you misunderstood my question', 'that answer has nothing to do with my request'],
         'I misunderstood your request. Please restate what you need, and I will try to answer that specific question.'),
        (['what date is today', 'what is the date today', "what's the date", 'what day is it', "today's date", 'what time is it', 'tell me the time', 'current date', 'current time'],
         ['tell me what the date is today please', 'could you tell me the current time?'],
         'I cannot know the current date or time from my training. Use the local clock helper for a live answer.'),
        (['what do you mean by that', "i don't get it", 'i do not understand that', 'explain your answer', 'wait what do you mean', 'what was that supposed to mean', 'what are you saying', 'huh explain', 'say that more clearly'],
         ['could you explain your last reply more clearly?', 'I am confused by that response'],
         'I may have misunderstood. What would you like me to explain?'),
        (['what are you called', 'tell me about yourself', 'tell me who you are', 'what ai are you', 'are you b1', 'who r u', 'your name?', 'what is kelus', 'what is b1', 'introduce yourself please'],
         ['which assistant am I speaking with?', 'please introduce your model'],
         "I'm Kelus Model B1, a small model trained locally from scratch for basic chat and code."),
        (['thanks for the help', 'thank you for helping', 'thank u', 'thx', 'that helps', 'appreciate it', 'thank you very much', 'great thank you', 'thanks bro', 'tysm'],
         ['many thanks for your help', 'thanks, that clears it up'],
         "You're welcome! What would you like to try next?"),
        (['what is the capital of France', 'who won the election', 'give me stock prices', 'how old is the universe', 'what happened yesterday', 'write a complete web server with login and a database', 'build a complete Minecraft client', 'write a full operating system', 'make a complete social network', 'explain general relativity', 'give me medical advice', 'who is the president', 'predict the future', 'translate everything into Korean', 'tell me the latest sports results'],
         ['build a full multiplayer server with authentication', 'give me live weather in Seoul'],
         'I do not have enough training to answer that reliably. I can help with a short coding example or a basic programming concept.'),
    ]
    new_train, new_val = [], []
    for prompts, heldout, answer in groups:
        for p in prompts:
            for variant in (p, p.capitalize(), p.upper(), p.rstrip('?!')+'?', p.rstrip('?!')+' please'):
                new_train.append((variant, answer))
        new_val.extend((p, answer) for p in heldout)
    base_train = [(r['messages'][0]['content'],r['messages'][1]['content']) for r in base if r['split']=='train' and len(r['messages'])==2]
    base_val = [(r['messages'][0]['content'],r['messages'][1]['content']) for r in base if r['split']=='validation' and len(r['messages'])==2]
    forbidden = {p for p,_ in new_val + base_val}
    train = list(dict.fromkeys((p,a) for p,a in base_train + new_train if p not in forbidden))
    val = list(dict.fromkeys(base_val+new_val))
    def record(pairs, split):
        return {'split':split, 'messages':[m for p,a in pairs for m in ({'role':'user','content':p},{'role':'assistant','content':a})]}
    records = [record([pair], 'train') for pair in train] + [record([pair], 'validation') for pair in val]
    # Balance basic chat against the much larger code-template family.
    fresh = list(dict.fromkeys(new_train))
    for i in range(3500):
        target = rng.choice(fresh if i%2 else train)
        history = rng.sample(train, rng.choice((1,1,2,3)))
        if i%3==0:
            p,_ = history[-1]
            history[-1] = (p, rng.choice(['def is when a function returns a result.', "You're welcome!", 'A dictionary maps keys to values.', 'jefem in x', "Hi! I'm Kelus Model B1. What would you like to build?"]))
        records.append(record(history+[target], 'train'))
    for i in range(100):
        records.append(record(rng.sample(val, 3), 'validation'))
    # Preserve context-dependent concept -> example conversations.
    records += [r for r in base if len(r['messages'])==4 and r['messages'][2]['content'] in ('show me an example','give me an example','example please','can you show an example','how does that look in code','could you give an example?')]
    unique = {json.dumps(r['messages'],sort_keys=True):r for r in records}
    return list(unique.values())

if __name__=='__main__':
    records=build_repair()
    path=Path(__file__).with_name('repair.jsonl')
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    print({s:sum(r['split']==s for r in records) for s in ('train','validation')})
