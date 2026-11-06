"""instruct-v14: Kani keeps to the subject (Kani Otter 4.5).

Otter 4.4 (instruct-v13) went off track in everyday chat (2026-10-29; kelus/ontopic_eval.py came from it):
  - one reply for every sad message: exam stress, a friend's anger and cancelled plans all got "I'm sorry you're
    feeling down", since only 24 feelings rows survived v13's cap;
  - a short follow-up lost the subject: "I want to learn guitar", then "how long will it take?" gave "SEARCH: Guitar".
    v13 had one follow-up lookup in 92k rows;
  - HuggingFaceTB/smol-smoltalk's 2,252 everyday conversations (Apache-2.0) were nearly all dropped: they open with the
    same "Hello! How can I help you today?", which was over the cap at once.
Added here, ahead of a REPLAY sample of instruct-v13's training rows (renamed NAME, its feelings rows dropped):
  - SITUATIONS: what happened to the user (exams, a fight with a friend, a breakup, good news...), each answered with
    replies built from parts that name it, and sometimes a thank-you turn after;
  - follow-ups: a subject (an activity, a trip, a pet, a purchase), a reply asking about it, then a short question that
    is looked up with the subject in the query; or a new, standalone question that leaves the subject behind;
  - "explain how X works" questions, looked up;
  - the everyday conversations, without their greeting turn.
Every exact reply still appears at most CAP times in training rows. The prompts of kelus/ontopic_eval.py,
identity_eval and offscript_eval are left out, and none of ontopic_eval's subjects is used here.

    .venv/bin/python data/make_instruct_v14.py   # writes data/instruct-v14.jsonl
"""
import collections
import itertools
import json
from pathlib import Path
import random
import re
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus import identity_eval, offscript_eval, ontopic_eval  # noqa: E402
from make_instruct_v12 import SMOLTALK, conversation  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.5', 'Kani Otter 4.4'
CAP = 3
REPLAY = 20_000
VALIDATION_SHARE = 0.05


SITUATIONS = [
    ('exams', 'bad',
     ["I have a big exam on Friday and I'm freaking out", "exams start next week and I'm so stressed",
      "I'm really anxious about my math test", "finals are coming and I can't focus",
      "I have three tests this week and I'm panicking", "I'm scared I'll fail my chemistry exam"],
     ["Exam stress is really common, and it shows you care about doing well.",
      "That's a lot of pressure, especially with the exam so close.",
      "Feeling nervous before a test is normal, even when you've prepared."],
     ["Try splitting what's left into small chunks and doing one at a time, with short breaks.",
      "Practice questions under a timer help more than rereading your notes.",
      "Sleep matters more than one more late night of studying: tired brains forget faster.",
      "Write down the topics you're least sure about and start with those."],
     ["Which subject is worrying you most?", "Do you want help making a quick study plan?",
      "How much time do you have before it starts?"],
     ["Glad that helped. Good luck on your exam, you've got this!", "Anytime! Get some sleep before the test, too.",
      "You're welcome. Come back if you want to quiz yourself on anything before the exam."]),
    ('interview', 'bad',
     ["I have a job interview tomorrow and I'm nervous", "my interview is in an hour and my hands are shaking",
      "I'm so anxious about my first real job interview", "what if I mess up my interview?",
      "I have a second-round interview and I'm scared"],
     ["Interview nerves are normal, and they usually fade once the conversation starts.",
      "Getting an interview already means they liked what they saw.",
      "It makes sense to feel nervous before an interview that matters to you."],
     ["Prepare two or three short stories about times you solved a problem; they fit most questions.",
      "Read the job post again and pick out the skills they care about most.",
      "Have one or two questions ready to ask them at the end.",
      "A minute of slow breathing before you go in really does steady your voice."],
     ["What kind of job is it?", "Want to practice a few common interview questions together?",
      "Which question are you most worried about?"],
     ["Good luck in the interview! I hope it goes really well.", "You're welcome. Let me know how the interview goes!",
      "Anytime. You'll do great, just be yourself in there."]),
    ('friend upset', 'bad',
     ["my best friend is ignoring me and I don't know what I did", "I think my friend is upset with me",
      "my friend got really angry at me today", "me and my friend had a big fight",
      "my friend said I'm selfish and now we're not talking"],
     ["That hurts, especially when it's someone you're close to.",
      "Not knowing why a friend is upset can be really stressful.",
      "Fights with good friends feel awful, but a lot of friendships come back stronger."],
     ["A short, calm message asking if you did something to upset them can open the door.",
      "Try to listen to their side first, even if you see it differently.",
      "Give it a little time to cool down, then talk in person if you can.",
      "If you did hurt them, a simple, honest apology goes a long way."],
     ["Did anything happen right before they got upset?", "Do you want help writing a message to them?",
      "How long have you two been friends?"],
     ["I hope you two can talk it out soon.", "You're welcome. Good luck with your friend, I hope it gets better.",
      "Anytime. Friendships can handle a rough patch, especially when someone reaches out."]),
    ('cancelled on a friend', 'bad',
     ["I cancelled plans with my friend last minute and now she won't reply", "I had to bail on my friends tonight and I feel bad",
      "I keep cancelling on people and I feel guilty", "I forgot I had plans with my friend and stood him up"],
     ["Cancelling last minute can sting for the other person, even when you had a good reason.",
      "Feeling guilty shows you care about the friendship.",
      "Missing plans happens, and it's fixable."],
     ["Tell them why you cancelled and say sorry without piling on excuses.",
      "Suggest a new plan with a real date, so they know you still want to see them.",
      "Give them a little space if they need it; one sincere message is enough for now."],
     ["Did you get a chance to explain why you cancelled?", "Want help writing an apology text?"],
     ["Good luck, I hope they reply soon.", "You're welcome. A real apology and a new plan usually patch things up."]),
    ('friend cancelled', 'bad',
     ["my friend cancelled on me again", "my friends bailed on our plans and I'm disappointed",
      "I was really looking forward to tonight but they cancelled", "my friend never shows up when we make plans"],
     ["That's disappointing, especially when you were looking forward to it.",
      "Being cancelled on again can make you feel like you matter less, and that's hard.",
      "It's frustrating when plans keep falling through."],
     ["It's okay to tell them it bothered you, calmly and honestly.",
      "You could still do something nice for yourself tonight, like a movie or a walk.",
      "If it keeps happening, it's fair to ask whether something is going on with them."],
     ["Is this something that happens a lot with them?", "What were you planning to do together?"],
     ["You're welcome. I hope the rest of your evening turns out nice anyway.",
      "Anytime. You deserve friends who show up for you."]),
    ('breakup', 'bad',
     ["my girlfriend broke up with me", "I just went through a breakup", "my boyfriend and I split up yesterday",
      "we broke up after two years", "I can't stop thinking about my ex"],
     ["I'm sorry. Breakups hurt, even when they're for the best.",
      "Splitting up is a real loss, and it's okay to grieve it.",
      "It's normal to keep thinking about your ex for a while after a breakup."],
     ["Lean on friends and family this week; you don't have to go through it alone.",
      "Muting their social media for a while can make the days easier.",
      "Keep some routine: meals, sleep, a walk outside. It helps more than it sounds.",
      "Writing down what you're feeling can stop the same thoughts from looping."],
     ["How are you holding up today?", "Do you want to talk about what happened?",
      "Do you have someone nearby you can spend time with?"],
     ["Take it one day at a time. I'm here if you want to talk about it again.",
      "You're welcome. Breakups get easier, slowly but surely."]),
    ('lonely', 'bad',
     ["I feel lonely lately", "I don't really have any friends at my new school", "everyone has plans except me",
      "I moved to a new city and I don't know anyone"],
     ["Feeling lonely is hard, and a lot more people feel it than talk about it.",
      "Starting over somewhere new can feel really isolating at first.",
      "Watching everyone else make plans can make loneliness sting even more."],
     ["Clubs, classes or volunteering around something you enjoy are easier ways to meet people than walking up to strangers.",
      "Sending one message to someone you used to talk to can restart a friendship.",
      "Saying yes to small invitations, even short ones, adds up over time."],
     ["What do you like doing in your free time?", "Is there anyone you've been meaning to reach out to?"],
     ["Anytime. I hope you find your people soon, they're out there.",
      "You're welcome. Small steps really do add up when making friends."]),
    ('sleep', 'bad',
     ["I can't sleep", "I've been so tired all week", "it's 3am and I'm still awake", "I keep waking up in the middle of the night"],
     ["Not sleeping well makes everything feel harder.",
      "Being awake in the middle of the night with your thoughts can be really frustrating.",
      "A week of bad sleep really adds up."],
     ["Putting your phone away for the last half hour before bed helps your brain wind down.",
      "Going to bed and waking up at the same time every day, even weekends, resets your sleep.",
      "If you can't fall asleep after 20 minutes, get up, read something calm in dim light, then try again.",
      "Caffeine after lunch can still keep you awake at night."],
     ["Is something on your mind keeping you up?", "How many hours have you been getting?"],
     ["Sleep well! I hope tonight is better.", "You're welcome. Give the new routine a few nights to work."]),
    ('bored', 'neutral',
     ["I'm so bored right now", "there's nothing to do today", "I'm bored at home all weekend"],
     ["Boredom can be a sign you're ready for something new.", "A slow day can be a chance to try something different."],
     ["You could pick a short project, like drawing, a puzzle or cooking something new.",
      "A walk somewhere you've never been can be surprisingly fun.",
      "Learning a few chords, a card trick or a new recipe takes less than an hour."],
     ["Are you in the mood for something active or something cozy?", "Are you stuck inside, or can you get out?"],
     ["Have fun! Tell me what you end up doing.", "Enjoy! A bored afternoon can turn into a good one."]),
    ('pet died', 'bad',
     ["my dog died today", "we had to put our cat down", "my hamster passed away", "I lost my dog last week and I miss him so much"],
     ["I'm so sorry. Losing a pet is losing a member of the family.",
      "That's heartbreaking. Pets are with us every day, so home can feel empty without them."],
     ["It's okay to cry and to take your time; grief for a pet is real grief.",
      "Some people find it helps to make a small memory box with a photo, the collar or a favorite toy.",
      "Talking about your favorite memories of them with someone who knew them can help."],
     ["What was their name?", "How long were they with you?"],
     ["They were lucky to have someone who loved them so much.", "You're welcome. Be gentle with yourself while you miss them."]),
    ('bad grade', 'bad',
     ["I failed my test", "I got a really bad grade on my essay", "I got a D in math", "I studied so hard and still failed"],
     ["That's really discouraging, especially after you worked for it.",
      "One bad grade feels huge right now, but it's one result, not who you are."],
     ["Ask your teacher what went wrong; most are glad to help students who ask.",
      "Look at which questions you lost points on: that tells you exactly what to practice.",
      "Check whether there's a retake, extra credit or a chance to raise the final grade."],
     ["Which part of the test was hardest?", "Do you know when the next test is?"],
     ["You've got this next time. One grade doesn't decide everything.", "You're welcome. Good luck with the next one!"]),
    ('cut from team', 'bad',
     ["I didn't make the soccer team", "I got cut from the basketball team", "I didn't get picked for the school play"],
     ["That's really disappointing, especially after trying out.", "Not getting picked hurts, and it's okay to be upset about it."],
     ["Ask the coach or director what to work on; it shows you're serious about next time.",
      "Lots of people who got cut once made it the following year.",
      "You could keep going in a club or community group to stay sharp."],
     ["Are you thinking of trying out again?", "What were you hoping to do on the team?"],
     ["You're welcome. Keep at it, next tryouts could go differently.", "Anytime. I hope you try again next season."]),
    ('first day', 'bad',
     ["tomorrow is my first day at a new school", "I start my new job on Monday and I'm nervous",
      "first day of college tomorrow and I'm scared"],
     ["First days are nerve-racking for almost everyone.", "Starting somewhere new is a big step, and it's normal to feel jittery."],
     ["Plan your route and what you'll wear tonight, so the morning is calm.",
      "Asking someone a simple question, like where something is, is an easy way to start talking.",
      "Nobody expects you to know everything on day one."],
     ["What are you most nervous about?", "What are you looking forward to?"],
     ["Good luck tomorrow! I hope your first day goes great.", "You're welcome. Tell me how the first day goes!"]),
    ('family argument', 'bad',
     ["my parents are always fighting", "I had a huge argument with my mom", "my dad yelled at me and I'm upset",
      "my parents don't understand me"],
     ["Arguments at home are really draining, because there's nowhere to get away from them.",
      "It hurts when the people closest to you don't seem to understand you."],
     ["When things are calm, try telling them how you feel, starting with \"I feel\" instead of blaming.",
      "Taking a short break in another room before answering can stop a fight from growing.",
      "Talking to another adult you trust, like a relative or a school counselor, can help."],
     ["What was the argument about?", "Has this been happening a lot lately?"],
     ["You're welcome. I hope things at home calm down soon.", "Anytime. I hope you and your family can talk it through."]),
    ('overwhelmed', 'bad',
     ["I have so much homework I don't know where to start", "I'm overwhelmed at work", "I have too many deadlines this week"],
     ["When everything is due at once, it's hard to even start.",
      "Feeling overwhelmed usually means there's too much on your plate, not that you're doing badly."],
     ["Write everything down, then pick the one task that's due first and do only that.",
      "Start with a ten-minute task to build some momentum.",
      "If a deadline is impossible, ask early for more time; people say yes more often than you'd think."],
     ["What's due first?", "Want help putting your tasks in order?"],
     ["You're welcome. One task at a time, you'll get through the list.", "Good luck! Tackle the first one and the rest gets easier."]),
    ('lost job', 'bad',
     ["I got fired today", "I lost my job", "I got laid off this morning"],
     ["I'm sorry. Losing a job is a shock, even when you saw it coming.",
      "That's a lot to take in, both for money and for how you feel."],
     ["Give yourself a day before making big decisions.",
      "Check whether you can get unemployment benefits or severance where you live.",
      "Update your resume while your recent work is fresh in your mind."],
     ["How are you feeling about it?", "Do you know what you'd like to do next?"],
     ["You're welcome. I hope the next job turns out even better.", "Anytime. Good luck with the job search."]),
    ('sick', 'bad',
     ["I have a bad cold", "I'm sick and I feel terrible", "I have a fever and a sore throat"],
     ["Being sick is miserable, I'm sorry.", "A cold can really knock you out for a few days."],
     ["Rest, water and warm drinks like tea with honey help a sore throat.",
      "If your fever is high or lasts more than a few days, check with a doctor.",
      "Skip work or school if you can; rest helps you get better faster."],
     ["How long have you been feeling sick?", "Do you have someone who can bring you food?"],
     ["Feel better soon!", "You're welcome. Get lots of rest and I hope you're better in a few days."]),
    ('presentation', 'bad',
     ["I have to give a presentation tomorrow and I'm terrified", "public speaking makes me so nervous",
      "I have to speak in front of the whole class"],
     ["Speaking in front of a group is one of the most common fears, so you're not alone.",
      "Your heart racing before a presentation is your body getting ready, not a sign it'll go badly."],
     ["Practice out loud two or three times; it feels much easier after that.",
      "Know your first sentence by heart; the rest flows more easily once you've started.",
      "Look at a few friendly faces instead of the whole room."],
     ["What's the presentation about?", "How long do you have to speak?"],
     ["Good luck with your presentation! You'll do better than you think.", "You're welcome. Break a leg tomorrow!"]),
    ('procrastinating', 'neutral',
     ["I keep procrastinating", "I can't make myself start my essay", "I've been putting off cleaning my room for weeks"],
     ["Procrastinating happens to everyone, especially with tasks that feel big.",
      "Putting something off usually means it feels bigger than it is."],
     ["Set a timer for ten minutes and only promise yourself those ten minutes.",
      "Break it into the smallest first step, like writing one sentence or clearing one shelf.",
      "Put your phone in another room while you work."],
     ["What's the first small step you could do right now?", "When does it need to be done?"],
     ["Go for it! Ten minutes, starting now.", "You're welcome. Once you start, it usually gets easier."]),
    ('homesick', 'bad',
     ["I'm homesick", "I miss my family so much at college", "I moved abroad and I miss home"],
     ["Missing home is normal, especially in the first few months.",
      "Homesickness can hit hardest when everything around you is new."],
     ["A regular call home, like every Sunday, gives you something to look forward to.",
      "Bring a few things from home into your space, like photos or a favorite blanket.",
      "Getting to know a few people nearby makes a new place feel more like home."],
     ["How long have you been away?", "What do you miss most?"],
     ["You're welcome. It gets easier as the new place starts to feel familiar.", "Anytime. Call home soon, they probably miss you too."]),
    ('stuck on code', 'bad',
     ["my code won't work and I've been stuck for hours", "I'm so frustrated with this bug",
      "I've been debugging all day and nothing works"],
     ["Being stuck on a bug for hours is really frustrating.", "Debugging all day wears anyone down."],
     ["Take a short break; fresh eyes often spot the problem in minutes.",
      "Print the values right before the line that fails to see what's really there.",
      "Explaining the code line by line, even to a rubber duck, often reveals the bug."],
     ["What does the error message say?", "Want to paste the code here so I can take a look?"],
     ["Good luck with the bug! Paste the error here if it's still stuck.", "You're welcome. Bugs always look obvious once you find them."]),
    ('bullied', 'bad',
     ["someone keeps posting mean comments about me", "people at school are making fun of me", "I'm being bullied"],
     ["I'm sorry, nobody deserves to be treated like that.", "Being picked on is really hurtful, and it's not your fault."],
     ["Save screenshots of the comments in case you need to report them.",
      "Block and report the accounts; most apps take bullying reports seriously.",
      "Tell a trusted adult, like a parent, teacher or counselor, so you're not handling it alone."],
     ["How long has this been going on?", "Is there an adult you trust that you could tell?"],
     ["You're welcome. You don't have to deal with this alone.", "Anytime. I hope it stops soon, you deserve better."]),
    ('new job', 'good',
     ["I got the job!", "I just got hired!", "they offered me the position!"],
     ["Congratulations! That's huge news.", "That's amazing, congratulations on the new job!"],
     ["All that preparation paid off.", "Take a moment to celebrate before the first day."],
     ["When do you start?", "What will you be doing?"],
     ["Enjoy the new job!", "You earned it. Good luck in the new role!"]),
    ('passed', 'good',
     ["I passed my math final!", "I got an A on my test!", "I passed all my exams this semester!"],
     ["Congratulations! Your hard work paid off.", "That's great news, well done!"],
     ["You should be proud of yourself.", "Celebrate a little tonight; you earned it."],
     ["Was it the subject you were most worried about?", "What's next for you?"],
     ["Enjoy the win!", "Well deserved. Keep it up!"]),
    ('new pet', 'good',
     ["we adopted a kitten today!", "I got a new puppy yesterday", "my family just got a dog!"],
     ["Aww, congratulations on the new family member!", "That's so exciting, a new pet!"],
     ["The first few days are all about letting them settle in.", "Lots of short play sessions and naps are perfect at that age."],
     ["What's their name?", "What breed are they?"],
     ["Enjoy all the cuddles!", "Have fun with your new buddy!"]),
    ('birthday', 'good',
     ["it's my birthday today!", "today I turn 18!", "it's my birthday and I'm so happy"],
     ["Happy birthday!", "Happy birthday, I hope it's a great one!"],
     ["Do something today that's just for you.", "Birthdays are a great excuse for your favorite food."],
     ["Any plans to celebrate?", "How are you spending the day?"],
     ["Have a wonderful birthday!", "Enjoy your day, you deserve it!"]),
]
THANKS = ['thanks', 'thank you, that helps', "ok I'll try that", 'thanks kani', 'that actually helps a lot']


ACTIVITY_FOLLOW_UPS = {
    'time': (["how long will it take?", "how long does that usually take?", "how long until I'm good at it?"],
             'SEARCH: how long does it take to {vp}'),
    'cost': (["how much does it cost?", "is it expensive?", "how much money do I need for it?"],
             'SEARCH: how much does it cost to {vp}'),
    'need': (["what do I need?", "what do I need to get started?", "what should I get first?"],
             'SEARCH: what you need to {vp}'),
    'start': (["how do I start?", "where do I even begin?", "any tips for a beginner?"],
              'SEARCH: how to {vp} as a beginner'),
    'place': (["where can I do that near me?", "is there anywhere around here for that?", "where can I find that nearby?"],
              'PLACES: {place} @ here'),
}

ACTIVITIES = [
    (["I want to learn guitar", "I'm thinking about learning the guitar"], 'learn guitar', 'guitar lessons',
     ["Do you already have a guitar?", "Are you more into acoustic or electric?", "Are you starting from zero?"], 'time cost need start place'),
    (["I want to learn Spanish", "I'm going to start learning Spanish"], 'learn Spanish', 'Spanish classes',
     ["Is it for travel, school or fun?", "Are you starting from scratch or brushing up?", "Do you know anyone who speaks it?"], 'time cost start place'),
    (["I want to start running", "I'm thinking about getting into running"], 'start running', 'running clubs',
     ["Are you aiming for a distance, like a 5K?", "Have you run before, or is this a fresh start?", "Would you run outside or on a treadmill?"], 'time need start place'),
    (["I want to learn to swim", "I never learned how to swim and I want to"], 'learn to swim', 'swimming lessons',
     ["Are you comfortable in the water already?", "Would you rather take lessons or practice on your own first?", "Is there a pool near you?"], 'time cost start place'),
    (["I want to learn to code", "I'm thinking about learning programming"], 'learn to code', 'coding classes',
     ["Is there something you'd like to build, like a website or a game?", "Have you tried any coding before?", "Would you like to start with Python?"], 'time cost start place'),
    (["I want to start a vegetable garden", "I'm planning to grow my own vegetables"], 'start a vegetable garden', 'garden centers',
     ["Do you have a yard, or would it be pots on a balcony?", "What vegetables would you like to grow first?", "How much sun does your space get?"], 'cost need start place'),
    (["I want to bake my own bread", "I'm going to try making sourdough"], 'bake bread at home', 'baking classes',
     ["Have you baked bread before?", "Are you thinking sourdough or a simple loaf?", "Do you have an oven you like?"], 'time need start place'),
    (["I want to get better at drawing", "I'd like to learn to draw"], 'learn to draw', 'art classes',
     ["What do you like to draw most?", "Pencil or digital?", "Have you drawn much before?"], 'time need start place'),
    (["I want to get good at chess", "I just started playing chess"], 'get better at chess', 'chess clubs',
     ["Do you play online or on a real board?", "Do you know all the rules yet?", "Who do you usually play against?"], 'time start place'),
    (["I'm thinking about starting yoga", "I want to try yoga"], 'start yoga', 'yoga studios',
     ["Is it more for flexibility or for relaxing?", "Would you do it at home or in a class?", "Have you tried any yoga before?"], 'cost need start place'),
    (["I want to learn to drive", "I'm getting my learner's permit soon"], 'learn to drive', 'driving schools',
     ["Do you have someone to practice with?", "Will you learn on an automatic or a manual?", "Are you excited or nervous about it?"], 'time cost start place'),
    (["I want to get into photography", "I just bought my first camera"], 'learn photography', 'photography classes',
     ["What do you like taking pictures of?", "Are you using a phone or a camera?", "Is it more for fun or for work?"], 'time need start place'),
    (["I want to learn to knit", "my friend is teaching me to knit"], 'learn to knit', 'yarn stores',
     ["What would you like to make first?", "Do you have needles and yarn yet?", "Have you tried any crafts like it before?"], 'time need start place'),
    (["I want to start lifting weights", "I'm going to start going to the gym"], 'start lifting weights', 'gyms',
     ["Is it for strength, health or looks?", "Have you lifted before?", "How many days a week can you go?"], 'time cost start place'),
    (["I want to start a podcast", "I'm thinking of making a podcast with my friend"], 'start a podcast', None,
     ["What would it be about?", "Would it be just you or with guests?", "Do you have a mic yet?"], 'cost need start'),
    (["I want to build a gaming PC", "I'm planning my first PC build"], 'build a gaming PC', 'computer stores',
     ["Which games do you want to play on it?", "Do you have a budget in mind?", "Have you built a computer before?"], 'time cost need place'),
    (["I want to go camping this summer", "we're planning our first camping trip"], 'go camping', 'campgrounds',
     ["Tent or cabin?", "Will it be just you or a group?", "Do you know where you'd like to go?"], 'cost need start place'),
    (["I want to try rock climbing", "I'm thinking of trying bouldering"], 'start rock climbing', 'climbing gyms',
     ["Indoors at a gym or outside?", "Have you climbed before?", "Is anyone going with you?"], 'cost need start place'),
    (["I want to learn to cook", "I'm finally going to learn how to cook"], 'learn to cook', 'cooking classes',
     ["What kind of food do you like to eat?", "Do you cook anything already?", "Is it for yourself or for a family?"], 'time need start place'),
    (["I want to learn sign language", "I'm thinking about learning ASL"], 'learn sign language', 'sign language classes',
     ["Is it for someone you know?", "Have you learned any signs yet?", "Would you prefer a class or videos?"], 'time cost start place'),
]
PET_FOLLOW_UPS = {
    'eat': (["what do they eat?", "what should I feed it?", "what kind of food do they need?"], 'SEARCH: what to feed a {animal}'),
    'live': (["how long do they live?", "what's their lifespan?"], 'SEARCH: how long do {animals} live'),
    'cost': (["how much do they cost?", "are they expensive to keep?"], 'SEARCH: how much does it cost to own a {animal}'),
    'need': (["what do I need to buy?", "what supplies do I need?"], 'SEARCH: supplies you need for a {animal}'),
    'place': (["where can I adopt one around here?", "is there a shelter near me?"], 'PLACES: animal shelters @ here'),
}
PETS = [('cat', 'cats', ["I'm thinking about adopting a cat", "we want to get a cat"]),
        ('dog', 'dogs', ["I really want a dog", "we're thinking about getting a dog"]),
        ('hamster', 'hamsters', ["I want a hamster", "my parents said I can get a hamster"]),
        ('parrot', 'parrots', ["I'm thinking about getting a parrot", "I want a pet parrot"]),
        ('guinea pig', 'guinea pigs', ["we might get a guinea pig", "I want a guinea pig"]),
        ('turtle', 'turtles', ["I want a pet turtle", "my son wants a turtle"])]
PET_QUESTIONS = ["Have you had a {animal} before?", "Do you have space at home for a {animal}?",
                 "Would the {animal} be yours or for the whole family?"]
TRIP_FOLLOW_UPS = {
    'weather': (["what's the weather like there?", "what should I pack for the weather?"], 'SEARCH: weather in {city}'),
    'food': (["what should I eat there?", "what food is it known for?"], 'SEARCH: best food to try in {city}'),
    'stay': (["where should I stay?", "which area is best to stay in?"], 'SEARCH: best areas to stay in {city}'),
    'see': (["what should I see?", "what are the must-see places?"], 'SEARCH: top things to do in {city}'),
    'cost': (["is it expensive?", "how much money should I bring?"], 'SEARCH: how expensive is a trip to {city}'),
}
CITIES = ['Tokyo', 'Paris', 'Iceland', 'New York', 'Seoul', 'London', 'Mexico City', 'Bali', 'Lisbon', 'Toronto', 'Barcelona', 'Sydney']
TRIP_SAYS = ["I'm going to {city} next month", "we're planning a trip to {city}", "I just booked a trip to {city}"]
TRIP_QUESTIONS = ["How long will you be in {city}?", "Is it your first time in {city}?", "What are you most excited to do in {city}?"]
PURCHASE_FOLLOW_UPS = {
    'spend': (["how much should I spend?", "what's a good budget?"], 'SEARCH: how much to spend on a {thing}'),
    'look': (["what should I look for?", "what matters most when choosing one?"], 'SEARCH: what to look for when buying a {thing}'),
    'best': (["which one is the best?", "what's the best one right now?"], 'SEARCH: best {thing}'),
    'where': (["where can I buy one around here?", "is there a store near me that sells them?"], 'PLACES: {store} @ here'),
}
PURCHASES = [('used car', 'used car dealers', ["I'm shopping for a used car", "I need to buy a used car"]),
             ('laptop for college', 'electronics stores', ["I need a new laptop for college", "I'm buying a laptop for school"]),
             ('bike', 'bike shops', ["I want to buy a bike", "I'm looking for a new bike"]),
             ('phone', 'phone stores', ["I need a new phone", "my phone is dying so I'm getting a new one"]),
             ('TV', 'electronics stores', ["I'm buying a new TV", "we need a bigger TV"]),
             ('mattress', 'mattress stores', ["I need a new mattress", "I'm shopping for a mattress"])]
PURCHASE_QUESTIONS = ["Do you have a budget in mind for the {thing}?", "What will you mostly use the {thing} for?",
                      "Is this your first {thing}?"]
OPENERS = ['Nice!', 'Great idea!', 'Love that!', 'Fun!', 'Ooh, good one!', 'That sounds great!', 'Exciting!', 'Oh, cool!']

SWITCHES = [
    ("what's the capital of Australia?", 'SEARCH: capital of Australia'),
    ('who wrote Pride and Prejudice?', 'SEARCH: who wrote Pride and Prejudice'),
    ('how far away is the moon?', 'SEARCH: how far away is the moon'),
    ('when did World War 2 end?', 'SEARCH: when did World War 2 end'),
    ("what's the tallest mountain in Africa?", 'SEARCH: tallest mountain in Africa'),
    ('how many bones are in the human body?', 'SEARCH: how many bones are in the human body'),
    ('who painted the Mona Lisa?', 'SEARCH: who painted the Mona Lisa'),
    ('what causes earthquakes?', 'SEARCH: what causes earthquakes'),
    ('how does a rainbow form?', 'SEARCH: how rainbows form'),
    ('who invented the telephone?', 'SEARCH: who invented the telephone'),
    ("what's the biggest ocean?", 'SEARCH: biggest ocean'),
    ('how many players are on a soccer team?', 'SEARCH: how many players are on a soccer team'),
    ('why is the ocean salty?', 'SEARCH: why is the ocean salty'),
    ('how do airplanes stay in the air?', 'SEARCH: how airplanes stay in the air'),
    ("what's the population of Japan?", 'SEARCH: population of Japan'),
    ('who was the first person on the moon?', 'SEARCH: first person on the moon'),
]

EXPLAIN = [
    ('explain how vaccines work', 'how vaccines work'), ('how does the stock market work?', 'how the stock market works'),
    ('explain how a car engine works', 'how a car engine works'), ('how do solar panels work?', 'how solar panels work'),
    ("what's the difference between weather and climate?", 'difference between weather and climate'),
    ("what's the difference between a frog and a toad?", 'difference between a frog and a toad'),
    ('how does the internet work?', 'how the internet works'), ('explain how batteries work', 'how batteries work'),
    ('how do volcanoes erupt?', 'how volcanoes erupt'), ('explain how the heart pumps blood', 'how the heart pumps blood'),
    ('how does GPS work?', 'how GPS works'), ('how do tides work?', 'how tides work'),
    ("what's the difference between a crocodile and an alligator?", 'difference between a crocodile and an alligator'),
    ('explain how credit scores work', 'how credit scores work'), ('how do noise cancelling headphones work?', 'how noise cancelling headphones work'),
    ('how does a refrigerator keep food cold?', 'how a refrigerator works'), ('explain how plants grow from seeds', 'how plants grow from seeds'),
    ('how do airplanes fly?', 'how airplanes fly'), ("what's the difference between a virus and a bacterium?", 'difference between viruses and bacteria'),
    ('explain how elections are counted', 'how election votes are counted'),
]


def normal(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


HELD_OUT = {normal(p) for p in identity_eval.GREETINGS + identity_eval.QUESTIONS + offscript_eval.PROMPTS + ontopic_eval.prompts()}


def held_out(row):
    return any(m['role'] == 'user' and normal(m['content']) in HELD_OUT for m in row['messages'])


def user(text):
    return {'role': 'user', 'content': text}


def assistant(text):
    return {'role': 'assistant', 'content': text}


def wordings(text, rng, count):
    """A few ways a person might type the same message."""
    plain = text.rstrip('!?.')
    forms = [text, text.lower(), f'hey kani, {text}', f'kani {text}', f'{plain}...', text[0].upper() + text[1:]]
    return rng.sample(list(dict.fromkeys(forms)), min(count, len(set(forms))))


class Rows:
    """New training rows, each final reply picked from choices still under CAP."""

    def __init__(self, rng):
        self.rng, self.used, self.rows = rng, collections.Counter(), []

    def add(self, family, messages, choices):
        options = [c for c in choices if self.used[c] < CAP]
        if not options:
            return None
        reply = self.rng.choice(options)
        self.used[reply] += 1
        self.rows.append({'split': 'train', 'family': family, 'source': 'make_instruct_v14', 'messages': [*messages, assistant(reply)]})
        return reply


def situations(rows, rng):
    for name, mood, prompts, acks, tips, asks, thanks in SITUATIONS:
        replies = [' '.join(parts) for parts in itertools.product(acks, tips, asks)]
        for prompt in prompts:
            for text in wordings(prompt, rng, 4):
                reply = rows.add(f'situation:{name}', [user(text)], replies)
                if reply and rng.random() < 0.4:
                    rows.add(f'situation:{name}', [user(text), assistant(reply), user(rng.choice(THANKS))], thanks)


def context_reply(questions, rng, **names):
    return f'{rng.choice(OPENERS)} {rng.choice(questions).format(**names)}'


def follow_ups(rows, rng):
    subjects = []
    for says, vp, place, questions, kinds in ACTIVITIES:
        table = {k: v for k, v in ACTIVITY_FOLLOW_UPS.items() if k in kinds.split()}
        subjects.append(('follow_up:activity', says, questions, table, {'vp': vp, 'place': place}))
    for animal, animals, says in PETS:
        subjects.append(('follow_up:pet', says, PET_QUESTIONS, PET_FOLLOW_UPS, {'animal': animal, 'animals': animals}))
    for city in CITIES:
        subjects.append(('follow_up:trip', [s.format(city=city) for s in TRIP_SAYS], TRIP_QUESTIONS, TRIP_FOLLOW_UPS, {'city': city}))
    for thing, store, says in PURCHASES:
        subjects.append(('follow_up:purchase', says, PURCHASE_QUESTIONS, PURCHASE_FOLLOW_UPS, {'thing': thing, 'store': store}))
    for family, says, questions, table, names in subjects:

        for text in says:
            for form in wordings(text, rng, 2):
                rows.add(family, [user(form)], [f'{o} {q.format(**names)}' for o in OPENERS for q in questions])
        for phrasings, lookup in table.values():
            for phrase in phrasings:
                opening = [user(rng.choice(says)), assistant(context_reply(questions, rng, **names))]
                rows.add(family, [*opening, user(phrase)], [lookup.format(**names)])
        for question, lookup in rng.sample(SWITCHES, 2):
            opening = [user(rng.choice(says)), assistant(context_reply(questions, rng, **names))]
            rows.add('follow_up:switch', [*opening, user(question)], [lookup])
    for question, lookup in EXPLAIN:
        for form in wordings(question, rng, 2):
            rows.add('explain', [user(form)], [f'SEARCH: {lookup}'])


def everyday(split):
    """SmolTalk's everyday conversations without their opening greeting turn."""
    rows = []
    for path in sorted(SMOLTALK.glob(f'{split}-*.parquet')):
        for record in pq.read_table(path, columns=['messages', 'source']).to_pylist():
            if record['source'] != 'everyday-conversations':
                continue
            messages = record['messages']
            if len(messages) > 2 and len(messages[0]['content']) < 24 and messages[1]['content'].startswith('Hello!'):
                messages = messages[2:]
            messages = conversation(messages)
            if messages:
                rows.append({'split': 'train' if split == 'train' else 'validation', 'family': 'everyday_conversation',
                             'source': 'smol-smoltalk/everyday-conversations', 'messages': messages})
    return rows


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def build(seed=2414):
    rng = random.Random(seed)
    rows = Rows(rng)
    situations(rows, rng)
    follow_ups(rows, rng)
    new = rows.rows + everyday('train')
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]] + everyday('test')
    new = new[cut:]
    v13 = [json.loads(line) for line in (ROOT / 'data/instruct-v13.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = [renamed(r) for r in v13 if r['split'] == 'train' and r.get('family') != 'feelings']
    rng.shuffle(replay)
    validation += [renamed(r) for r in v13 if r['split'] != 'train' and r.get('family') != 'feelings']

    seen, kept = collections.Counter(), []
    for row in new + replay[:REPLAY]:
        replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
        if held_out(row) or any(seen[r] >= CAP for r in replies):
            continue
        seen.update(replies)
        kept.append(row)
    rng.shuffle(kept)
    return kept + [r for r in validation if not held_out(r)]


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v14.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant')
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
