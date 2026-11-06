"""websites-v1: complete, working single-file websites for Cide (its main goal is building a working website).

Each example asks for a site in one of many wordings and answers with one HTML document: a header with navigation, a
hero, a features section, one or two interactive features wired with JavaScript, and a footer. Pages are built from
the templates below, so every element the script uses exists and every handler is attached; `check` verifies that,
and data/check_websites.mjs loads a sample in a browser and clicks through it. Themes are split: validation and
kelus.website_eval use themes never trained on, so a passing page is not a remembered one.

    python3 data/make_websites.py   # writes data/websites-v1.jsonl
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent.parent
TRAIN_EXAMPLES, VALIDATION_EXAMPLES = 9_000, 240
MAX_CHARACTERS = 4_000


THEMES = [
    ('Sunny Crumb Bakery', 'bakery', 'Fresh bread every morning', ['Sourdough baked daily', 'Custom birthday cakes', 'Gluten-free options'], ('#fff8ef', '#ffffff', '#3b2a1a', '#e07a2f'), '🥐'),
    ('Iron Oak Gym', 'gym', 'Get stronger, one rep at a time', ['Open 24 hours', 'Personal trainers', 'Group classes'], ('#111418', '#1c2128', '#f2f4f7', '#ff5a36'), '🏋️'),
    ('Bean Theory Café', 'coffee shop', 'Coffee worth waking up for', ['Single-origin beans', 'Oat milk at no extra cost', 'Free Wi-Fi'], ('#f6f1eb', '#ffffff', '#2d211b', '#8b5a3c'), '☕'),
    ('Gearheads Robotics Club', 'school robotics club', 'Build it, code it, race it', ['Weekly build nights', 'Competition team', 'Beginners welcome'], ('#0f172a', '#1e293b', '#e2e8f0', '#38bdf8'), '🤖'),
    ('Happy Paws Walking', 'dog walking service', 'Long walks for good dogs', ['Solo or group walks', 'GPS-tracked routes', 'Photo updates'], ('#f2fbf5', '#ffffff', '#1d3a2a', '#2f9e5b'), '🐕'),
    ('Mira Lens Studio', 'photography portfolio', 'Light, shadow and stories', ['Portraits', 'Weddings', 'Product shoots'], ('#141414', '#1f1f1f', '#f5f5f5', '#d4a373'), '📷'),
    ('Leaf & Pot', 'plant shop', 'Plants that make a home', ['Easy-care plants', 'Repotting service', 'Plant doctor Saturdays'], ('#f4f9f1', '#ffffff', '#20331b', '#4c8a3e'), '🪴'),
    ('Chapter Seven Book Club', 'book club', 'One book, many opinions', ['Monthly meetups', 'Snacks provided', 'All genres'], ('#fbf7f2', '#ffffff', '#2b2235', '#7b4fa0'), '📚'),
    ('Wander Notes', 'travel blog', 'Small trips, big stories', ['City guides', 'Budget tips', 'Packing lists'], ('#eef6fb', '#ffffff', '#16324a', '#1f7ab8'), '🧭'),
    ('The Static Hum', 'indie band', 'Loud songs for quiet towns', ['New album out now', 'Live shows', 'Merch store'], ('#160f1f', '#241a33', '#f3ecff', '#c084fc'), '🎸'),
    ('ByteCamp', 'coding bootcamp', 'Learn to code in twelve weeks', ['Project-based lessons', 'Mentor support', 'Career coaching'], ('#0b1020', '#151c33', '#e6ebff', '#7c9cff'), '💻'),
    ('Slice Society', 'pizza place', 'Wood-fired since 1998', ['Hand-stretched dough', 'Vegan cheese', 'Late-night delivery'], ('#fff5f2', '#ffffff', '#3a1a12', '#d9381e'), '🍕'),
    ('Still Water Yoga', 'yoga studio', 'Breathe in, slow down', ['Morning flow', 'Beginner classes', 'Meditation room'], ('#f5f7f4', '#ffffff', '#24302a', '#6b9080'), '🧘'),
    ('Pixel Forge Games', 'indie game studio', 'Small games with big hearts', ['Puzzle adventures', 'Free demos', 'Discord community'], ('#10101a', '#1b1b2b', '#ececff', '#ff4fa3'), '🎮'),
    ('Spark Science Fair', 'science fair', 'Show us what you discovered', ['Project showcase', 'Judges from local labs', 'Prizes for every grade'], ('#f6f8ff', '#ffffff', '#1c2340', '#4f46e5'), '🔬'),
    ('Sunny Side Lemonade', 'lemonade stand', 'Cold drinks, warm smiles', ['Fresh-squeezed lemons', 'Strawberry special', 'Open weekends'], ('#fffbe6', '#ffffff', '#3d3200', '#f2b705'), '🍋'),
    ('Spoke & Chain', 'bike repair shop', 'Fixed fast, ridden far', ['Same-day tune-ups', 'Flat tire fixes', 'Used bikes for sale'], ('#f4f4f4', '#ffffff', '#1f1f1f', '#e4572e'), '🚲'),
    ('Bright Minds Tutoring', 'tutoring service', 'Homework help that sticks', ['Math and science', 'Essay coaching', 'Online sessions'], ('#f3f8ff', '#ffffff', '#152238', '#2563eb'), '✏️'),
    ('Northlight Gallery', 'art gallery', 'Local artists, open doors', ['Monthly exhibitions', 'Artist talks', 'Free entry'], ('#fafafa', '#ffffff', '#202020', '#b5651d'), '🖼️'),
    ('Bubble Bay Car Wash', 'car wash', 'Shine in ten minutes', ['Touch-free wash', 'Interior detailing', 'Monthly pass'], ('#eefbff', '#ffffff', '#0d3346', '#0891b2'), '🚗'),
    ('Green Fork Kitchen', 'vegetarian restaurant', 'Vegetables, done right', ['Seasonal menu', 'Chef tasting nights', 'Takeout'], ('#f3faf3', '#ffffff', '#1e3320', '#3a8d40'), '🥗'),
    ('Orbit Astronomy Club', 'astronomy club', 'Look up, together', ['Telescope nights', 'Star maps', 'Meteor shower trips'], ('#070b1a', '#111833', '#e4e9ff', '#a78bfa'), '🔭'),
    ('Tidy Nest Cleaning', 'house cleaning service', 'A clean home, without the work', ['Weekly visits', 'Eco-friendly products', 'Move-out cleaning'], ('#f6fbfb', '#ffffff', '#1d3333', '#14b8a6'), '🧽'),
    ('Paper Crane Crafts', 'craft shop', 'Make something today', ['Origami kits', 'Workshops', 'Gift wrapping'], ('#fff7fb', '#ffffff', '#3a1f2d', '#db2777'), '🦢'),
    ('Fast Lane Karting', 'go-kart track', 'Race your friends', ['Indoor track', 'Birthday packages', 'Lap timing'], ('#121212', '#1e1e1e', '#f0f0f0', '#facc15'), '🏁'),
    ('Rooted Farm Market', 'farmers market', 'From the field to your table', ['Local produce', 'Honey and jam', 'Saturday mornings'], ('#fbf9f1', '#ffffff', '#2e2a16', '#7c9a2b'), '🥕'),
    ('Clever Cats Daycare', 'pet daycare', 'Playtime while you work', ['Indoor play area', 'Nap corners', 'Webcam access'], ('#fff9f2', '#ffffff', '#3a2a17', '#f97316'), '🐈'),
    ('Summit Trail Guides', 'hiking guide company', 'Trails worth the climb', ['Guided day hikes', 'Gear rental', 'Sunrise treks'], ('#f1f6f4', '#ffffff', '#1b2e28', '#2f855a'), '⛰️'),
    ('Note & Tone Music School', 'music school', 'Every kid can play', ['Piano and guitar', 'Small classes', 'Spring recital'], ('#faf5ff', '#ffffff', '#2a1f3d', '#9333ea'), '🎹'),
    ('Wave Rider Surf School', 'surf school', 'Stand up on your first day', ['Board rental', 'Kids lessons', 'Wetsuits included'], ('#ecfeff', '#ffffff', '#083344', '#0ea5e9'), '🏄'),
    ('Golden Hour Florist', 'flower shop', 'Flowers for every feeling', ['Same-day bouquets', 'Wedding flowers', 'Plant subscriptions'], ('#fff7f5', '#ffffff', '#3b1f1f', '#e11d48'), '💐'),
    ('Codeine-Free Coders', 'study group for programmers', 'Debug together, learn faster', ['Pair programming', 'Weekly challenges', 'Mock interviews'], ('#0f1115', '#191c22', '#e8eaed', '#22c55e'), '🧑‍💻'),
    ('Moonbeam Ice Cream', 'ice cream shop', 'Scoops under the stars', ['Thirty flavors', 'Dairy-free sorbet', 'Waffle cones made daily'], ('#fdf4ff', '#ffffff', '#3b0d3f', '#c026d3'), '🍦'),
    ('Thunder Paws Rescue', 'animal rescue', 'Every pet deserves a home', ['Adopt a pet', 'Volunteer days', 'Foster program'], ('#fffaf0', '#ffffff', '#33240e', '#d97706'), '🐾'),
    ('Ace Court Tennis', 'tennis club', 'Serve, rally, repeat', ['Night courts', 'Coaching', 'Junior league'], ('#f2fbf2', '#ffffff', '#16301a', '#16a34a'), '🎾'),
    ('Little Lab Makerspace', 'makerspace', 'Tools for every idea', ['3D printers', 'Laser cutter', 'Open shop nights'], ('#f5f5f4', '#ffffff', '#1c1917', '#ea580c'), '🛠️'),
]

HELD_OUT_THEMES = {'Thunder Paws Rescue', 'Ace Court Tennis', 'Little Lab Makerspace', 'Golden Hour Florist'}

ASKS = [
    'Build a complete website for {name}, a {kind}. {features} Use a single HTML file with inline CSS and JavaScript.',
    'Make me a website for my {kind} called {name}. {features}',
    'Create a one-page site for {name} ({kind}). {features} Put everything in one HTML file.',
    'I need a landing page for {name}, a {kind}. {features} Give me the full HTML.',
    'Can you code a website for {name}? It is a {kind}. {features}',
    'write a full html website for a {kind} named {name}. {features} css and js inside the file pls',
    'Design a simple, nice-looking website for {name}. {features} One file please.',
    'Build {name} a homepage. They are a {kind}. {features} Make sure the JavaScript works.',
    'make a website for {name} ({kind}) {features}',
    'Could you write a complete HTML page for a {kind} called {name}? {features} No external libraries.',
]
INTROS = ['Here is a complete website for {name}. Save it as index.html and open it in a browser.',
          "Here's the full single-file site for {name}. Save it as index.html and open it in any browser.",
          'This is a complete one-page website for {name}, with the CSS and JavaScript inside the file.',
          'Here is {name}\'s website in one HTML file. Open it in a browser to try it.']
OUTROS = ['Everything is in one file, so there is nothing else to install.',
          'You can change the colors in the :root block at the top of the CSS.',
          'Edit the text in the HTML to make it yours.', '']


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')






def counter(theme, rng):
    label = rng.choice(['Visitors today', 'Cups served', 'Happy customers', 'Tickets', 'Likes', 'Points'])
    step = rng.choice([1, 1, 1, 5, 10])
    ask = rng.choice([f'Add a counter for "{label}" with plus, minus and reset buttons.',
                      f'I want a {label.lower()} counter that goes up by {step} when you click a button, plus a reset button.' if step > 1
                      else f'Include a {label.lower()} counter with + and - buttons and a reset.'])
    html = f'''<section id="counter-section" class="card">
  <h2>{label}</h2>
  <p class="big-number" id="count">0</p>
  <div class="row">
    <button id="decrease" type="button">−</button>
    <button id="increase" type="button">+</button>
    <button id="reset" type="button" class="secondary">Reset</button>
  </div>
</section>'''
    css = '.big-number { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }'
    js = f'''let count = 0;
const countText = document.getElementById('count');
function showCount() {{ countText.textContent = count; }}
document.getElementById('increase').addEventListener('click', () => {{ count += {step}; showCount(); }});
document.getElementById('decrease').addEventListener('click', () => {{ count = Math.max(0, count - {step}); showCount(); }});
document.getElementById('reset').addEventListener('click', () => {{ count = 0; showCount(); }});'''
    return ask, html, css, js


def todo(theme, rng):
    title = rng.choice(['To-do list', 'Wish list', 'Shopping list', 'Reading list', 'Checklist'])
    save = rng.random() < 0.5
    ask = rng.choice([f'Add a {title.lower()} where I can add items, check them off and delete them.',
                      f'It needs a {title.lower()}: type an item, press Add, click to mark it done, and remove it.'])
    if save:
        ask += ' Save the list so it is still there after a refresh.'
    html = f'''<section id="todo-section" class="card">
  <h2>{title}</h2>
  <form id="todo-form" class="row">
    <input id="todo-input" type="text" placeholder="Add an item" required>
    <button type="submit">Add</button>
  </form>
  <ul id="todo-list"></ul>
</section>'''
    css = '''#todo-list { list-style: none; padding: 0; }
#todo-list li { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(127,127,127,0.25); cursor: pointer; }
#todo-list li.done span { text-decoration: line-through; opacity: 0.6; }
#todo-list button { padding: 0.2rem 0.6rem; }'''
    load = "let items = JSON.parse(localStorage.getItem('items') || '[]');" if save else 'let items = [];'
    store = "localStorage.setItem('items', JSON.stringify(items));" if save else ''
    js = f'''{load}
const list = document.getElementById('todo-list');
function renderList() {{
  list.innerHTML = '';
  items.forEach((item, index) => {{
    const li = document.createElement('li');
    if (item.done) li.classList.add('done');
    const text = document.createElement('span');
    text.textContent = item.text;
    text.addEventListener('click', () => {{ items[index].done = !items[index].done; renderList(); }});
    const remove = document.createElement('button');
    remove.textContent = 'Delete';
    remove.addEventListener('click', () => {{ items.splice(index, 1); renderList(); }});
    li.append(text, remove);
    list.appendChild(li);
  }});
  {store}
}}
document.getElementById('todo-form').addEventListener('submit', (event) => {{
  event.preventDefault();
  const input = document.getElementById('todo-input');
  const text = input.value.trim();
  if (!text) return;
  items.push({{ text, done: false }});
  input.value = '';
  renderList();
}});
renderList();'''
    return ask, html, css, js


def tip(theme, rng):
    ask = rng.choice(['Add a tip calculator: bill amount, tip percent and number of people, showing the tip and the amount each person pays.',
                      'Include a tip calculator that splits the bill between people.'])
    html = '''<section id="tip-section" class="card">
  <h2>Tip calculator</h2>
  <label>Bill ($) <input id="bill" type="number" min="0" step="0.01" value="50"></label>
  <label>Tip (%) <input id="tip-percent" type="number" min="0" value="15"></label>
  <label>People <input id="people" type="number" min="1" value="2"></label>
  <p>Tip: <strong id="tip-amount">$0.00</strong> · Each person pays: <strong id="per-person">$0.00</strong></p>
</section>'''
    css = 'label { display: block; margin: 0.5rem 0; } label input { width: 8rem; margin-left: 0.5rem; }'
    js = '''function updateTip() {
  const bill = parseFloat(document.getElementById('bill').value) || 0;
  const percent = parseFloat(document.getElementById('tip-percent').value) || 0;
  const people = Math.max(1, parseInt(document.getElementById('people').value, 10) || 1);
  const tipAmount = bill * percent / 100;
  document.getElementById('tip-amount').textContent = '$' + tipAmount.toFixed(2);
  document.getElementById('per-person').textContent = '$' + ((bill + tipAmount) / people).toFixed(2);
}
['bill', 'tip-percent', 'people'].forEach((id) => document.getElementById(id).addEventListener('input', updateTip));
updateTip();'''
    return ask, html, css, js


def converter(theme, rng):
    kind = rng.choice(['temperature', 'distance', 'weight'])
    if kind == 'temperature':
        a, b, forward, back, note = 'Celsius', 'Fahrenheit', 'value * 9 / 5 + 32', '(value - 32) * 5 / 9', 'a temperature converter between Celsius and Fahrenheit'
    elif kind == 'distance':
        a, b, forward, back, note = 'Kilometers', 'Miles', 'value * 0.621371', 'value / 0.621371', 'a kilometers to miles converter that works both ways'
    else:
        a, b, forward, back, note = 'Kilograms', 'Pounds', 'value * 2.20462', 'value / 2.20462', 'a kilograms and pounds converter'
    ask = f'Add {note}: typing in either box updates the other.'
    html = f'''<section id="converter-section" class="card">
  <h2>{a} ⇄ {b}</h2>
  <label>{a} <input id="first-unit" type="number" step="any"></label>
  <label>{b} <input id="second-unit" type="number" step="any"></label>
</section>'''
    css = 'label { display: block; margin: 0.5rem 0; } label input { width: 9rem; margin-left: 0.5rem; }'
    js = f'''const firstUnit = document.getElementById('first-unit');
const secondUnit = document.getElementById('second-unit');
firstUnit.addEventListener('input', () => {{
  const value = parseFloat(firstUnit.value);
  secondUnit.value = Number.isNaN(value) ? '' : ({forward}).toFixed(2);
}});
secondUnit.addEventListener('input', () => {{
  const value = parseFloat(secondUnit.value);
  firstUnit.value = Number.isNaN(value) ? '' : ({back}).toFixed(2);
}});'''
    return ask, html, css, js


QUIZZES = [
    [('What color do you get by mixing blue and yellow?', ['Green', 'Purple', 'Orange'], 0),
     ('How many legs does a spider have?', ['Six', 'Eight', 'Ten'], 1),
     ('Which planet is known as the Red Planet?', ['Venus', 'Jupiter', 'Mars'], 2)],
    [('What does HTML stand for?', ['HyperText Markup Language', 'High Tech Modern Language', 'Home Tool Markup Language'], 0),
     ('Which tag makes a link?', ['<p>', '<a>', '<div>'], 1),
     ('Which language styles a web page?', ['Python', 'SQL', 'CSS'], 2)],
    [('How many minutes are in an hour?', ['60', '100', '30'], 0),
     ('What is 7 × 8?', ['54', '56', '64'], 1),
     ('Which is the largest ocean?', ['Atlantic', 'Indian', 'Pacific'], 2)],
]


def quiz(theme, rng):
    questions = rng.choice(QUIZZES)
    ask = rng.choice(['Add a short multiple-choice quiz that shows the score at the end.',
                      'Include a three-question quiz with buttons for answers and a final score.'])
    data = json.dumps([{'q': q, 'options': o, 'answer': a} for q, o, a in questions], ensure_ascii=False)
    html = '''<section id="quiz-section" class="card">
  <h2>Quick quiz</h2>
  <p id="question"></p>
  <div id="answers" class="column"></div>
  <p id="quiz-result"></p>
  <button id="restart" type="button" class="secondary" hidden>Try again</button>
</section>'''
    css = '.column { display: flex; flex-direction: column; gap: 0.5rem; max-width: 24rem; }'
    js = f'''const questions = {data};
let current = 0;
let score = 0;
function showQuestion() {{
  const answers = document.getElementById('answers');
  answers.innerHTML = '';
  if (current >= questions.length) {{
    document.getElementById('question').textContent = 'Done!';
    document.getElementById('quiz-result').textContent = 'You scored ' + score + ' out of ' + questions.length + '.';
    document.getElementById('restart').hidden = false;
    return;
  }}
  const item = questions[current];
  document.getElementById('question').textContent = item.q;
  item.options.forEach((option, index) => {{
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = option;
    button.addEventListener('click', () => {{
      if (index === item.answer) score++;
      current++;
      showQuestion();
    }});
    answers.appendChild(button);
  }});
}}
document.getElementById('restart').addEventListener('click', () => {{
  current = 0;
  score = 0;
  document.getElementById('quiz-result').textContent = '';
  document.getElementById('restart').hidden = true;
  showQuestion();
}});
showQuestion();'''
    return ask, html, css, js


def stopwatch(theme, rng):
    ask = rng.choice(['Add a stopwatch with start, stop and reset buttons.', 'I need a stopwatch on the page (start, pause, reset).'])
    html = '''<section id="stopwatch-section" class="card">
  <h2>Stopwatch</h2>
  <p class="big-number" id="time">00:00.0</p>
  <div class="row">
    <button id="start" type="button">Start</button>
    <button id="stop" type="button">Stop</button>
    <button id="clear" type="button" class="secondary">Reset</button>
  </div>
</section>'''
    css = '.big-number { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; font-variant-numeric: tabular-nums; }'
    js = '''let elapsed = 0;
let timer = null;
function showTime() {
  const minutes = Math.floor(elapsed / 600);
  const seconds = Math.floor(elapsed / 10) % 60;
  const tenths = elapsed % 10;
  document.getElementById('time').textContent = String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0') + '.' + tenths;
}
document.getElementById('start').addEventListener('click', () => {
  if (timer) return;
  timer = setInterval(() => { elapsed++; showTime(); }, 100);
});
document.getElementById('stop').addEventListener('click', () => { clearInterval(timer); timer = null; });
document.getElementById('clear').addEventListener('click', () => { clearInterval(timer); timer = null; elapsed = 0; showTime(); });'''
    return ask, html, css, js


def contact(theme, rng):
    ask = rng.choice(['Add a contact form that checks the name, a valid email and a message before it says thank you.',
                      'Include a contact form with validation (email must look right, message at least 10 characters).'])
    html = '''<section id="contact" class="card">
  <h2>Contact us</h2>
  <form id="contact-form" novalidate>
    <label>Name <input id="contact-name" type="text"></label>
    <label>Email <input id="contact-email" type="email"></label>
    <label>Message <textarea id="contact-message" rows="4"></textarea></label>
    <p id="form-error" class="error"></p>
    <button type="submit">Send</button>
  </form>
  <p id="form-success" hidden>Thanks! We will get back to you soon.</p>
</section>'''
    css = '''#contact-form label { display: block; margin: 0.6rem 0; }
#contact-form input, #contact-form textarea { display: block; width: 100%; max-width: 28rem; margin-top: 0.3rem; }
.error { color: #dc2626; min-height: 1.2em; }'''
    js = '''document.getElementById('contact-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const name = document.getElementById('contact-name').value.trim();
  const email = document.getElementById('contact-email').value.trim();
  const message = document.getElementById('contact-message').value.trim();
  const error = document.getElementById('form-error');
  if (!name) { error.textContent = 'Please enter your name.'; return; }
  if (!/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email)) { error.textContent = 'Please enter a valid email address.'; return; }
  if (message.length < 10) { error.textContent = 'Your message should be at least 10 characters.'; return; }
  error.textContent = '';
  document.getElementById('contact-form').hidden = true;
  document.getElementById('form-success').hidden = false;
});'''
    return ask, html, css, js


def dark_mode(theme, rng):
    ask = rng.choice(['Add a dark mode toggle button that remembers the choice.', 'Include a light/dark theme switch.'])
    html = '<button id="theme-toggle" type="button" class="secondary">🌙 Dark mode</button>'
    css = '''body.dark { --bg: #121212; --surface: #1e1e1e; --text: #f1f1f1; }
#theme-toggle { position: fixed; right: 1rem; bottom: 1rem; }'''
    js = '''const themeButton = document.getElementById('theme-toggle');
function applyTheme(dark) {
  document.body.classList.toggle('dark', dark);
  themeButton.textContent = dark ? '☀️ Light mode' : '🌙 Dark mode';
}
applyTheme(localStorage.getItem('theme') === 'dark');
themeButton.addEventListener('click', () => {
  const dark = !document.body.classList.contains('dark');
  localStorage.setItem('theme', dark ? 'dark' : 'light');
  applyTheme(dark);
});'''
    return ask, html, css, js


def faq(theme, rng):
    ask = rng.choice(['Add an FAQ section where clicking a question shows the answer.', 'Include a collapsible FAQ (accordion).'])
    pairs = [('What are your hours?', 'We are open every day from 9am to 6pm.'),
             ('Do I need to book ahead?', 'Booking helps, but walk-ins are always welcome.'),
             ('Where are you?', f"Right in the middle of town — look for the {theme[5]} sign.")]
    items = '\n'.join(f'''  <div class="faq-item">
    <button type="button" class="faq-question">{q}</button>
    <p class="faq-answer" hidden>{a}</p>
  </div>''' for q, a in pairs)
    html = f'''<section id="faq" class="card">
  <h2>Questions</h2>
{items}
</section>'''
    css = '''.faq-question { width: 100%; text-align: left; margin-top: 0.5rem; }
.faq-answer { margin: 0.5rem 0 0.5rem 0.5rem; }'''
    js = '''document.querySelectorAll('.faq-question').forEach((button) => {
  button.addEventListener('click', () => {
    const answer = button.nextElementSibling;
    answer.hidden = !answer.hidden;
  });
});'''
    return ask, html, css, js


def tabs(theme, rng):
    ask = rng.choice(['Show the menu, prices and location in tabs.', 'Add tabs so visitors can switch between About, Prices and Location.'])
    html = '''<section id="tabs-section" class="card">
  <div class="row" role="tablist">
    <button type="button" class="tab active" data-tab="about">About</button>
    <button type="button" class="tab" data-tab="prices">Prices</button>
    <button type="button" class="tab" data-tab="location">Location</button>
  </div>
  <div class="tab-panel" id="about"><p>We started small and grew with our neighbors.</p></div>
  <div class="tab-panel" id="prices" hidden><p>Most visits cost between $5 and $20.</p></div>
  <div class="tab-panel" id="location" hidden><p>123 Main Street, open every day.</p></div>
</section>'''
    css = '.tab.active { background: var(--accent); color: #fff; } .tab-panel { padding-top: 0.75rem; }'
    js = '''document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((other) => other.classList.toggle('active', other === tab));
    document.querySelectorAll('.tab-panel').forEach((panel) => { panel.hidden = panel.id !== tab.dataset.tab; });
  });
});'''
    return ask, html, css, js


def modal(theme, rng):
    ask = rng.choice(['Add a "Book now" button that opens a popup with a close button.', 'Include a modal popup for a special offer.'])
    offer = rng.choice(['10% off your first visit', 'a free drink with any order', 'two-for-one on Tuesdays'])
    html = f'''<section id="offer" class="card">
  <h2>Special offer</h2>
  <button id="open-modal" type="button">Book now</button>
  <div id="modal" class="modal" hidden>
    <div class="modal-box">
      <p>This week only: {offer}!</p>
      <button id="close-modal" type="button" class="secondary">Close</button>
    </div>
  </div>
</section>'''
    css = '''.modal { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
.modal[hidden] { display: none; }
.modal-box { background: var(--surface); color: var(--text); padding: 1.5rem; border-radius: 12px; max-width: 20rem; }'''
    js = '''const modal = document.getElementById('modal');
document.getElementById('open-modal').addEventListener('click', () => { modal.hidden = false; });
document.getElementById('close-modal').addEventListener('click', () => { modal.hidden = true; });
modal.addEventListener('click', (event) => { if (event.target === modal) modal.hidden = true; });'''
    return ask, html, css, js


def slideshow(theme, rng):
    ask = rng.choice(['Add a slideshow with previous and next buttons.', 'Include an image carousel (use colored boxes as placeholders).'])
    colors = rng.sample(['#f87171', '#60a5fa', '#34d399', '#fbbf24', '#a78bfa', '#f472b6'], 3)
    captions = ['Our place', 'Our team', 'Our favorites']
    data = json.dumps([{'color': c, 'caption': t} for c, t in zip(colors, captions)])
    html = '''<section id="gallery" class="card">
  <h2>Gallery</h2>
  <div id="slide" class="slide"></div>
  <p id="caption"></p>
  <div class="row">
    <button id="previous" type="button">‹ Previous</button>
    <button id="next" type="button">Next ›</button>
  </div>
</section>'''
    css = '.slide { height: 200px; border-radius: 12px; transition: background 0.3s; }'
    js = f'''const slides = {data};
let slideIndex = 0;
function showSlide() {{
  document.getElementById('slide').style.background = slides[slideIndex].color;
  document.getElementById('caption').textContent = slides[slideIndex].caption + ' (' + (slideIndex + 1) + '/' + slides.length + ')';
}}
document.getElementById('next').addEventListener('click', () => {{ slideIndex = (slideIndex + 1) % slides.length; showSlide(); }});
document.getElementById('previous').addEventListener('click', () => {{ slideIndex = (slideIndex - 1 + slides.length) % slides.length; showSlide(); }});
showSlide();'''
    return ask, html, css, js


def quotes(theme, rng):
    ask = rng.choice(['Add a button that shows a random tip each time it is clicked.', 'Include a random quote generator.'])
    lines = rng.sample(['Small steps every day add up.', 'Ask questions — that is how you learn.', 'Rest is part of the work.',
                        'Be kind; everyone is figuring it out.', 'Done is better than perfect.', 'Try again tomorrow.'], 4)
    html = '''<section id="tips" class="card">
  <h2>Tip of the moment</h2>
  <p id="quote">Click the button for a tip.</p>
  <button id="new-quote" type="button">New tip</button>
</section>'''
    css = '#quote { font-style: italic; min-height: 1.5em; }'
    js = f'''const tips = {json.dumps(lines)};
let lastTip = -1;
document.getElementById('new-quote').addEventListener('click', () => {{
  let pick = Math.floor(Math.random() * tips.length);
  if (pick === lastTip) pick = (pick + 1) % tips.length;
  lastTip = pick;
  document.getElementById('quote').textContent = tips[pick];
}});'''
    return ask, html, css, js


def guess(theme, rng):
    top = rng.choice([10, 20, 50, 100])
    ask = f'Add a guess-the-number game from 1 to {top} that says higher or lower.'
    html = f'''<section id="game" class="card">
  <h2>Guess the number (1–{top})</h2>
  <form id="guess-form" class="row">
    <input id="guess-input" type="number" min="1" max="{top}" required>
    <button type="submit">Guess</button>
  </form>
  <p id="guess-hint"></p>
  <button id="new-game" type="button" class="secondary">New game</button>
</section>'''
    css = '#guess-input { width: 6rem; }'
    js = f'''let secret = Math.floor(Math.random() * {top}) + 1;
let tries = 0;
document.getElementById('guess-form').addEventListener('submit', (event) => {{
  event.preventDefault();
  const guessValue = parseInt(document.getElementById('guess-input').value, 10);
  const hint = document.getElementById('guess-hint');
  if (Number.isNaN(guessValue)) {{ hint.textContent = 'Type a number first.'; return; }}
  tries++;
  if (guessValue < secret) hint.textContent = 'Higher!';
  else if (guessValue > secret) hint.textContent = 'Lower!';
  else hint.textContent = 'You got it in ' + tries + ' tries!';
}});
document.getElementById('new-game').addEventListener('click', () => {{
  secret = Math.floor(Math.random() * {top}) + 1;
  tries = 0;
  document.getElementById('guess-hint').textContent = '';
  document.getElementById('guess-input').value = '';
}});'''
    return ask, html, css, js


def cart(theme, rng):
    products = rng.sample([('Small', 3.5), ('Medium', 5.0), ('Large', 6.5), ('Combo', 9.0), ('Gift card', 20.0), ('Sticker', 1.0)], 3)
    ask = rng.choice(['Add a small menu where clicking "Add" puts items in a cart and shows the total.',
                      'Include a mini shop: a few products with Add buttons, a cart list and a running total.'])
    items = '\n'.join(f'''    <li>{name} — ${price:.2f} <button type="button" class="add-item" data-name="{name}" data-price="{price}">Add</button></li>'''
                      for name, price in products)
    html = f'''<section id="shop" class="card">
  <h2>Order</h2>
  <ul class="menu">
{items}
  </ul>
  <h3>Cart</h3>
  <ul id="cart"></ul>
  <p>Total: <strong id="cart-total">$0.00</strong></p>
  <button id="empty-cart" type="button" class="secondary">Empty cart</button>
</section>'''
    css = '.menu { list-style: none; padding: 0; } .menu li { margin: 0.4rem 0; }'
    js = '''const cart = [];
function showCart() {
  const list = document.getElementById('cart');
  list.innerHTML = '';
  cart.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item.name + ' — $' + item.price.toFixed(2);
    list.appendChild(li);
  });
  const total = cart.reduce((sum, item) => sum + item.price, 0);
  document.getElementById('cart-total').textContent = '$' + total.toFixed(2);
}
document.querySelectorAll('.add-item').forEach((button) => {
  button.addEventListener('click', () => {
    cart.push({ name: button.dataset.name, price: parseFloat(button.dataset.price) });
    showCart();
  });
});
document.getElementById('empty-cart').addEventListener('click', () => { cart.length = 0; showCart(); });'''
    return ask, html, css, js


def filter_list(theme, rng):
    entries = rng.sample(['Apple', 'Banana', 'Cherry', 'Mango', 'Grape', 'Peach', 'Lemon', 'Orange', 'Kiwi', 'Plum'], 6)
    ask = rng.choice(['Add a search box that filters a list as you type.', 'Include a searchable list of items.'])
    items = '\n'.join(f'    <li>{e}</li>' for e in entries)
    html = f'''<section id="search-section" class="card">
  <h2>Find something</h2>
  <input id="search" type="search" placeholder="Search…">
  <ul id="items">
{items}
  </ul>
  <p id="no-results" hidden>Nothing matches.</p>
</section>'''
    css = '#search { width: 100%; max-width: 20rem; }'
    js = '''document.getElementById('search').addEventListener('input', (event) => {
  const term = event.target.value.toLowerCase();
  let shown = 0;
  document.querySelectorAll('#items li').forEach((item) => {
    const match = item.textContent.toLowerCase().includes(term);
    item.hidden = !match;
    if (match) shown++;
  });
  document.getElementById('no-results').hidden = shown > 0;
});'''
    return ask, html, css, js


def char_count(theme, rng):
    limit = rng.choice([100, 140, 200, 280])
    ask = f'Add a message box with a live character counter (limit {limit}).'
    html = f'''<section id="message-section" class="card">
  <h2>Leave a note</h2>
  <textarea id="note" rows="4" maxlength="{limit}" placeholder="Write something…"></textarea>
  <p><span id="chars-left">{limit}</span> characters left</p>
</section>'''
    css = '#note { width: 100%; max-width: 30rem; }'
    js = f'''document.getElementById('note').addEventListener('input', (event) => {{
  document.getElementById('chars-left').textContent = {limit} - event.target.value.length;
}});'''
    return ask, html, css, js


FEATURES = [counter, todo, tip, converter, quiz, stopwatch, contact, dark_mode, faq, tabs, modal, slideshow, quotes, guess, cart,
            filter_list, char_count]




BASE_CSS = ''':root {{ --bg: {bg}; --surface: {surface}; --text: {text}; --accent: {accent}; }}
body {{ margin: 0; font-family: {font}; background: var(--bg); color: var(--text); }}
header {{ display: flex; justify-content: space-between; padding: 1rem; }}
header a {{ color: inherit; margin-left: 1rem; }}
.hero {{ text-align: center; padding: 3rem 1rem; }}
main {{ max-width: 900px; margin: auto; padding: 0 1rem; }}
.features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
.card {{ background: var(--surface); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }}
.row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
button {{ background: var(--accent); color: #fff; border: 0; border-radius: 8px; padding: 0.5rem 1rem; cursor: pointer; }}
button.secondary {{ background: none; color: var(--text); border: 1px solid; }}
input, textarea {{ padding: 0.5rem; border-radius: 8px; }}
footer {{ text-align: center; padding: 2rem; }}'''
FONTS = ['system-ui, sans-serif', "'Segoe UI', Roboto, sans-serif", 'Georgia, serif', 'Verdana, sans-serif', "'Trebuchet MS', sans-serif"]


def page(theme, features, rng):
    name, kind, tagline, perks, (bg, surface, text, accent), icon = theme
    sections = [feature(theme, rng) for feature in features]
    nav = ''.join(f'<a href="#{s[1].split(chr(34))[1]}">{label}</a>'
                  for s, label in zip(sections, ['Try it', 'More', 'Extra']) if s[1].startswith('<section'))
    perk_cards = '\n'.join(f'    <div class="card"><h3>{p}</h3></div>' for p in perks)
    css = BASE_CSS.format(bg=bg, surface=surface, text=text, accent=accent, font=rng.choice(FONTS)) + '\n' + '\n'.join(s[2] for s in sections)
    script = '\n\n'.join(s[3] for s in sections) + "\n\ndocument.getElementById('year').textContent = new Date().getFullYear();"
    first = next((s[1].split('"')[1] for s in sections if s[1].startswith('<section')), 'features')
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name}</title>
<style>
{css}
</style>
</head>
<body>
<header>
  <span class="logo">{icon} {name}</span>
  <nav>{nav}</nav>
</header>
<section class="hero">
  <h1>{name}</h1>
  <p>{tagline}</p>
  <a href="#{first}"><button type="button">{rng.choice(['Get started', 'Take a look', 'Try it now', 'See more'])}</button></a>
</section>
<main>
  <section id="features" class="features">
{perk_cards}
  </section>
{chr(10).join(s[1] for s in sections)}
</main>
<footer>&copy; <span id="year"></span> {name}</footer>
<script>
{script}
</script>
</body>
</html>'''
    return [s[0] for s in sections], html


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.open, self.ok = set(), [], True

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ok &= attrs['id'] not in self.ids
            self.ids.add(attrs['id'])
        if tag not in ('meta', 'input', 'br', 'img', 'link'):
            self.open.append(tag)

    def handle_endtag(self, tag):
        if not self.open or self.open[-1] != tag:
            self.ok = False
        else:
            self.open.pop()


def check(html):
    """Balanced tags, unique ids, and every id the script looks up exists."""
    parser = Structure()
    parser.feed(html)
    script = html.split('<script>')[1].split('</script>')[0]
    wanted = set(re.findall(r"getElementById\('([^']+)'\)", script))
    return parser.ok and not parser.open and wanted <= parser.ids


def example(theme, rng, split):
    features = rng.sample(FEATURES, rng.choice([1, 1, 2]))
    asks, html = page(theme, features, rng)
    name, kind = theme[0], theme[1]
    prompt = rng.choice(ASKS).format(name=name, kind=kind, features=' '.join(asks))
    if rng.random() < 0.15:
        prompt = prompt.lower()
    reply = rng.choice(INTROS).format(name=name) + '\n\n```html\n' + html + '\n```'
    outro = rng.choice(OUTROS)
    if outro:
        reply += '\n\n' + outro
    if not check(html) or len(prompt) + len(reply) > MAX_CHARACTERS:
        return None
    return {'split': split, 'family': 'website', 'source': 'make_websites', 'language': 'html',
            'features': [f.__name__ for f in features], 'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]}


def build(seed=2414):
    rng = random.Random(seed)
    train_themes = [t for t in THEMES if t[0] not in HELD_OUT_THEMES]
    held_out = [t for t in THEMES if t[0] in HELD_OUT_THEMES]
    rows = []
    for split, themes, wanted in (('train', train_themes, TRAIN_EXAMPLES), ('validation', held_out, VALIDATION_EXAMPLES)):
        made, seen = 0, set()
        while made < wanted:
            row = example(rng.choice(themes), rng, split)
            if row and row['messages'][1]['content'] not in seen:
                seen.add(row['messages'][1]['content'])
                rows.append(row)
                made += 1
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/websites-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    sizes = sorted(len(r['messages'][0]['content']) + len(r['messages'][1]['content']) for r in rows)
    print(f'{destination}: {len(rows):,} sites; characters median {sizes[len(sizes) // 2]:,}, longest {sizes[-1]:,}')
