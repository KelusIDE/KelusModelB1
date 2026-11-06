"""web-basics-v1: plain, everyday web requests for Cide ("Code a basic login html", "todo app html", "make a calculator").

Cide's other website data always asks for "a website for {business}, a {kind}" with a list of features, so a short
request for one common page got a wrong or empty answer (Cide 4.1: "A simple HTML file that includes a <a> tag";
"build a basic hello world html" gave one <a> link). Each example here asks for one thing in casual wording and
answers with one working HTML file: the hello world, starter template, table, personal, navbar, button, login, signup,
calculator, countdown and landing pages below, plus every feature from make_websites.py as a page of its own. `valid`
(balanced tags, every id the script uses exists) runs on each page, and data/check_websites.mjs clicks through a
sample in a browser. Validation rows use wordings that never appear in training (HELD_OUT_ASKS).

    python3 data/make_web_basics.py   # writes data/web-basics-v1.jsonl
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from make_websites import BASE_CSS, FEATURES, FONTS, HELD_OUT_THEMES, OUTROS, THEMES, Structure, check, page  # noqa: E402

TRAIN_EXAMPLES, VALIDATION_EXAMPLES = 7_000, 168
MAX_CHARACTERS = 9_000

TRAIN_ASKS = [
    'Code a basic {thing} html', 'code {a} in html', 'Make {a} in HTML.', 'make a simple {thing}', 'Make a simple {thing} page.',
    'Write {a} using HTML, CSS and JavaScript.', '{thing} html', 'html {thing}', '{thing} html css js', 'Can you make me {a}?',
    'can u code {a}', 'I need a basic {thing} for my website.', 'Create a simple {thing} with HTML, CSS and JS.',
    'Build {a} webpage.', 'Give me the code for {a}.', 'How do I make {a} in HTML? Write the code.', 'simple {thing} website',
    'Write HTML for {a}.', 'Code {a} for me', 'basic {thing} in one html file', 'Make a working {thing} with JavaScript.',
    'I want to make {a}. Can you write it?', 'Build me a quick {thing}.', 'write code for {a} page',
    'build a basic {thing} html', 'Make a basic {thing}.', 'basic {thing}',
]
HELD_OUT_ASKS = [
    'Could you put together {a} in a single HTML file?', 'quick {thing} in html please', 'I want {a}. Write it in HTML.',
    'Show me a working {thing} made with HTML and JavaScript.', 'need {a} webpage, can you do it',
]
INTROS = ['Here is {a} in one HTML file, with the CSS and JavaScript inside. Save it as index.html and open it in a browser.',
          "Here's {a}. Everything is in one file: save it as index.html and open it in any browser.",
          'This is a complete {thing} in a single HTML file. Open it in a browser to try it.',
          'Here is a simple, working {thing}. Save the code below as an .html file and open it.']
EMAIL = r'/^[^\s@]+@[^\s@]+\.[^\s@]+$/'
PAGE_CSS = '''main { max-width: 32rem; padding: 2rem 1rem; }
h1 { text-align: center; }
label { display: block; margin: 0.8rem 0; }
label input { display: block; width: 100%; box-sizing: border-box; margin-top: 0.3rem; }'''


def hello(theme, rng):
    greeting = rng.choice(['Hello, World!', 'Hello World', 'Hello, world!'])
    head = '<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Hello World</title>'
    style = rng.random() < 0.6
    if style:
        head += f'''
<style>
body {{ font-family: {rng.choice(FONTS)}; display: grid; place-items: center; min-height: 100vh; margin: 0; }}
h1 {{ color: {theme[4][3]}; }}
</style>'''
    body = f'<h1>{greeting}</h1>'
    if rng.random() < 0.4:
        body += '\n<p>This is my first web page.</p>'
    script = rng.random() < 0.3
    if script:
        body += '''
<button id="greet" type="button">Say hello</button>
<p id="greeting"></p>
<script>
document.getElementById('greet').addEventListener('click', () => {
  document.getElementById('greeting').textContent = 'Hello from JavaScript!';
});
</script>'''
    html = f'<!DOCTYPE html>\n<html lang="en">\n<head>\n{head}\n</head>\n<body>\n{body}\n</body>\n</html>'
    note = ('Save it as index.html and open it in your browser. <h1> is the biggest heading: change the text between the tags '
            'to change what the page says.' + (' The <style> block sets the font and centers the text.' if style else '')
            + (' The <script> at the bottom runs when the button is clicked.' if script else ''))
    return html, note


def template(theme, rng):
    title = rng.choice(['My Page', 'My Website', 'Document', 'Home'])
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    /* Your CSS goes here. */
    body {{ font-family: {rng.choice(FONTS)}; margin: 0; line-height: 1.5; }}
    header, main, footer {{ padding: 1rem 2rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
  </header>
  <main>
    <p>Your content goes here.</p>
  </main>
  <footer>
    <p>&copy; {title}</p>
  </footer>
  <script>
    // Your JavaScript goes here.
  </script>
</body>
</html>'''
    note = ('The <head> holds the page title and settings, <body> holds what people see, and the CSS and JavaScript go in '
            'the <style> and <script> tags.')
    return html, note


TABLES = [
    ('Products', ['Item', 'Price', 'In stock'], [['Notebook', '$4.99', 'Yes'], ['Pencil set', '$2.50', 'Yes'], ['Backpack', '$29.00', 'No'], ['Water bottle', '$12.00', 'Yes']]),
    ('Class schedule', ['Day', 'Subject', 'Time'], [['Monday', 'Math', '9:00'], ['Tuesday', 'Science', '10:30'], ['Wednesday', 'History', '13:00'], ['Thursday', 'Art', '14:15']]),
    ('Scores', ['Player', 'Games', 'Points'], [['Mia', '12', '340'], ['Leo', '10', '295'], ['Ava', '11', '310'], ['Sam', '9', '260']]),
    ('Planets', ['Planet', 'Moons', 'Day length'], [['Mercury', '0', '59 days'], ['Venus', '0', '243 days'], ['Earth', '1', '24 hours'], ['Mars', '2', '24.6 hours']]),
]


def table(theme, rng):
    title, columns, rows = rng.choice(TABLES)
    head = ''.join(f'<th>{c}</th>' for c in columns)
    body = '\n'.join('      <tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{ font-family: {rng.choice(FONTS)}; padding: 2rem; }}
table {{ border-collapse: collapse; width: 100%; max-width: 40rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
th {{ background: {theme[4][3]}; color: #fff; }}
tbody tr:nth-child(even) {{ background: #f4f4f4; }}
</style>
</head>
<body>
  <table>
    <caption>{title}</caption>
    <thead>
      <tr>{head}</tr>
    </thead>
    <tbody>
{body}
    </tbody>
  </table>
</body>
</html>'''
    return html, 'Add a row by copying one <tr> line inside <tbody>; <th> cells are the column headings.'


PEOPLE = ['Alex Rivera', 'Jordan Lee', 'Sam Patel', 'Riley Chen', 'Taylor Brooks', 'Morgan Diaz']


def profile(theme, rng):
    name = rng.choice(PEOPLE)
    role = rng.choice(['web developer', 'student', 'designer', 'photographer'])
    skills = rng.sample(['HTML', 'CSS', 'JavaScript', 'Python', 'Design', 'Photography', 'Writing'], 4)
    projects = rng.sample([('Weather app', 'Shows the forecast for any city.'), ('Recipe site', 'Simple recipes with photos.'),
                           ('Game', 'A small browser game made with JavaScript.'), ('Blog', 'Notes about what I am learning.')], 3)
    cards = '\n'.join(f'      <div class="card"><h3>{p}</h3><p>{d}</p></div>' for p, d in projects)
    items = ''.join(f'<li>{s}</li>' for s in skills)
    _, _, _, _, (bg, surface, text, accent), _ = theme
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name}</title>
<style>
body {{ margin: 0; font-family: {rng.choice(FONTS)}; background: {bg}; color: {text}; }}
header {{ text-align: center; padding: 3rem 1rem; }}
section {{ max-width: 48rem; margin: auto; padding: 1rem; }}
.projects {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 1rem; }}
.card {{ background: {surface}; border-radius: 12px; padding: 1rem; }}
a {{ color: {accent}; }}
</style>
</head>
<body>
  <header>
    <h1>{name}</h1>
    <p>I am a {role}.</p>
  </header>
  <section id="about">
    <h2>About me</h2>
    <p>Hi! I'm {name.split()[0]}, a {role} who likes building things and learning new skills.</p>
  </section>
  <section id="skills">
    <h2>Skills</h2>
    <ul>{items}</ul>
  </section>
  <section id="projects">
    <h2>Projects</h2>
    <div class="projects">
{cards}
    </div>
  </section>
  <section id="contact">
    <h2>Contact</h2>
    <p>Email me at <a href="mailto:hello@example.com">hello@example.com</a>.</p>
  </section>
</body>
</html>'''
    return html, 'Replace the name, text and email with your own.'


def navbar(theme, rng):
    brand = rng.choice(['Brand', 'MySite', 'Logo'])
    links = rng.choice([['Home', 'About', 'Services', 'Contact'], ['Home', 'Blog', 'Projects', 'Contact'], ['Home', 'Menu', 'About']])
    items = '\n'.join(f'      <li><a href="#{l.lower()}">{l}</a></li>' for l in links)
    sections = '\n'.join(f'  <section id="{l.lower()}"><h2>{l}</h2><p>The {l.lower()} section.</p></section>' for l in links)
    _, _, _, _, (bg, surface, text, accent), _ = theme
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{brand}</title>
<style>
body {{ margin: 0; font-family: {rng.choice(FONTS)}; }}
.topbar {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; background: {accent}; }}
.topbar a {{ color: #fff; text-decoration: none; }}
.logo {{ font-weight: bold; font-size: 1.2rem; }}
#menu {{ list-style: none; display: flex; gap: 1rem; margin: 0; padding: 0; }}
#menu-toggle {{ display: none; background: none; border: 1px solid #fff; color: #fff; border-radius: 6px; padding: 0.3rem 0.6rem; }}
section {{ padding: 2rem 1rem; }}
@media (max-width: 600px) {{
  #menu-toggle {{ display: block; }}
  nav {{ width: 100%; }}
  #menu {{ display: none; flex-direction: column; padding-top: 0.5rem; }}
  #menu.open {{ display: flex; }}
}}
</style>
</head>
<body>
  <header class="topbar">
    <a class="logo" href="#home">{brand}</a>
    <button id="menu-toggle" type="button" aria-expanded="false" aria-controls="menu">&#9776; Menu</button>
    <nav>
      <ul id="menu">
{items}
      </ul>
    </nav>
  </header>
{sections}
  <script>
    const toggle = document.getElementById('menu-toggle');
    toggle.addEventListener('click', () => {{
      const open = document.getElementById('menu').classList.toggle('open');
      toggle.setAttribute('aria-expanded', open);
    }});
  </script>
</body>
</html>'''
    return html, 'On narrow screens the links fold into the Menu button.'


def button(theme, rng):
    label = rng.choice(['Click me', 'Press me', 'Say hi'])
    counting = rng.random() < 0.5
    html = f'''<section class="card">
  <button id="message-button" type="button">{label}</button>
  <p id="message"></p>
</section>'''
    if counting:
        js = '''let clicks = 0;
document.getElementById('message-button').addEventListener('click', () => {
  clicks += 1;
  document.getElementById('message').textContent = 'You clicked the button ' + clicks + (clicks === 1 ? ' time.' : ' times.');
});'''
    else:
        js = '''document.getElementById('message-button').addEventListener('click', () => {
  document.getElementById('message').textContent = 'Hello! You clicked the button.';
});'''
    return 'Button', html, '', js, 'addEventListener runs the function each time the button is clicked.'


def login(theme, rng):
    email = rng.random() < 0.6
    show, remember = rng.random() < 0.5, rng.random() < 0.4
    title = rng.choice(['Log in', 'Sign in', 'Login'])
    field = ('Email', 'email', 'email') if email else ('Username', 'text', 'username')
    html = f'''<section class="card">
  <form id="login-form" novalidate>
    <label>{field[0]} <input id="login-user" type="{field[1]}" autocomplete="{field[2]}"></label>
    <label>Password <input id="login-password" type="password" autocomplete="current-password"></label>'''
    if show:
        html += '\n    <button id="toggle-password" type="button" class="secondary">Show</button>'
    if remember:
        html += '\n    <label class="inline"><input id="remember" type="checkbox"> Remember me</label>'
    html += f'''
    <p id="login-error" class="error"></p>
    <button type="submit">{title}</button>
  </form>
  <p id="login-success" hidden></p>
</section>'''
    css = '.error { color: #dc2626; min-height: 1.2em; }\n.inline { display: flex; gap: 0.5rem; align-items: center; }\n.inline input { width: auto; margin: 0; }'
    check_user = (f"if (!{EMAIL}.test(user)) {{ error.textContent = 'Please enter a valid email address.'; return; }}" if email else
                  "if (user.length < 3) { error.textContent = 'Your username needs at least 3 characters.'; return; }")
    js = ''
    if remember:
        js += '''const saved = localStorage.getItem('remembered-user');
if (saved) {
  document.getElementById('login-user').value = saved;
  document.getElementById('remember').checked = true;
}
'''
    js += f'''document.getElementById('login-form').addEventListener('submit', (event) => {{
  event.preventDefault();
  const user = document.getElementById('login-user').value.trim();
  const password = document.getElementById('login-password').value;
  const error = document.getElementById('login-error');
  {check_user}
  if (password.length < 6) {{ error.textContent = 'Your password needs at least 6 characters.'; return; }}
  error.textContent = '';'''
    if remember:
        js += '''
  if (document.getElementById('remember').checked) localStorage.setItem('remembered-user', user);
  else localStorage.removeItem('remembered-user');'''
    js += '''
  document.getElementById('login-form').hidden = true;
  const success = document.getElementById('login-success');
  success.textContent = 'Welcome back, ' + user + '!';
  success.hidden = false;
});'''
    if show:
        js += '''
document.getElementById('toggle-password').addEventListener('click', () => {
  const password = document.getElementById('login-password');
  const hidden = password.type === 'password';
  password.type = hidden ? 'text' : 'password';
  document.getElementById('toggle-password').textContent = hidden ? 'Hide' : 'Show';
});'''
    note = ('This page only checks the form in the browser. A real login needs a server that checks the password '
            '(stored as a hash, never as plain text), so connect the form to your backend before using it.')
    return title, html, css, js, note


def signup(theme, rng):
    digit = rng.random() < 0.5
    title = rng.choice(['Create an account', 'Sign up', 'Register'])
    html = f'''<section class="card">
  <form id="signup-form" novalidate>
    <label>Name <input id="signup-name" type="text" autocomplete="name"></label>
    <label>Email <input id="signup-email" type="email" autocomplete="email"></label>
    <label>Password <input id="signup-password" type="password" autocomplete="new-password"></label>
    <label>Confirm password <input id="signup-confirm" type="password" autocomplete="new-password"></label>
    <p id="signup-error" class="error"></p>
    <button type="submit">{title}</button>
  </form>
  <p id="signup-success" hidden></p>
</section>'''
    css = '.error { color: #dc2626; min-height: 1.2em; }'
    rule = ("\n  if (!/[0-9]/.test(password)) { error.textContent = 'Your password needs at least one number.'; return; }" if digit else '')
    js = f'''document.getElementById('signup-form').addEventListener('submit', (event) => {{
  event.preventDefault();
  const name = document.getElementById('signup-name').value.trim();
  const email = document.getElementById('signup-email').value.trim();
  const password = document.getElementById('signup-password').value;
  const confirm = document.getElementById('signup-confirm').value;
  const error = document.getElementById('signup-error');
  if (!name) {{ error.textContent = 'Please enter your name.'; return; }}
  if (!{EMAIL}.test(email)) {{ error.textContent = 'Please enter a valid email address.'; return; }}
  if (password.length < 8) {{ error.textContent = 'Your password needs at least 8 characters.'; return; }}{rule}
  if (password !== confirm) {{ error.textContent = 'The passwords do not match.'; return; }}
  error.textContent = '';
  document.getElementById('signup-form').hidden = true;
  const success = document.getElementById('signup-success');
  success.textContent = 'Welcome, ' + name + '! Your account is ready.';
  success.hidden = false;
}});'''
    note = 'The checks run in the browser only. To really create accounts, send the form to a server that stores a hash of the password.'
    return title, html, css, js, note


def calculator(theme, rng):
    keys = ['7', '8', '9', '/', '4', '5', '6', '*', '1', '2', '3', '-', '0', '.', '=', '+']
    labels = {'/': '÷', '*': '×', '-': '−'}
    buttons = '\n'.join(f'    <button type="button" data-key="{k}">{labels.get(k, k)}</button>' for k in keys)
    html = f'''<section class="card calculator">
  <div id="display">0</div>
  <div id="keys">
    <button type="button" data-key="C" class="secondary wide">C</button>
{buttons}
  </div>
</section>'''
    css = '''#display { font-size: 2rem; text-align: right; padding: 0.5rem; margin-bottom: 0.5rem; background: var(--bg); border-radius: 8px; overflow-x: auto; }
#keys { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; }
#keys button { font-size: 1.2rem; padding: 0.8rem 0; }
#keys .wide { grid-column: span 4; }'''
    js = r'''const display = document.getElementById('display');
let current = '0', previous = null, operator = null, fresh = false;
function calculate(a, b, op) {
  if (op === '+') return a + b;
  if (op === '-') return a - b;
  if (op === '*') return a * b;
  return b === 0 ? NaN : a / b;
}
function finish() {
  const result = calculate(previous, parseFloat(current), operator);
  current = Number.isFinite(result) ? String(parseFloat(result.toFixed(10))) : 'Error';
}
document.getElementById('keys').addEventListener('click', (event) => {
  const key = event.target.dataset.key;
  if (!key) return;
  if (key === 'C') {
    current = '0'; previous = null; operator = null; fresh = false;
  } else if (current === 'Error' && !/[0-9]/.test(key)) {
    return;
  } else if (/[0-9]/.test(key)) {
    current = fresh || current === '0' || current === 'Error' ? key : current + key;
    fresh = false;
  } else if (key === '.') {
    if (fresh) { current = '0'; fresh = false; }
    if (!current.includes('.')) current += '.';
  } else if (key === '=') {
    if (operator === null) return;
    finish();
    previous = null; operator = null; fresh = true;
  } else {
    if (operator !== null && !fresh) finish();
    previous = parseFloat(current); operator = key; fresh = true;
  }
  display.textContent = current;
});'''
    note = 'It does the math itself instead of calling eval(), so nothing typed into the page can run as code.'
    return 'Calculator', html, css, js, note


def countdown(theme, rng):
    minutes = rng.choice([1, 5, 10, 25])
    html = f'''<section class="card">
  <label>Minutes <input id="timer-minutes" type="number" min="0" step="0.5" value="{minutes}"></label>
  <p id="timer-display" class="big">00:00</p>
  <div class="row">
    <button id="timer-start" type="button">Start</button>
    <button id="timer-pause" type="button" class="secondary">Pause</button>
    <button id="timer-reset" type="button" class="secondary">Reset</button>
  </div>
  <p id="timer-message"></p>
</section>'''
    css = '.big { font-size: 3rem; text-align: center; font-variant-numeric: tabular-nums; }'
    js = '''let left = 0, timer = null;
function draw() {
  const minutes = String(Math.floor(left / 60)).padStart(2, '0');
  const seconds = String(left % 60).padStart(2, '0');
  document.getElementById('timer-display').textContent = minutes + ':' + seconds;
}
function stop() {
  clearInterval(timer);
  timer = null;
}
document.getElementById('timer-start').addEventListener('click', () => {
  if (timer) return;
  if (left === 0) left = Math.round(Number(document.getElementById('timer-minutes').value) * 60);
  if (left <= 0) return;
  document.getElementById('timer-message').textContent = '';
  draw();
  timer = setInterval(() => {
    left -= 1;
    draw();
    if (left <= 0) {
      stop();
      document.getElementById('timer-message').textContent = "Time's up!";
    }
  }, 1000);
});
document.getElementById('timer-pause').addEventListener('click', stop);
document.getElementById('timer-reset').addEventListener('click', () => {
  stop();
  left = 0;
  draw();
  document.getElementById('timer-message').textContent = '';
});
draw();'''
    return 'Countdown timer', html, css, js, ''


# thing -> (names it is asked by, builder). A builder returns (title, section html, css, js, closing note) for a
# section placed on a plain page, or (whole page, closing note).
THINGS = {
    'hello': (['hello world page', 'hello world website', 'hello world'], hello),
    'template': (['html template', 'basic html page', 'html boilerplate', 'starter html page', 'blank web page'], template),
    'table': (['html table', 'table', 'data table'], table),
    'profile': (['personal website', 'portfolio page', 'about me page', 'portfolio website'], profile),
    'navbar': (['navbar', 'navigation bar', 'nav menu', 'responsive navbar'], navbar),
    'button': (['button that shows a message', 'click me button', 'button with a click event'], button),
    'login': (['login page', 'login form', 'login screen', 'sign in page', 'log in page'], login),
    'signup': (['signup page', 'sign up form', 'registration form', 'register page', 'create account page'], signup),
    'calculator': (['calculator', 'calculator app', 'simple calculator', 'basic calculator'], calculator),
    'countdown': (['countdown timer', 'timer', 'pomodoro timer', 'countdown'], countdown),
}
STANDALONE = {
    'counter': ['click counter', 'counter', 'counter app'], 'todo': ['to-do list', 'todo list', 'todo app', 'checklist app'],
    'tip': ['tip calculator'], 'converter': ['temperature converter', 'unit converter'], 'quiz': ['quiz', 'quiz game'],
    'stopwatch': ['stopwatch'], 'contact': ['contact form', 'contact page'], 'dark_mode': ['dark mode toggle', 'dark mode button'],
    'faq': ['FAQ section', 'FAQ page', 'accordion'], 'tabs': ['tabs component', 'tabbed page'], 'modal': ['popup', 'modal popup', 'modal'],
    'slideshow': ['image slider', 'slideshow', 'carousel'], 'quotes': ['random quote generator', 'quote generator'],
    'guess': ['number guessing game', 'guess the number game'], 'cart': ['shopping cart', 'add to cart page'],
    'filter_list': ['search filter', 'searchable list', 'live search list'], 'char_count': ['character counter', 'word limit text box'],
}


def standalone(feature):
    def build(theme, rng):
        _, html, css, js = feature(theme, rng)
        return STANDALONE[feature.__name__][0].capitalize(), html, css, js, ''
    return build


for feature in FEATURES:
    THINGS[feature.__name__] = (STANDALONE[feature.__name__], standalone(feature))


def article(thing):
    return ('an ' if thing[0].lower() in 'aeiou' or thing.lower().startswith('html') else 'a ') + thing


def valid(html):
    """make_websites.check for pages with a script; balanced tags and unique ids for plain ones."""
    if '<script>' in html:
        return check(html)
    parser = Structure()
    parser.feed(html)
    return parser.ok and not parser.open


def single_page(theme, title, section, css, js, rng):
    _, _, _, _, (bg, surface, text, accent), _ = theme
    style = BASE_CSS.format(bg=bg, surface=surface, text=text, accent=accent, font=rng.choice(FONTS)) + '\n' + PAGE_CSS + '\n' + css
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{style}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
{section}
</main>
<script>
{js}
</script>
</body>
</html>'''


def example(themes, rng, split):
    theme = rng.choice(themes)
    asks = TRAIN_ASKS if split == 'train' else HELD_OUT_ASKS
    if rng.random() < 0.12:  # a landing page: the business page from make_websites with one feature
        feature = rng.choice([f for f in FEATURES if f.__name__ in ('contact', 'faq', 'modal', 'tabs', 'quotes')])
        _, html = page(theme, [feature], rng)
        thing = rng.choice(['landing page', 'home page', 'homepage']) + f' for {article(theme[1])}'
        features, note = ['landing', feature.__name__], 'Change the name, text and colors to make it yours.'
    else:
        key = rng.choice(list(THINGS))
        names, builder = THINGS[key]
        thing = rng.choice(names)
        built = builder(theme, rng)
        html, note = built if len(built) == 2 else (single_page(theme, *built[:4], rng), built[4])
        features = [key]
    prompt = rng.choice(asks).format(thing=thing, a=article(thing))
    if rng.random() < 0.2:
        prompt = prompt.lower()
    if rng.random() < 0.15:
        prompt = prompt.rstrip('.?')
    reply = rng.choice(INTROS).format(thing=thing, a=article(thing)) + '\n\n```html\n' + html + '\n```'
    outro = note or rng.choice(OUTROS)
    if outro:
        reply += '\n\n' + outro
    if not valid(html) or len(prompt) + len(reply) > MAX_CHARACTERS:
        return None
    return {'split': split, 'family': 'website', 'source': 'make_web_basics', 'language': 'html', 'features': features,
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]}


def build(seed=2431):
    rng = random.Random(seed)
    train_themes = [t for t in THEMES if t[0] not in HELD_OUT_THEMES]
    held_out = [t for t in THEMES if t[0] in HELD_OUT_THEMES]
    rows = []
    for split, themes, wanted in (('train', train_themes, TRAIN_EXAMPLES), ('validation', held_out, VALIDATION_EXAMPLES)):
        made, seen = 0, set()
        while made < wanted:
            row = example(themes, rng, split)
            if row and row['messages'][1]['content'] not in seen:
                seen.add(row['messages'][1]['content'])
                rows.append(row)
                made += 1
    # kelus.website_eval reads the first validation rows: put one of each thing first.
    validation = [r for r in rows if r['split'] == 'validation']
    first, rest, kinds = [], [], set()
    for row in validation:
        (rest if row['features'][0] in kinds else first).append(row)
        kinds.add(row['features'][0])
    return [r for r in rows if r['split'] == 'train'] + first + rest


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/web-basics-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    sizes = sorted(len(r['messages'][0]['content']) + len(r['messages'][1]['content']) for r in rows)
    kinds = len({r['features'][0] for r in rows})
    print(f'{destination}: {len(rows):,} pages ({kinds} kinds); characters median {sizes[len(sizes) // 2]:,}, longest {sizes[-1]:,}')
