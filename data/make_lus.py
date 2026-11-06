"""lus-v1: teaches Cide Kel, the user's own language (.lus files; ~/Downloads/kel, GPL-3.0).

Every program here was compiled and run with the real compiler, and its output compared with the answer worked out in
Python, so the rows cannot teach broken Kel. Kinds of rows:
  - writing a program or a function from a request that names Kel or .lus;
  - converting Python to Kel;
  - fixing a program, given the compiler's real error message;
  - explaining a program, changing it afterwards, or asking for something new in the same chat;
  - questions about the language, including what it does not have yet: no imports, files, networking or null, so Cide
    should say so instead of inventing them.
Kel has lists, its own types with methods, ask, break and continue, text tools and math since version 0.2.0 (2026-11-04).
None of kelus.lus_eval's task kinds is a template here (triangular numbers, divisors, lcm, season, tip, second largest,
a Rectangle or Book type, and so on).

    .venv/bin/python -I data/make_lus.py   # writes data/lus-v1.jsonl
"""
import collections
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus import lus_eval  # noqa: E402

TEMPLATES = []
MAX_COPIES = 5  # most rows with the same program
HELD_OUT = 60


def template(fn):
    TEMPLATES.append(fn)
    return fn


def K(source, /, **values):
    """Fills $names; Kel's own { } stay as they are."""
    return re.sub(r'\$(\w+)', lambda m: str(values[m.group(1)]), source.strip('\n')) + '\n'


def show(value):
    """A number as Kel's `say` prints it."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return f'{value:.1f}' if value == int(value) else repr(value)
    return str(value)


def one(r, *names):
    return r.choice(names)


# ---- Programs. Each returns: task (finishes "a program that ..."), kel, out (what it must print), and sometimes py and sig.
@template
def hello(r):
    text = one(r, 'Hello, world!', 'Hello from Kel!', 'Good morning!', 'Welcome to my program', 'Hi there!', 'Kel is fun')
    return dict(task=f'prints "{text}"', kel=K('fn main {\n    say "$text"\n}', text=text), out=[text], py=f'print("{text}")')


@template
def greet_vars(r):
    name, age, grow = one(r, 'Ana', 'Sam', 'Mina', 'Leo', 'Ravi', 'Jo'), r.randint(8, 40), r.randint(1, 5)
    return dict(task=f'stores the name {name} and the age {age}, adds {grow} to the age, and prints a sentence with both',
                kel=K('fn main {\n    let name = "$name"\n    var age = $age\n    age += $grow\n    say "{name} is {age} years old"\n}', name=name, age=age, grow=grow),
                out=[f'{name} is {age + grow} years old'],
                py=f'name = "{name}"\nage = {age}\nage += {grow}\nprint(f"{{name}} is {{age}} years old")')


@template
def sum_to(r):
    n, total = one(r, 5, 10, 20, 50, 100, 200, 1000), one(r, 'total', 'sum', 'result')
    return dict(task=f'adds the numbers from 1 to {n} and prints the total',
                kel=K('fn main {\n    var $total = 0\n    for i in 1..$n {\n        $total += i\n    }\n    say $total\n}', n=n, total=total), out=[str(n * (n + 1) // 2)],
                py=f'{total} = 0\nfor i in range(1, {n} + 1):\n    {total} += i\nprint({total})')


@template
def factorial(r):
    n, name = r.randint(3, 15), one(r, 'factorial', 'fact')
    result = 1
    for i in range(2, n + 1):
        result *= i
    return dict(task=f'prints the factorial of {n} using a recursive function', sig=f'{name}(n: int) -> int',
                kel=K('fn $name(n: int) -> int {\n    if n <= 1 { return 1 }\n    return n * $name(n - 1)\n}\n\nfn main {\n    say $name($n)\n}', name=name, n=n), out=[str(result)],
                py=f'def {name}(n):\n    if n <= 1:\n        return 1\n    return n * {name}(n - 1)\n\nprint({name}({n}))')


@template
def factorial_loop(r):
    n, value = r.randint(3, 15), one(r, 'result', 'product', 'answer')
    result = 1
    for i in range(2, n + 1):
        result *= i
    return dict(task=f'works out the factorial of {n} with a loop', kel=K('fn main {\n    var $value = 1\n    for i in 2..$n {\n        $value *= i\n    }\n    say "$n! = {$value}"\n}', value=value, n=n), out=[f'{n}! = {result}'])


@template
def fib_list(r):
    n = r.randint(5, 15)
    a = [0, 1]
    while len(a) <= n:
        a.append(a[-1] + a[-2])
    return dict(task=f'prints the Fibonacci numbers from fib(0) to fib({n})', sig='fib(n: int) -> int',
                kel=K('fn fib(n: int) -> int {\n    if n < 2 { return n }\n    return fib(n - 1) + fib(n - 2)\n}\n\nfn main {\n    for i in 0..$n {\n        say "fib({i}) = {fib(i)}"\n    }\n}', n=n),
                out=[f'fib({i}) = {a[i]}' for i in range(n + 1)])


@template
def fib_loop(r):
    n = r.randint(5, 40)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return dict(task=f'finds the Fibonacci number at position {n} with a loop instead of recursion',
                kel=K('fn main {\n    var a = 0\n    var b = 1\n    repeat $n {\n        let next = a + b\n        a = b\n        b = next\n    }\n    say a\n}', n=n), out=[str(a)],
                py=f'a = 0\nb = 1\nfor _ in range({n}):\n    a, b = b, a + b\nprint(a)')


def primes_to(n):
    return [p for p in range(2, n + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


IS_PRIME = 'fn is_prime(n: int) -> bool {\n    if n < 2 { return false }\n    var d = 2\n    while d * d <= n {\n        if n % d == 0 { return false }\n        d += 1\n    }\n    return true\n}\n\n'


@template
def count_primes(r):
    n = one(r, 50, 100, 500, 1000, 5000)
    return dict(task=f'counts the prime numbers up to {n}', sig='is_prime(n: int) -> bool',
                kel=K(IS_PRIME + 'fn main {\n    var count = 0\n    for n in 2..$n {\n        if is_prime(n) { count += 1 }\n    }\n    say "Primes up to $n: {count}"\n}', n=n), out=[f'Primes up to {n}: {len(primes_to(n))}'])


@template
def list_primes(r):
    n = one(r, 20, 30, 50, 70, 100)
    return dict(task=f'prints every prime number up to {n}', sig='is_prime(n: int) -> bool',
                kel=K(IS_PRIME + 'fn main {\n    for n in 2..$n {\n        if is_prime(n) { say n }\n    }\n}', n=n), out=[str(p) for p in primes_to(n)])


@template
def check_prime(r):
    n = one(r, 7, 9, 13, 21, 37, 49, 97, 91, 101)
    return dict(task=f'checks whether {n} is prime', sig='is_prime(n: int) -> bool', kel=K(IS_PRIME + 'fn main {\n    say is_prime($n)\n}', n=n), out=[show(n in primes_to(n))])


@template
def gcd(r):
    import math
    a, b, name = r.randint(6, 200), r.randint(6, 200), one(r, 'gcd', 'greatest_common')
    return dict(task=f'finds the greatest common divisor of {a} and {b}', sig=f'{name}(a: int, b: int) -> int',
                kel=K('fn $name(a: int, b: int) -> int {\n    if b == 0 { return a }\n    return $name(b, a % b)\n}\n\nfn main {\n    say $name($a, $b)\n}', name=name, a=a, b=b), out=[str(math.gcd(a, b))],
                py=f'def {name}(a, b):\n    if b == 0:\n        return a\n    return {name}(b, a % b)\n\nprint({name}({a}, {b}))')


@template
def even_odd(r):
    n, name = r.randint(4, 12), one(r, 'is_even', 'even')
    return dict(task=f'says whether each number from 1 to {n} is even or odd', sig=f'{name}(n: int) -> bool',
                kel=K('fn $name(n: int) -> bool {\n    return n % 2 == 0\n}\n\nfn main {\n    for i in 1..$n {\n        if $name(i) {\n            say "{i} is even"\n        } else {\n            say "{i} is odd"\n        }\n    }\n}', name=name, n=n),
                out=[f'{i} is {"even" if i % 2 == 0 else "odd"}' for i in range(1, n + 1)])


@template
def fizzbuzz(r):
    n = one(r, 15, 20, 30, 50)
    a, b, wa, wb = one(r, (3, 5, 'Fizz', 'Buzz'), (3, 5, 'Fizz', 'Buzz'), (2, 7, 'Foo', 'Bar'), (4, 6, 'Ping', 'Pong'))
    out = []
    for i in range(1, n + 1):
        out.append(wa + wb if i % a == 0 and i % b == 0 else wa if i % a == 0 else wb if i % b == 0 else str(i))
    return dict(task=f'prints the numbers from 1 to {n}, but {wa} for multiples of {a}, {wb} for multiples of {b}, and {wa}{wb} for both',
                kel=K('fn main {\n    for i in 1..$n {\n        if i % $a == 0 && i % $b == 0 {\n            say "$wa$wb"\n        } else if i % $a == 0 {\n            say "$wa"\n        } else if i % $b == 0 {\n            say "$wb"\n        } else {\n            say i\n        }\n    }\n}', n=n, a=a, b=b, wa=wa, wb=wb), out=out,
                py=f'for i in range(1, {n} + 1):\n    if i % {a} == 0 and i % {b} == 0:\n        print("{wa}{wb}")\n    elif i % {a} == 0:\n        print("{wa}")\n    elif i % {b} == 0:\n        print("{wb}")\n    else:\n        print(i)')


@template
def countdown(r):
    n, end = r.randint(3, 20), one(r, 'Liftoff!', 'Go!', 'Happy New Year!', 'Time is up')
    return dict(task=f'counts down from {n} to 1 and then prints "{end}"',
                kel=K('fn main {\n    var n = $n\n    while n > 0 {\n        say n\n        n -= 1\n    }\n    say "$end"\n}', n=n, end=end), out=[str(i) for i in range(n, 0, -1)] + [end],
                py=f'n = {n}\nwhile n > 0:\n    print(n)\n    n -= 1\nprint("{end}")')


@template
def times_table(r):
    k, m = r.randint(2, 15), one(r, 5, 10, 12)
    return dict(task=f'prints the {k} times table up to {k} x {m}', kel=K('fn main {\n    for i in 1..$m {\n        say "$k x {i} = {$k * i}"\n    }\n}', k=k, m=m), out=[f'{k} x {i} = {k * i}' for i in range(1, m + 1)],
                py=f'for i in range(1, {m} + 1):\n    print(f"{k} x {{i}} = {{{k} * i}}")')


@template
def power(r):
    b, e, name = r.randint(2, 9), r.randint(2, 12), one(r, 'power', 'pow', 'raise_to')
    return dict(task=f'works out {b} to the power of {e} with a loop', sig=f'{name}(base: int, exp: int) -> int',
                kel=K('fn $name(base: int, exp: int) -> int {\n    var result = 1\n    repeat exp {\n        result *= base\n    }\n    return result\n}\n\nfn main {\n    say $name($b, $e)\n}', name=name, b=b, e=e), out=[str(b ** e)],
                py=f'def {name}(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result\n\nprint({name}({b}, {e}))')


@template
def digit_sum(r):
    n, name = r.randint(100, 999999), one(r, 'digit_sum', 'sum_digits', 'add_digits')
    return dict(task=f'adds up the digits of {n}', sig=f'{name}(n: int) -> int',
                kel=K('fn $name(n: int) -> int {\n    var left = n\n    var total = 0\n    while left > 0 {\n        total += left % 10\n        left /= 10\n    }\n    return total\n}\n\nfn main {\n    say $name($n)\n}', name=name, n=n), out=[str(sum(map(int, str(n))))],
                py=f'def {name}(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total\n\nprint({name}({n}))')


@template
def count_digits(r):
    n = r.randint(1, 99999999)
    return dict(task=f'counts how many digits {n} has', sig='count_digits(n: int) -> int',
                kel=K('fn count_digits(n: int) -> int {\n    var left = n\n    var count = 1\n    while left >= 10 {\n        left /= 10\n        count += 1\n    }\n    return count\n}\n\nfn main {\n    say count_digits($n)\n}', n=n), out=[str(len(str(n)))])


REVERSE = 'fn reverse(n: int) -> int {\n    var left = n\n    var result = 0\n    while left > 0 {\n        result = result * 10 + left % 10\n        left /= 10\n    }\n    return result\n}\n\n'


@template
def reverse_number(r):
    n = r.randint(10, 999999)
    return dict(task=f'reverses the digits of {n}', sig='reverse(n: int) -> int', kel=K(REVERSE + 'fn main {\n    say reverse($n)\n}', n=n), out=[str(int(str(n)[::-1]))])


@template
def palindrome_number(r):
    n = one(r, 121, 123, 1221, 1234, 12321, 45654, 100, 7)
    return dict(task=f'checks whether the number {n} reads the same backwards', sig='is_palindrome(n: int) -> bool',
                kel=K(REVERSE + 'fn is_palindrome(n: int) -> bool {\n    return reverse(n) == n\n}\n\nfn main {\n    say is_palindrome($n)\n}', n=n), out=[show(str(n) == str(n)[::-1])])


@template
def collatz(r):
    n = r.randint(3, 200)
    steps, x = 0, n
    while x != 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        steps += 1
    return dict(task=f'counts the Collatz steps it takes {n} to reach 1',
                kel=K('fn main {\n    var n = $n\n    var steps = 0\n    while n != 1 {\n        if n % 2 == 0 {\n            n /= 2\n        } else {\n            n = 3 * n + 1\n        }\n        steps += 1\n    }\n    say steps\n}', n=n), out=[str(steps)])


@template
def max_two(r):
    a, b, name = r.randint(-20, 90), r.randint(-20, 90), one(r, 'max', 'bigger', 'larger')
    return dict(task=f'prints the larger of {a} and {b}', sig=f'{name}(a: int, b: int) -> int',
                kel=K('fn $name(a: int, b: int) -> int {\n    if a > b { return a }\n    return b\n}\n\nfn main {\n    say $name($a, $b)\n}', name=name, a=a, b=b), out=[str(max(a, b))],
                py=f'def {name}(a, b):\n    if a > b:\n        return a\n    return b\n\nprint({name}({a}, {b}))')


@template
def min_three(r):
    a, b, c, name = r.randint(1, 90), r.randint(1, 90), r.randint(1, 90), one(r, 'smallest', 'min3')
    return dict(task=f'prints the smallest of {a}, {b} and {c}', sig=f'{name}(a: int, b: int, c: int) -> int',
                kel=K('fn $name(a: int, b: int, c: int) -> int {\n    var low = a\n    if b < low { low = b }\n    if c < low { low = c }\n    return low\n}\n\nfn main {\n    say $name($a, $b, $c)\n}', name=name, a=a, b=b, c=c), out=[str(min(a, b, c))])


@template
def absolute(r):
    n, name = r.randint(-90, 90), one(r, 'abs', 'absolute')
    return dict(task=f'prints the absolute value of {n}', sig=f'{name}(n: int) -> int',
                kel=K('fn $name(n: int) -> int {\n    if n < 0 { return -n }\n    return n\n}\n\nfn main {\n    say $name($n)\n}', name=name, n=n), out=[str(abs(n))])


@template
def clamp(r):
    n, low, high = r.randint(-20, 150), r.randint(0, 20), r.randint(50, 100)
    return dict(task=f'keeps {n} between {low} and {high}', sig='clamp(n: int, low: int, high: int) -> int',
                kel=K('fn clamp(n: int, low: int, high: int) -> int {\n    if n < low { return low }\n    if n > high { return high }\n    return n\n}\n\nfn main {\n    say clamp($n, $low, $high)\n}', n=n, low=low, high=high), out=[str(max(low, min(n, high)))])


@template
def c_to_f(r):
    c = float(r.randint(-10, 40))
    return dict(task=f'converts {c} degrees Celsius to Fahrenheit', sig='to_fahrenheit(c: float) -> float',
                kel=K('fn to_fahrenheit(c: float) -> float {\n    return c * 9.0 / 5.0 + 32.0\n}\n\nfn main {\n    say to_fahrenheit($c)\n}', c=c), out=[show(c * 9.0 / 5.0 + 32.0)],
                py=f'def to_fahrenheit(c):\n    return c * 9.0 / 5.0 + 32.0\n\nprint(to_fahrenheit({c}))')


@template
def average_three(r):
    a, b, c = r.randint(50, 100), r.randint(50, 100), r.randint(50, 100)
    return dict(task=f'prints the average of {a}, {b} and {c} as a float',
                kel=K('fn main {\n    let total = $a + $b + $c\n    let average = float(total) / 3.0\n    say "Average: {average}"\n}', a=a, b=b, c=c), out=[f'Average: {show((a + b + c) / 3.0)}'],
                py=f'total = {a} + {b} + {c}\naverage = total / 3.0\nprint(f"Average: {{average}}")')


@template
def grade(r):
    score = float(r.randint(40, 100))
    letter = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'F'
    return dict(task=f'turns a score of {score} into a letter grade from A to F', sig='grade(score: float) -> str',
                kel=K('fn grade(score: float) -> str {\n    if score >= 90.0 {\n        return "A"\n    } else if score >= 80.0 {\n        return "B"\n    } else if score >= 70.0 {\n        return "C"\n    } else if score >= 60.0 {\n        return "D"\n    }\n    return "F"\n}\n\nfn main {\n    say "Grade: " + grade($score)\n}', score=score), out=[f'Grade: {letter}'])


@template
def leap(r):
    y = one(r, 1900, 2000, 2023, 2024, 2100, 2400, 2028, 1996)
    return dict(task=f'checks whether {y} is a leap year', sig='is_leap(year: int) -> bool',
                kel=K('fn is_leap(year: int) -> bool {\n    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)\n}\n\nfn main {\n    say is_leap($y)\n}', y=y), out=[show(y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))],
                py=f'def is_leap(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n\nprint(is_leap({y}))')


@template
def triangle(r):
    n, ch = r.randint(3, 8), one(r, '*', '#', '+', 'o')
    return dict(task=f'prints a triangle of {ch} with {n} rows',
                kel=K('fn main {\n    for row in 1..$n {\n        var line = ""\n        repeat row {\n            line += "$ch"\n        }\n        say line\n    }\n}', n=n, ch=ch), out=[ch * i for i in range(1, n + 1)])


@template
def square(r):
    n, ch = r.randint(2, 7), one(r, '*', '#', 'x', '@')
    return dict(task=f'prints a {n} by {n} square of {ch}', kel=K('fn main {\n    repeat $n {\n        var line = ""\n        repeat $n {\n            line += "$ch"\n        }\n        say line\n    }\n}', n=n, ch=ch), out=[ch * n] * n)


@template
def repeat_text(r):
    n, word, name = r.randint(2, 9), one(r, 'ha', 'la', 'na', 'bye', 'go'), one(r, 'repeat_text', 'times', 'copies')
    return dict(task=f'builds a text with "{word}" repeated {n} times', sig=f'{name}(word: str, n: int) -> str',
                kel=K('fn $name(word: str, n: int) -> str {\n    var result = ""\n    repeat n {\n        result += word\n    }\n    return result\n}\n\nfn main {\n    say $name("$word", $n)\n}', name=name, word=word, n=n), out=[word * n])


@template
def say_many(r):
    n, text = r.randint(2, 8), one(r, 'Hip hip hooray!', 'I love coding', 'Never give up', 'Kel rocks')
    return dict(task=f'prints "{text}" {n} times', kel=K('fn main {\n    repeat $n {\n        say "$text"\n    }\n}', n=n, text=text), out=[text] * n, py=f'for _ in range({n}):\n    print("{text}")')


@template
def sum_evens(r):
    n = one(r, 10, 20, 50, 100, 250)
    return dict(task=f'adds up the even numbers from 1 to {n}', kel=K('fn main {\n    var total = 0\n    for i in 1..$n {\n        if i % 2 == 0 {\n            total += i\n        }\n    }\n    say total\n}', n=n), out=[str(sum(i for i in range(1, n + 1) if i % 2 == 0))],
                py=f'total = 0\nfor i in range(1, {n} + 1):\n    if i % 2 == 0:\n        total += i\nprint(total)')


@template
def count_multiples(r):
    n, k = one(r, 50, 100, 500, 1000), r.randint(2, 13)
    return dict(task=f'counts the multiples of {k} from 1 to {n}', kel=K('fn main {\n    var count = 0\n    for i in 1..$n {\n        if i % $k == 0 {\n            count += 1\n        }\n    }\n    say "{count} multiples of $k"\n}', n=n, k=k), out=[f'{n // k} multiples of {k}'])


@template
def compound(r):
    start, years = float(one(r, 100, 500, 1000, 2500)), r.randint(2, 10)
    rate = one(r, 1.02, 1.05, 1.1)
    value = start
    for _ in range(years):
        value = value * rate
    return dict(task=f'grows {start} by a factor of {rate} every year for {years} years', kel=K('fn main {\n    var money = $start\n    repeat $years {\n        money = money * $rate\n    }\n    say money\n}', start=start, years=years, rate=rate), out=[show(value)])


@template
def rectangle(r):
    w, h = r.randint(2, 30), r.randint(2, 30)
    return dict(task=f'prints the area and the perimeter of a rectangle {w} wide and {h} high',
                kel=K('fn area(w: int, h: int) -> int {\n    return w * h\n}\n\nfn perimeter(w: int, h: int) -> int {\n    return 2 * (w + h)\n}\n\nfn main {\n    say "Area: {area($w, $h)}"\n    say "Perimeter: {perimeter($w, $h)}"\n}', w=w, h=h), out=[f'Area: {w * h}', f'Perimeter: {2 * (w + h)}'],
                py=f'def area(w, h):\n    return w * h\n\ndef perimeter(w, h):\n    return 2 * (w + h)\n\nprint(f"Area: {{area({w}, {h})}}")\nprint(f"Perimeter: {{perimeter({w}, {h})}}")')


@template
def circle(r):
    radius = float(r.randint(1, 20))
    return dict(task=f'works out the area of a circle with radius {radius}, using 3.14 for pi', sig='circle_area(radius: float) -> float',
                kel=K('fn circle_area(radius: float) -> float {\n    let pi = 3.14\n    return pi * radius * radius\n}\n\nfn main {\n    say circle_area($radius)\n}', radius=radius), out=[show(3.14 * radius * radius)])


@template
def shopping(r):
    price, count, tax = one(r, 2.5, 4.0, 9.5, 12.0), r.randint(2, 9), one(r, 0.05, 0.1, 0.2)
    subtotal = price * float(count)
    return dict(task=f'works out the total for {count} items at {price} each with {tax} tax', kel=K('fn main {\n    let count = $count\n    let subtotal = $price * float(count)\n    let total = subtotal + subtotal * $tax\n    say "Total: {total}"\n}', count=count, price=price, tax=tax), out=[f'Total: {show(subtotal + subtotal * tax)}'])


@template
def bmi(r):
    weight, height = float(r.randint(45, 110)), one(r, 1.5, 1.6, 1.7, 1.8, 1.9)
    return dict(task=f'works out the body mass index for {weight} kg and {height} m', sig='bmi(weight: float, height: float) -> float', kel=K('fn bmi(weight: float, height: float) -> float {\n    return weight / (height * height)\n}\n\nfn main {\n    say bmi($weight, $height)\n}', weight=weight, height=height), out=[show(weight / (height * height))])


@template
def discount(r):
    price, percent = float(r.randint(10, 300)), r.randint(5, 60)
    return dict(task=f'takes {percent} percent off a price of {price}', sig='discounted(price: float, percent: int) -> float',
                kel=K('fn discounted(price: float, percent: int) -> float {\n    return price - price * float(percent) / 100.0\n}\n\nfn main {\n    say discounted($price, $percent)\n}', price=price, percent=percent), out=[show(price - price * float(percent) / 100.0)])


@template
def weather(r):
    t = r.randint(-10, 40)
    word = 'hot' if t > 30 else 'warm' if t > 20 else 'cool' if t > 10 else 'cold'
    return dict(task=f'prints hot, warm, cool or cold for a temperature of {t}', kel=K('fn main {\n    let temp = $t\n    if temp > 30 {\n        say "hot"\n    } else if temp > 20 {\n        say "warm"\n    } else if temp > 10 {\n        say "cool"\n    } else {\n        say "cold"\n    }\n}', t=t), out=[word],
                py=f'temp = {t}\nif temp > 30:\n    print("hot")\nelif temp > 20:\n    print("warm")\nelif temp > 10:\n    print("cool")\nelse:\n    print("cold")')


@template
def adult(r):
    age = r.randint(5, 40)
    return dict(task=f'checks whether someone aged {age} is an adult', sig='is_adult(age: int) -> bool', kel=K('fn is_adult(age: int) -> bool {\n    return age >= 18\n}\n\nfn main {\n    say is_adult($age)\n}', age=age), out=[show(age >= 18)])


@template
def word_length(r):
    word = one(r, 'Kel', 'hello', 'programming', 'WebAssembly', 'otter', 'banana')
    return dict(task=f'prints how many letters "{word}" has', kel=K('fn main {\n    let word = "$word"\n    say "{word} has {len(word)} letters"\n}', word=word), out=[f'{word} has {len(word)} letters'])


@template
def shout(r):
    word = one(r, 'hello', 'stop', 'kel', 'wow', 'yes')
    return dict(task=f'adds an exclamation mark to "{word}" with a function', sig='shout(word: str) -> str', kel=K('fn shout(word: str) -> str {\n    return word + "!"\n}\n\nfn main {\n    say shout("$word")\n}', word=word), out=[word + '!'])


@template
def full_name(r):
    a, b = one(r, 'Ana', 'Sam', 'Mina', 'Leo'), one(r, 'Kim', 'Park', 'Lee', 'Silva')
    return dict(task=f'joins {a} and {b} into a full name', sig='full_name(first: str, last: str) -> str', kel=K('fn full_name(first: str, last: str) -> str {\n    return first + " " + last\n}\n\nfn main {\n    say full_name("$a", "$b")\n}', a=a, b=b), out=[f'{a} {b}'])


@template
def password(r):
    secret, tried = one(r, 'otter', 'kel123', 'swordfish'), one(r, 'otter', 'kel123', 'swordfish', 'guess')
    return dict(task=f'checks a stored guess "{tried}" against the password "{secret}"', kel=K('fn main {\n    let password = "$secret"\n    let guess = "$tried"\n    if guess == password {\n        say "Welcome!"\n    } else {\n        say "Wrong password"\n    }\n}', secret=secret, tried=tried), out=['Welcome!' if secret == tried else 'Wrong password'])


@template
def minutes(r):
    total = r.randint(61, 900)
    return dict(task=f'splits {total} minutes into hours and minutes', kel=K('fn main {\n    let total = $total\n    let hours = total / 60\n    let minutes = total % 60\n    say "{hours} hours and {minutes} minutes"\n}', total=total), out=[f'{total // 60} hours and {total % 60} minutes'],
                py=f'total = {total}\nhours = total // 60\nminutes = total % 60\nprint(f"{{hours}} hours and {{minutes}} minutes")')


@template
def coins(r):
    cents = r.randint(30, 299)
    q, d, n, p = cents // 25, cents % 25 // 10, cents % 25 % 10 // 5, cents % 25 % 10 % 5
    return dict(task=f'breaks {cents} cents into quarters, dimes, nickels and pennies', kel=K('fn main {\n    var left = $cents\n    let quarters = left / 25\n    left %= 25\n    let dimes = left / 10\n    left %= 10\n    let nickels = left / 5\n    left %= 5\n    say "{quarters} quarters, {dimes} dimes, {nickels} nickels, {left} pennies"\n}', cents=cents), out=[f'{q} quarters, {d} dimes, {n} nickels, {p} pennies'])


@template
def savings(r):
    goal, week = one(r, 100, 250, 500, 1000), r.randint(5, 60)
    weeks = 0
    saved = 0
    while saved < goal:
        saved += week
        weeks += 1
    return dict(task=f'counts how many weeks it takes to save {goal} at {week} a week', kel=K('fn main {\n    var saved = 0\n    var weeks = 0\n    while saved < $goal {\n        saved += $week\n        weeks += 1\n    }\n    say "It takes {weeks} weeks"\n}', goal=goal, week=week), out=[f'It takes {weeks} weeks'])


@template
def bottles(r):
    n = r.randint(3, 10)
    return dict(task=f'sings the bottles song starting from {n} bottles', kel=K('fn main {\n    for i in 0..$n - 1 {\n        let left = $n - i\n        say "{left} bottles on the wall, take one down"\n    }\n    say "No more bottles"\n}', n=n), out=[f'{n - i} bottles on the wall, take one down' for i in range(n)] + ['No more bottles'])


@template
def sum_range(r):
    a, b, name = r.randint(1, 20), r.randint(21, 80), one(r, 'sum_range', 'add_between')
    return dict(task=f'adds the numbers from {a} to {b}', sig=f'{name}(first: int, last: int) -> int',
                kel=K('fn $name(first: int, last: int) -> int {\n    var total = 0\n    for i in first..last {\n        total += i\n    }\n    return total\n}\n\nfn main {\n    say $name($a, $b)\n}', name=name, a=a, b=b), out=[str(sum(range(a, b + 1)))],
                py=f'def {name}(first, last):\n    total = 0\n    for i in range(first, last + 1):\n        total += i\n    return total\n\nprint({name}({a}, {b}))')


@template
def step_multiples(r):
    k, limit = r.randint(2, 12), one(r, 30, 50, 100)
    return dict(task=f'prints the multiples of {k} up to {limit} using a while loop', kel=K('fn main {\n    var n = $k\n    while n <= $limit {\n        say n\n        n += $k\n    }\n}', k=k, limit=limit), out=[str(i) for i in range(k, limit + 1, k)])


@template
def count_even_odd(r):
    n = one(r, 9, 20, 25, 33, 100)
    return dict(task=f'counts the even and the odd numbers from 1 to {n}', kel=K('fn main {\n    var evens = 0\n    var odds = 0\n    for i in 1..$n {\n        if i % 2 == 0 {\n            evens += 1\n        } else {\n            odds += 1\n        }\n    }\n    say "{evens} even, {odds} odd"\n}', n=n), out=[f'{n // 2} even, {n - n // 2} odd'])


@template
def highest_power(r):
    n = r.randint(10, 5000)
    p = 1
    while p * 2 <= n:
        p *= 2
    return dict(task=f'finds the biggest power of 2 that is not above {n}', kel=K('fn main {\n    var power = 1\n    while power * 2 <= $n {\n        power *= 2\n    }\n    say power\n}', n=n), out=[str(p)])


@template
def int_sqrt(r):
    n = r.randint(10, 5000)
    s = 0
    while (s + 1) * (s + 1) <= n:
        s += 1
    return dict(task=f'finds the whole-number square root of {n}, rounded down', sig='root(n: int) -> int', kel=K('fn root(n: int) -> int {\n    var r = 0\n    while (r + 1) * (r + 1) <= n {\n        r += 1\n    }\n    return r\n}\n\nfn main {\n    say root($n)\n}', n=n), out=[str(s)])


@template
def doubling(r):
    start, limit = r.randint(1, 9), one(r, 100, 500, 1000, 10000)
    x, count = start, 0
    while x < limit:
        x *= 2
        count += 1
    return dict(task=f'doubles {start} until it reaches {limit} and counts the steps', kel=K('fn main {\n    var value = $start\n    var steps = 0\n    while value < $limit {\n        value *= 2\n        steps += 1\n    }\n    say "{value} after {steps} steps"\n}', start=start, limit=limit), out=[f'{x} after {count} steps'])


@template
def bank(r):
    start, d, w = r.randint(50, 500), r.randint(10, 200), r.randint(10, 400)
    final = start + d if start + d < w else start + d - w
    return dict(task=f'starts a balance at {start}, deposits {d}, then withdraws {w} only if there is enough money', sig='withdraw(balance: int, amount: int) -> int',
                kel=K('fn deposit(balance: int, amount: int) -> int {\n    return balance + amount\n}\n\nfn withdraw(balance: int, amount: int) -> int {\n    if amount > balance {\n        say "Not enough money"\n        return balance\n    }\n    return balance - amount\n}\n\nfn main {\n    var balance = $start\n    balance = deposit(balance, $d)\n    balance = withdraw(balance, $w)\n    say "Balance: {balance}"\n}', start=start, d=d, w=w),
                out=(['Not enough money'] if start + d < w else []) + [f'Balance: {final}'])


@template
def traffic(r):
    color = one(r, 'red', 'yellow', 'green', 'blue')
    word = {'red': 'Stop', 'yellow': 'Slow down', 'green': 'Go'}.get(color, 'Unknown color')
    return dict(task=f'prints what to do at a {color} traffic light', sig='action(color: str) -> str', kel=K('fn action(color: str) -> str {\n    if color == "red" {\n        return "Stop"\n    } else if color == "yellow" {\n        return "Slow down"\n    } else if color == "green" {\n        return "Go"\n    }\n    return "Unknown color"\n}\n\nfn main {\n    say action("$color")\n}', color=color), out=[word])


DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


@template
def day_name(r):
    n = r.randint(1, 9)
    body = '\n'.join(f'    if n == {i + 1} {{ return "{d}" }}' for i, d in enumerate(DAYS))
    return dict(task=f'prints the day name for day number {n} of the week', sig='day_name(n: int) -> str', kel=K('fn day_name(n: int) -> str {\n' + body + '\n    return "Not a day"\n}\n\nfn main {\n    say day_name($n)\n}', n=n), out=[DAYS[n - 1] if n <= 7 else 'Not a day'])


@template
def ticket(r):
    age = r.randint(3, 80)
    price = 5 if age < 12 else 8 if age >= 65 else 12
    return dict(task=f'prints the ticket price for someone aged {age}: 5 for children under 12, 8 for people 65 or older, 12 for everyone else', sig='ticket_price(age: int) -> int', kel=K('fn ticket_price(age: int) -> int {\n    if age < 12 { return 5 }\n    if age >= 65 { return 8 }\n    return 12\n}\n\nfn main {\n    say ticket_price($age)\n}', age=age), out=[str(price)])


@template
def divisible(r):
    a, b = r.randint(10, 200), r.randint(2, 12)
    return dict(task=f'checks whether {a} can be divided evenly by {b}', sig='divides(a: int, b: int) -> bool', kel=K('fn divides(a: int, b: int) -> bool {\n    return a % b == 0\n}\n\nfn main {\n    say divides($a, $b)\n}', a=a, b=b), out=[show(a % b == 0)])


@template
def cubes(r):
    n = r.randint(3, 12)
    return dict(task=f'adds up the cubes of the numbers from 1 to {n}', kel=K('fn main {\n    var total = 0\n    for i in 1..$n {\n        total += i * i * i\n    }\n    say total\n}', n=n), out=[str(sum(i ** 3 for i in range(1, n + 1)))])


@template
def swap(r):
    a, b = r.randint(1, 99), r.randint(1, 99)
    return dict(task=f'swaps two variables holding {a} and {b} and prints them', kel=K('fn main {\n    var a = $a\n    var b = $b\n    let temp = a\n    a = b\n    b = temp\n    say "a = {a}, b = {b}"\n}', a=a, b=b), out=[f'a = {b}, b = {a}'])


@template
def pass_fail(r):
    score, mark = r.randint(20, 100), one(r, 50, 60, 70)
    return dict(task=f'prints a score of {score} and whether it passes a mark of {mark}', kel=K('fn main {\n    let score = $score\n    let passed = score >= $mark\n    say "Score: {score}, passed: {passed}"\n}', score=score, mark=mark), out=[f'Score: {score}, passed: {show(score >= mark)}'])


@template
def even_list(r):
    n = one(r, 10, 12, 20, 30)
    return dict(task=f'prints only the even numbers up to {n}', kel=K('fn main {\n    for i in 1..$n {\n        if i % 2 == 0 {\n            say i\n        }\n    }\n}', n=n), out=[str(i) for i in range(2, n + 1, 2)],
                py=f'for i in range(1, {n} + 1):\n    if i % 2 == 0:\n        print(i)')



# ---- Lists
def numbers(r, low=1, high=99, size=None):
    return [r.randint(low, high) for _ in range(size or r.randint(3, 7))]


def kel_list(items):
    return '[' + ', '.join(f'"{i}"' if isinstance(i, str) else show(i) for i in items) + ']'


@template
def list_total(r):
    items, name = numbers(r), one(r, 'total', 'sum_all', 'add_up')
    return dict(task=f'adds up the numbers in the list {kel_list(items)}', sig=f'{name}(numbers: [int]) -> int',
                kel=K('fn $name(numbers: [int]) -> int {\n    var sum = 0\n    for n in numbers {\n        sum += n\n    }\n    return sum\n}\n\nfn main {\n    let items = $items\n    say $name(items)\n}', name=name, items=kel_list(items)), out=[str(sum(items))],
                py=f'def {name}(numbers):\n    sum = 0\n    for n in numbers:\n        sum += n\n    return sum\n\nitems = {kel_list(items)}\nprint({name}(items))')


@template
def list_largest(r):
    items = numbers(r)
    return dict(task=f'finds the largest number in the list {kel_list(items)}', sig='largest(numbers: [int]) -> int',
                kel=K('fn largest(numbers: [int]) -> int {\n    var best = numbers[0]\n    for n in numbers {\n        if n > best { best = n }\n    }\n    return best\n}\n\nfn main {\n    say largest($items)\n}', items=kel_list(items)), out=[str(max(items))],
                py=f'def largest(numbers):\n    best = numbers[0]\n    for n in numbers:\n        if n > best:\n            best = n\n    return best\n\nprint(largest({kel_list(items)}))')


@template
def list_smallest(r):
    items = numbers(r)
    return dict(task=f'finds the smallest number in {kel_list(items)}', kel=K('fn main {\n    let items = $items\n    var low = items[0]\n    for n in items {\n        low = min(low, n)\n    }\n    say "Smallest: {low}"\n}', items=kel_list(items)), out=[f'Smallest: {min(items)}'])


@template
def list_average(r):
    items = numbers(r)
    return dict(task=f'prints the average of the numbers in {kel_list(items)}', sig='average(numbers: [int]) -> float',
                kel=K('fn average(numbers: [int]) -> float {\n    var total = 0\n    for n in numbers {\n        total += n\n    }\n    return float(total) / float(len(numbers))\n}\n\nfn main {\n    say average($items)\n}', items=kel_list(items)), out=[show(sum(items) / len(items))])


@template
def list_count(r):
    items = numbers(r, -20, 20, 6)
    return dict(task=f'counts how many numbers in {kel_list(items)} are negative', kel=K('fn main {\n    var count = 0\n    for n in $items {\n        if n < 0 {\n            count += 1\n        }\n    }\n    say count\n}', items=kel_list(items)), out=[str(sum(i < 0 for i in items))])


@template
def list_filter(r):
    items = numbers(r, -20, 20, 6)
    kept = [i for i in items if i > 0]
    return dict(task=f'makes a new list with only the positive numbers from {kel_list(items)}', sig='positives(numbers: [int]) -> [int]',
                kel=K('fn positives(numbers: [int]) -> [int] {\n    var result: [int] = []\n    for n in numbers {\n        if n > 0 {\n            result.add(n)\n        }\n    }\n    return result\n}\n\nfn main {\n    say positives($items)\n}', items=kel_list(items)), out=[kel_list(kept)],
                py=f'def positives(numbers):\n    result = []\n    for n in numbers:\n        if n > 0:\n            result.append(n)\n    return result\n\nprint(positives({kel_list(items)}))')


@template
def list_doubled(r):
    items = numbers(r, 1, 30)
    return dict(task=f'makes a new list with every number in {kel_list(items)} doubled', kel=K('fn main {\n    let items = $items\n    var doubled: [int] = []\n    for n in items {\n        doubled.add(n * 2)\n    }\n    say doubled\n}', items=kel_list(items)), out=[kel_list([i * 2 for i in items])],
                py=f'items = {kel_list(items)}\ndoubled = []\nfor n in items:\n    doubled.append(n * 2)\nprint(doubled)')


@template
def list_sorted(r):
    items = numbers(r)
    return dict(task=f'sorts the list {kel_list(items)} from smallest to largest', kel=K('fn main {\n    var items = $items\n    items.sort()\n    say items\n}', items=kel_list(items)), out=[kel_list(sorted(items))], py=f'items = {kel_list(items)}\nitems.sort()\nprint(items)')


@template
def list_sorted_down(r):
    items = numbers(r)
    return dict(task=f'sorts {kel_list(items)} from largest to smallest', kel=K('fn main {\n    var items = $items\n    items.sort()\n    items.reverse()\n    say items\n}', items=kel_list(items)), out=[kel_list(sorted(items, reverse=True))])


@template
def list_has(r):
    items, wanted = numbers(r, 1, 12), r.randint(1, 12)
    return dict(task=f'checks whether {kel_list(items)} contains {wanted}', kel=K('fn main {\n    let items = $items\n    say items.contains($wanted)\n}', items=kel_list(items), wanted=wanted), out=[show(wanted in items)])


@template
def list_position(r):
    items, wanted = numbers(r, 1, 12), r.randint(1, 12)
    return dict(task=f'finds the position of {wanted} in {kel_list(items)}, or -1 if it is not there', kel=K('fn main {\n    let items = $items\n    say items.index_of($wanted)\n}', items=kel_list(items), wanted=wanted), out=[str(items.index(wanted) if wanted in items else -1)])


NAMES = ['Ana', 'Bo', 'Mina', 'Leo', 'Ravi', 'Jo', 'Kim', 'Sam', 'Noor', 'Eli']


@template
def list_greet(r):
    who = r.sample(NAMES, r.randint(2, 5))
    return dict(task=f'says hello to each name in {kel_list(who)}', kel=K('fn main {\n    let names = $who\n    for name in names {\n        say "Hello, {name}!"\n    }\n}', who=kel_list(who)), out=[f'Hello, {n}!' for n in who],
                py=f'names = {kel_list(who)}\nfor name in names:\n    print(f"Hello, {{name}}!")'.replace('"Ana"', '"Ana"'))


@template
def list_join(r):
    who, mark = r.sample(NAMES, 3), one(r, ', ', ' - ', ' and ')
    return dict(task=f'joins {kel_list(who)} into one text with "{mark}" between them', kel=K('fn main {\n    let names = $who\n    say names.join("$mark")\n}', who=kel_list(who), mark=mark), out=[mark.join(who)])


@template
def list_stack(r):
    a, b, c = numbers(r, 1, 50, 3)
    return dict(task=f'adds {a}, {b} and {c} to an empty list, then takes the last one off and prints both', kel=K('fn main {\n    var stack: [int] = []\n    stack.add($a)\n    stack.add($b)\n    stack.add($c)\n    let last = stack.pop()\n    say last\n    say stack\n}', a=a, b=b, c=c), out=[str(c), kel_list([a, b])])


@template
def list_remove_at(r):
    items = numbers(r, 1, 50, 5)
    at = r.randint(0, 4)
    kept = items[:at] + items[at + 1:]
    return dict(task=f'removes the item at position {at} from {kel_list(items)}', kel=K('fn main {\n    var items = $items\n    items.remove($at)\n    say items\n}', items=kel_list(items), at=at), out=[kel_list(kept)])


@template
def list_set_item(r):
    items = numbers(r, 1, 50, 4)
    at, value = r.randint(0, 3), r.randint(100, 200)
    changed = list(items)
    changed[at] = value
    return dict(task=f'changes item {at} of {kel_list(items)} to {value}', kel=K('fn main {\n    var items = $items\n    items[$at] = $value\n    say items\n}', items=kel_list(items), at=at, value=value), out=[kel_list(changed)])


@template
def list_ends(r):
    items = numbers(r, 1, 99)
    return dict(task=f'prints the first and last items of {kel_list(items)} and how many there are', kel=K('fn main {\n    let items = $items\n    say "First: {items[0]}"\n    say "Last: {items[len(items) - 1]}"\n    say "Count: {len(items)}"\n}', items=kel_list(items)), out=[f'First: {items[0]}', f'Last: {items[-1]}', f'Count: {len(items)}'])


@template
def list_grid(r):
    grid = [numbers(r, 1, 9, 3) for _ in range(r.randint(2, 3))]
    return dict(task=f'adds up every number in the grid {kel_list_nested(grid)}', kel=K('fn main {\n    let grid = $grid\n    var total = 0\n    for row in grid {\n        for n in row {\n            total += n\n        }\n    }\n    say total\n}', grid=kel_list_nested(grid)), out=[str(sum(map(sum, grid)))])


def kel_list_nested(grid):
    return '[' + ', '.join(kel_list(row) for row in grid) + ']'


@template
def list_multiples(r):
    k, n = r.randint(2, 9), r.randint(3, 9)
    return dict(task=f'builds a list of the first {n} multiples of {k}', kel=K('fn main {\n    var multiples: [int] = []\n    for i in 1..$n {\n        multiples.add(i * $k)\n    }\n    say multiples\n}', n=n, k=k), out=[kel_list([i * k for i in range(1, n + 1)])],
                py=f'multiples = []\nfor i in range(1, {n} + 1):\n    multiples.append(i * {k})\nprint(multiples)')


@template
def list_first_over(r):
    items, limit = numbers(r, 1, 50, 6), r.randint(10, 40)
    found = next((i for i in items if i > limit), None)
    return dict(task=f'stops at the first number above {limit} in {kel_list(items)} and prints it', kel=K('fn main {\n    for n in $items {\n        if n > $limit {\n            say n\n            break\n        }\n    }\n}', items=kel_list(items), limit=limit), out=[str(found)] if found is not None else [],
                py=f'for n in {kel_list(items)}:\n    if n > {limit}:\n        print(n)\n        break')


@template
def list_skip(r):
    items = numbers(r, -9, 9, 6)
    return dict(task=f'prints the numbers in {kel_list(items)} but skips the negative ones using continue', kel=K('fn main {\n    for n in $items {\n        if n < 0 {\n            continue\n        }\n        say n\n    }\n}', items=kel_list(items)), out=[str(i) for i in items if i >= 0])


@template
def list_count_word(r):
    pool = ['cat', 'dog', 'fox']
    items, word = [r.choice(pool) for _ in range(6)], r.choice(pool)
    return dict(task=f'counts how many times "{word}" is in {kel_list(items)}', kel=K('fn main {\n    let words = $items\n    var count = 0\n    for word in words {\n        if word == "$word" {\n            count += 1\n        }\n    }\n    say count\n}', items=kel_list(items), word=word), out=[str(items.count(word))])


@template
def list_copy(r):
    items = numbers(r, 1, 9, 3)
    return dict(task=f'copies {kel_list(items)}, adds 100 to the copy, and shows that the first list did not change', kel=K('fn main {\n    let first = $items\n    var second = first\n    second.add(100)\n    say first\n    say second\n}', items=kel_list(items)), out=[kel_list(items), kel_list(items + [100])])


# ---- Text tools
WORDS = ['otter', 'kelus', 'hello', 'banana', 'rainbow', 'planet', 'music']
SENTENCES = ['the quick brown fox', 'kel is a small language', 'i like to code', 'rain falls on the sea', 'one two three four']


@template
def text_upper(r):
    word = one(r, *WORDS)
    return dict(task=f'prints "{word}" in capital letters', kel=K('fn main {\n    say "$word".upper()\n}', word=word), out=[word.upper()], py=f'print("{word}".upper())')


@template
def text_lower(r):
    word = one(r, 'HELLO', 'Kel', 'LOUD', 'MiXeD')
    return dict(task=f'prints "{word}" in small letters', kel=K('fn main {\n    let word = "$word"\n    say word.lower()\n}', word=word), out=[word.lower()])


@template
def text_has(r):
    sentence, part = one(r, *SENTENCES), one(r, 'fox', 'code', 'sea', 'small', 'two')
    return dict(task=f'checks whether "{sentence}" contains "{part}"', kel=K('fn main {\n    let text = "$sentence"\n    if text.contains("$part") {\n        say "Found it"\n    } else {\n        say "Not there"\n    }\n}', sentence=sentence, part=part), out=['Found it' if part in sentence else 'Not there'],
                py=f'text = "{sentence}"\nif "{part}" in text:\n    print("Found it")\nelse:\n    print("Not there")')


@template
def text_split(r):
    sentence = one(r, *SENTENCES)
    return dict(task=f'prints each word of "{sentence}" on its own line', kel=K('fn main {\n    let words = "$sentence".split(" ")\n    for word in words {\n        say word\n    }\n}', sentence=sentence), out=sentence.split(' '),
                py=f'words = "{sentence}".split(" ")\nfor word in words:\n    print(word)')


@template
def text_swap(r):
    sentence = one(r, *SENTENCES)
    old = sentence.split(' ')[0]
    new_word = one(r, 'a', 'my', 'our')
    return dict(task=f'replaces "{old}" with "{new_word}" in "{sentence}"', kel=K('fn main {\n    say "$sentence".replace("$old", "$new")\n}', sentence=sentence, old=old, new=new_word), out=[sentence.replace(old, new_word)])


@template
def text_first(r):
    word, n = one(r, *WORDS), r.randint(2, 4)
    return dict(task=f'prints the first {n} letters of "{word}"', kel=K('fn main {\n    let word = "$word"\n    say word.slice(0, $n)\n}', word=word, n=n), out=[word[:n]], py=f'word = "{word}"\nprint(word[0:{n}])')


@template
def text_each_letter(r):
    word = one(r, *WORDS)
    return dict(task=f'prints each letter of "{word}" on its own line', kel=K('fn main {\n    for letter in "$word" {\n        say letter\n    }\n}', word=word), out=list(word), py=f'for letter in "{word}":\n    print(letter)')


@template
def text_count_letter(r):
    word, letter = one(r, *WORDS), one(r, 'a', 'e', 'o', 'l', 'n')
    return dict(task=f'counts how many times "{letter}" is in "{word}"', kel=K('fn main {\n    var count = 0\n    for letter in "$word" {\n        if letter == "$letter" {\n            count += 1\n        }\n    }\n    say count\n}', word=word, letter=letter), out=[str(word.count(letter))])


@template
def text_letter_at(r):
    word = one(r, *WORDS)
    return dict(task=f'prints the first and last letters of "{word}"', kel=K('fn main {\n    let word = "$word"\n    say word[0]\n    say word[len(word) - 1]\n}', word=word), out=[word[0], word[-1]])


@template
def text_starts(r):
    word, part = one(r, *WORDS), one(r, 'he', 'ot', 'ra', 'ba')
    return dict(task=f'checks whether "{word}" starts with "{part}"', kel=K('fn main {\n    say "$word".starts_with("$part")\n}', word=word, part=part), out=[show(word.startswith(part))])


@template
def text_trimmed(r):
    word = one(r, *WORDS)
    return dict(task=f'removes the spaces around "  {word}  "', kel=K('fn main {\n    let messy = "  $word  "\n    say messy.trim()\n}', word=word), out=[word])


@template
def text_number(r):
    a, b = r.randint(1, 500), r.randint(1, 500)
    return dict(task=f'turns the texts "{a}" and "{b}" into numbers and adds them', kel=K('fn main {\n    let a = int("$a")\n    let b = int("$b")\n    say a + b\n}', a=a, b=b), out=[str(a + b)], py=f'a = int("{a}")\nb = int("{b}")\nprint(a + b)')


@template
def text_from_number(r):
    n = r.randint(1, 900)
    return dict(task=f'joins "Score: " and the number {n} with + by turning the number into text', kel=K('fn main {\n    let score = $n\n    say "Score: " + str(score)\n}', n=n), out=[f'Score: {n}'], py=f'score = {n}\nprint("Score: " + str(score))')


# ---- Your own types
@template
def type_point(r):
    a, b = float(r.randint(0, 9)), float(r.randint(0, 9))
    c, d = float(r.randint(0, 9)), float(r.randint(0, 9))
    dist = ((a - c) ** 2 + (b - d) ** 2) ** 0.5
    return dict(task=f'has a type Point and a method that gives the distance between ({a}, {b}) and ({c}, {d})',
                kel=K('type Point { x: float, y: float }\n\nfn Point.distance_to(other: Point) -> float {\n    let dx = self.x - other.x\n    let dy = self.y - other.y\n    return sqrt(dx * dx + dy * dy)\n}\n\nfn main {\n    let a = Point(x: $a, y: $b)\n    let b = Point(x: $c, y: $d)\n    say a.distance_to(b)\n}', a=a, b=b, c=c, d=d), out=[show(dist)])


@template
def type_circle(r):
    radius = float(r.randint(1, 12))
    return dict(task=f'has a type Circle with a radius and a method for its area, for a radius of {radius}',
                kel=K('type Circle { radius: float }\n\nfn Circle.area -> float {\n    return 3.14 * self.radius * self.radius\n}\n\nfn main {\n    let circle = Circle(radius: $radius)\n    say circle.area()\n}', radius=radius), out=[show(3.14 * radius * radius)],
                py=f'class Circle:\n    def __init__(self, radius):\n        self.radius = radius\n\n    def area(self):\n        return 3.14 * self.radius * self.radius\n\ncircle = Circle({radius})\nprint(circle.area())')


@template
def type_student(r):
    name, score = one(r, *NAMES), r.randint(30, 100)
    return dict(task=f'has a type Student with a name and a score, and a method that says whether {name} with a score of {score} passed with 60 or more',
                kel=K('type Student {\n    name: str\n    score: int\n}\n\nfn Student.passed -> bool {\n    return self.score >= 60\n}\n\nfn main {\n    let student = Student(name: "$name", score: $score)\n    say "{student.name} passed: {student.passed()}"\n}', name=name, score=score), out=[f'{name} passed: {show(score >= 60)}'])


@template
def type_show(r):
    name, age = one(r, *NAMES), r.randint(5, 80)
    return dict(task=f'makes a type Person for {name}, aged {age}, and prints it', kel=K('type Person { name: str, age: int }\n\nfn main {\n    let person = Person(name: "$name", age: $age)\n    say person\n}', name=name, age=age), out=[f'Person(name: "{name}", age: {age})'])


@template
def type_change(r):
    name, speed, more = one(r, 'Red', 'Comet', 'Bolt'), r.randint(10, 80), r.randint(5, 40)
    return dict(task=f'has a type Car named {name} with a speed of {speed}, then raises the speed by {more}', kel=K('type Car { name: str, speed: int }\n\nfn main {\n    var car = Car(name: "$name", speed: $speed)\n    car.speed += $more\n    say "{car.name} goes {car.speed}"\n}', name=name, speed=speed, more=more), out=[f'{name} goes {speed + more}'])


@template
def type_new_value(r):
    balance, amount = r.randint(50, 500), r.randint(10, 300)
    return dict(task=f'has a type Account with a balance of {balance} and a deposit method that returns a new Account with {amount} more', kel=K('type Account { balance: int }\n\nfn Account.deposit(amount: int) -> Account {\n    return Account(balance: self.balance + amount)\n}\n\nfn main {\n    var account = Account(balance: $balance)\n    account = account.deposit($amount)\n    say account.balance\n}', balance=balance, amount=amount), out=[str(balance + amount)])


@template
def type_list(r):
    pairs = [(one(r, 'Ana', 'Bo'), r.randint(10, 99)), (one(r, 'Mina', 'Leo'), r.randint(10, 99)), (one(r, 'Kim', 'Sam'), r.randint(10, 99))]
    made = ',\n        '.join(f'Student(name: "{n}", score: {s})' for n, s in pairs)
    return dict(task=f'keeps a list of three Student values with the scores {pairs[0][1]}, {pairs[1][1]} and {pairs[2][1]}, and prints their average', kel=K('type Student { name: str, score: int }\n\nfn main {\n    let students = [\n        $made\n    ]\n    var total = 0\n    for student in students {\n        total += student.score\n    }\n    say float(total) / float(len(students))\n}', made=made), out=[show(sum(s for _, s in pairs) / 3)])


@template
def type_best(r):
    pairs = [(n, r.randint(10, 99)) for n in r.sample(NAMES, 3)]
    best = pairs[0]
    for pair in pairs:
        if pair[1] > best[1]:
            best = pair
    made = ', '.join(f'Player("{n}", {s})' for n, s in pairs)
    return dict(task=f'finds the player with the highest score among {", ".join(f"{n} with {s}" for n, s in pairs)}', kel=K('type Player { name: str, score: int }\n\nfn main {\n    let players = [$made]\n    var best = players[0]\n    for player in players {\n        if player.score > best.score {\n            best = player\n        }\n    }\n    say "{best.name} wins with {best.score}"\n}', made=made), out=[f'{best[0]} wins with {best[1]}'])


@template
def type_with_list(r):
    name, first, second = one(r, *NAMES), one(r, 'map', 'key', 'rope'), one(r, 'lamp', 'coin', 'book')
    return dict(task=f'has a type Bag with an owner called {name} and a list of items, then adds "{first}" and "{second}"', kel=K('type Bag {\n    owner: str\n    items: [str]\n}\n\nfn main {\n    var bag = Bag(owner: "$name", items: [])\n    bag.items.add("$first")\n    bag.items.add("$second")\n    say "{bag.owner} has {len(bag.items)} items: {bag.items.join(", ")}"\n}'.replace('{bag.items.join(", ")}', '" + bag.items.join(", ")').replace('items: " + bag.items.join(", ")"', 'items: " + bag.items.join(", ")'), name=name, first=first, second=second), out=[f'{name} has 2 items: {first}, {second}'])


# ---- Asking
@template
def ask_name(r):
    name = one(r, *NAMES)
    return dict(task='asks for a name and says hello', kel=K('fn main {\n    let name = ask("What is your name?")\n    say "Hello, {name}!"\n}'), out=[f'Hello, {name}!'], answers=[name], py='name = input("What is your name? ")\nprint(f"Hello, {name}!")')


@template
def ask_age(r):
    age, more = r.randint(5, 60), r.randint(1, 10)
    return dict(task=f'asks for an age and prints the age in {more} years', kel=K('fn main {\n    let age = int(ask("How old are you?"))\n    say "In $more years you will be {age + $more}"\n}', more=more), out=[f'In {more} years you will be {age + more}'], answers=[str(age)],
                py=f'age = int(input("How old are you? "))\nprint(f"In {more} years you will be {{age + {more}}}")')


@template
def ask_two(r):
    a, b = r.randint(1, 99), r.randint(1, 99)
    return dict(task='asks for two numbers and prints their sum', kel=K('fn main {\n    let a = int(ask("First number:"))\n    let b = int(ask("Second number:"))\n    say a + b\n}'), out=[str(a + b)], answers=[str(a), str(b)])


@template
def ask_decimal(r):
    c = r.randint(-5, 40)
    return dict(task='asks for a temperature in Celsius as a decimal number and prints it in Fahrenheit', kel=K('fn main {\n    let c = float(ask("Temperature in Celsius:"))\n    say c * 9.0 / 5.0 + 32.0\n}'), out=[show(float(c) * 9.0 / 5.0 + 32.0)], answers=[f'{c}.0'])


@template
def ask_yes(r):
    answer = one(r, 'yes', 'no', 'YES', 'maybe')
    return dict(task='asks a yes or no question and answers either way', kel=K('fn main {\n    let answer = ask("Do you like coding? (yes or no)").lower()\n    if answer == "yes" {\n        say "Me too!"\n    } else {\n        say "Maybe later, then."\n    }\n}'), out=['Me too!' if answer.lower() == 'yes' else 'Maybe later, then.'], answers=[answer])


@template
def ask_guess(r):
    secret = r.randint(1, 20)
    guesses = [secret + 3, secret - 2, secret]
    return dict(task=f'is a guessing game with the secret number {secret}, saying higher or lower until the guess is right', kel=K('fn main {\n    let secret = $secret\n    var tries = 0\n    while true {\n        let guess = int(ask("Your guess:"))\n        tries += 1\n        if guess == secret {\n            break\n        }\n        if guess < secret {\n            say "Higher!"\n        } else {\n            say "Lower!"\n        }\n    }\n    say "Right, in {tries} tries!"\n}', secret=secret), out=['Lower!', 'Higher!', 'Right, in 3 tries!'], answers=[str(g) for g in guesses])


# ---- Math
@template
def math_root(r):
    n = float(r.randint(1, 30) ** 2)
    return dict(task=f'prints the square root of {n}', kel=K('fn main {\n    say sqrt($n)\n}', n=n), out=[show(n ** 0.5)])


@template
def math_hypotenuse(r):
    a, b = one(r, (3.0, 4.0), (5.0, 12.0), (8.0, 15.0), (7.0, 24.0))
    return dict(task=f'works out the long side of a right triangle with sides {a} and {b}', kel=K('fn main {\n    let a = $a\n    let b = $b\n    say sqrt(a * a + b * b)\n}', a=a, b=b), out=[show((a * a + b * b) ** 0.5)], py=f'import math\n\na = {a}\nb = {b}\nprint(math.sqrt(a * a + b * b))')


@template
def math_power(r):
    a, b = r.randint(2, 9), r.randint(2, 9)
    return dict(task=f'prints {a} to the power of {b} using pow', kel=K('fn main {\n    say pow($a, $b)\n}', a=a, b=b), out=[str(a ** b)], py=f'print({a} ** {b})')


@template
def math_abs(r):
    a, b = r.randint(1, 50), r.randint(51, 99)
    return dict(task=f'prints how far apart {a} and {b} are', kel=K('fn main {\n    say abs($a - $b)\n}', a=a, b=b), out=[str(abs(a - b))])


@template
def math_bounds(r):
    a, b = r.randint(1, 90), r.randint(1, 90)
    return dict(task=f'prints the smaller and the larger of {a} and {b} with min and max', kel=K('fn main {\n    say min($a, $b)\n    say max($a, $b)\n}', a=a, b=b), out=[str(min(a, b)), str(max(a, b))])


@template
def math_round(r):
    x = one(r, 1.2, 2.5, 3.7, 8.49, 9.51)
    import math
    return dict(task=f'prints {x} rounded, rounded down and rounded up', kel=K('fn main {\n    let x = $x\n    say round(x)\n    say floor(x)\n    say ceil(x)\n}', x=x), out=[str(math.floor(x + 0.5)), str(math.floor(x)), str(math.ceil(x))])


@template
def math_dice(r):
    n = r.randint(2, 6)
    return dict(task=f'rolls a six-sided die {n} times and prints each roll', kel=K('fn main {\n    repeat $n {\n        say random(1, 6)\n    }\n}', n=n), out=None, py=f'import random\n\nfor _ in range({n}):\n    print(random.randint(1, 6))')


# ---- Ways to ask. All name the language, so that "Python" and "Kel" stay apart.
PROGRAM_ASKS = ['Write a Kel program that {t}.', 'write a .lus program that {t}', 'In Kel, write a program that {t}.', 'Can you write a Kel program that {t}?', 'kel code that {t}',
                'make a .lus file that {t}', 'I need a program in Kel that {t}.', 'Using the Kel language, write code that {t}.', 'lus program that {t} please', 'Show me a program in Kel (.lus) that {t}.',
                'write me some kel that {t}', 'Create a Kel script that {t}.', 'Write main.lus: a program that {t}.', 'code in kel lang that {t}', 'Please write a program in .lus that {t}.',
                'how do i write a program in kel that {t}?']
FUNCTION_ASKS = ['Write a Kel function {s} for a program that {t}.', 'In .lus, write {s} for a program that {t}.', 'kel: write the function {s}, in a program that {t}']
CONVERT_ASKS = ['Convert this Python to Kel:\n\n```python\n{p}\n```', 'change this python code to .lus:\n\n```python\n{p}\n```', 'Rewrite this in Kel.\n\n```python\n{p}\n```', 'how would this look in kel?\n\n```python\n{p}\n```', 'translate to lus:\n\n{p}']
FIX_ASKS = ['My Kel program does not compile:\n\n```lus\n{c}```\n\nThe error says:\n{e}', 'why does this .lus code fail?\n\n```lus\n{c}```\n\n{e}', 'Fix this Kel code.\n\n```lus\n{c}```\n\nCompiler output:\n{e}', 'kel run gives me this error, can you fix it?\n\n{e}\n\n```lus\n{c}```']
EXPLAIN_ASKS = ['What does this Kel program do?\n\n```lus\n{c}```', 'explain this .lus code\n\n```lus\n{c}```', 'what does this do in kel?\n\n```lus\n{c}```']
SAY_RUN = ['', '', '', 'Save it as main.lus and run it with `kel run main.lus`.', 'Run it with `kel run main.lus`.', 'To try it: `node bin/kel.js run main.lus`.']

# Breaks a program the way people do, and the sentence that goes with the fix. The compiler supplies the error itself.
BREAKS = [
    (lambda c: re.sub(r'\bsay (.+)', r'print(\1)', c, count=1), 'Kel shows things with `say`, not `print(...)`.'),
    (lambda c: re.sub(r'\bfn (\w+)\(', r'def \1(', c, count=1), 'Kel makes functions with `fn`, not `def`.'),
    (lambda c: c.replace('&&', 'and', 1), 'Kel writes "and" as `&&`.'),
    (lambda c: c.replace('||', 'or', 1), 'Kel writes "or" as `||`.'),
    (lambda c: re.sub(r'\b(if [^\n{]+?) \{', r'\1:', c, count=1), 'Kel uses `{ }` for blocks, so there is no colon after the condition.'),
    (lambda c: re.sub(r'\b(while [^\n{]+?) \{', r'\1:', c, count=1), 'Kel uses `{ }` for blocks, so there is no colon after the condition.'),
    (lambda c: c.replace('var ', 'let ', 1), 'A `let` value can\'t change, and this one does, so it needs `var`.'),
    (lambda c: c.rstrip()[:-1].rstrip() + '\n', 'A block is missing its closing `}`.'),
    (lambda c: re.sub(r'    return ([^\n]+)\n\}', r'    let answer = \1\n}', c, count=1), 'The function has to give back its value with `return`.'),
    (lambda c: re.sub(r'say "([^"{]*)\{(\w+)\}"', r'say "\1" + \2', c, count=1), 'Text and numbers can\'t be joined with `+`. Put the value in a template instead: `"Total: {total}"`.'),
    (lambda c: c.replace('else if', 'elif', 1), 'Kel writes this as `else if`.'),
    (lambda c: c.replace('.add(', '.append(', 1), 'Kel lists use `.add(x)`, not `.append(x)`.'),
    (lambda c: c.replace('.add(', '.push(', 1), 'Kel lists use `.add(x)`, not `.push(x)`.'),
    (lambda c: re.sub(r'\blen\((\w+)\)', r'\1.length', c, count=1), 'Kel gets the length with `len(x)`.'),
    (lambda c: c.replace('type ', 'class ', 1), 'Kel makes your own types with `type`, not `class`.'),
    (lambda c: re.sub(r': \[(\w+)\] = \[\]', ' = []', c, count=1), 'An empty list has to say what it will hold, like `var items: [int] = []`.'),
    (lambda c: c.replace('self.', '', 1), 'Inside a method, fields are reached through `self`, like `self.x`.'),
]
# ---- Questions about the language. (ways to ask, answer)
FACTS = [
    (['What is Kel?', 'what is the kel language', 'what is a .lus file?', 'what is lus?', 'tell me about the kel programming language', 'what language uses .lus files?'],
     'Kel is a small, beginner-friendly language that compiles to WebAssembly, so it runs in every browser. It has clean `{ }` blocks, types like Rust and friendly words like `say` and `repeat`. Kel files end in `.lus`.'),
    (['How do I run a Kel program?', 'how do i run a .lus file', 'how to run kel code?', 'what command runs a lus file?'],
     'Use the command line:\n\n```sh\nnode bin/kel.js run main.lus\n```\n\nOr, if `kel` is installed, `kel run main.lus`. `kel check main.lus` finds mistakes without running, and `kel build main.lus` writes a `.wasm` file.'),
    (['How do I print in Kel?', 'how to print in kel', 'kel print statement?', 'how do i show text in lus', 'does kel have print?'],
     'Use `say`. There is no `print`:\n\n```lus\nfn main {\n    say "Hello!"\n    say 1 + 2\n}\n```'),
    (['What is the difference between let and var in Kel?', 'let vs var in kel', 'can you change a let in .lus?', 'how do i make a variable in kel'],
     '`let` makes a value that can\'t change, and `var` makes one that can. Kel works out the type for you:\n\n```lus\nfn main {\n    let name = "Kel"\n    var age = 14\n    age += 1\n    say "{name} is {age}"\n}\n```'),
    (['What types does Kel have?', 'kel data types', 'what types are in .lus?', 'list the types in kel'],
     'Kel has `int` (64-bit whole numbers), `float`, `bool`, `str` (text), lists like `[int]`, and your own types made with `type`.'),
    (['How do I write a function in Kel?', 'kel function syntax', 'how do i define a function in .lus', 'how to make a function in lus'],
     'Use `fn`, with a type for each parameter and `-> type` for what it gives back:\n\n```lus\nfn add(a: int, b: int) -> int {\n    return a + b\n}\n\nfn main {\n    say add(2, 3)\n}\n```\n\nEvery program starts at `fn main { }`.'),
    (['What loops does Kel have?', 'how do i loop in kel', 'kel for loop syntax', 'how do loops work in .lus', 'does kel have while loops?'],
     '`repeat 3 { }` runs a block a number of times, `for i in 1..10 { }` counts from 1 to 10, including 10, `for x in list { }` goes through a list or text, and `while x < 10 { }` runs while something is true. `break` and `continue` work in all of them.'),
    (['How does if work in Kel?', 'kel if else syntax', 'how do i write else if in .lus', 'if statement in lus'],
     'Use `if`, `else if` and `else`, with `{ }` and no colon or brackets:\n\n```lus\nfn main {\n    let score = 75\n    if score >= 90 {\n        say "A"\n    } else if score >= 70 {\n        say "B"\n    } else {\n        say "Keep going!"\n    }\n}\n```'),
    (['How do I put a variable in text in Kel?', 'string templates in kel', 'how do i format strings in .lus', 'kel string interpolation'],
     'Put it in `{ }` inside the text: `say "Hi {name}, you are {age}"`. Any value works, such as `{len(name)}` or `{a + b}`. Write `\\{` for a plain `{`.'),
    (['How do I join text in Kel?', 'concatenate strings in kel', 'can i add a number to a string in .lus?', 'why does "Score: " + 5 fail in kel'],
     'Join text with `+`, but only text with text. To mix in a number, use a template, `say "Score: {score}"`, or turn it into text first: `"Score: " + str(score)`.'),
    (['How do I convert between int and float in Kel?', 'kel convert int to float', 'why can\'t i add an int and a float in .lus'],
     'Kel doesn\'t mix them on its own. Convert with `float(x)` or `int(x)`, as in `float(total) / 3.0`. `int(x)` drops the part after the point. Both also read text, as in `int("42")`, and `str(x)` turns anything into text.'),
    (['What built-in functions does Kel have?', 'kel builtins', 'what built ins are in .lus?'],
     '`int(x)`, `float(x)` and `str(x)` convert, `len(x)` measures text and lists, `ask("question")` reads an answer, and for math there are `abs`, `min`, `max`, `pow`, `sqrt`, `floor`, `ceil`, `round` and `random`.'),
    (['How do I write comments in Kel?', 'kel comments', 'comment syntax in .lus', 'does # work for comments in kel'], 'Comments start with `//`, as in `// this is a comment`. `#` does not work.'),
    (['Does Kel need semicolons?', 'do i need ; in kel', 'semicolons in .lus?'], 'No. A line break ends a statement. You can use `;` to put two statements on one line.'),
    (['What operators does Kel have?', 'kel operators', 'how do i write and or not in .lus', 'how do you write and in kel'],
     'Arithmetic is `+ - * / %`, with `+= -= *= /= %=`. Comparison is `== != < <= > >=`. Logic is `&&`, `||` and `!`, not `and`, `or` and `not`. `/` on two ints gives a whole number, and `%` works on ints only.'),
    (['Does Kel have lists?', 'does kel have arrays?', 'how do i make a list in kel', 'how do i make an array in .lus', 'lists in lus', 'how do lists work in kel?'],
     'Yes. A list holds one type and counts from 0:\n\n```lus\nfn main {\n    var scores = [3, 1, 2]\n    scores.add(5)\n    scores[0] = 9\n    say scores[0]\n    say len(scores)\n    for score in scores {\n        say score\n    }\n}\n```\n\nAn empty list says what it will hold: `var names: [str] = []`.'),
    (['What can lists do in Kel?', 'kel list methods', 'what methods do lists have in .lus?', 'how do i append to a list in kel', 'how do i sort a list in kel', 'how do i remove an item from a list in lus'],
     'Lists have `.add(x)`, `.pop()`, `.remove(i)`, `.sort()`, `.reverse()`, `.contains(x)`, `.index_of(x)` and, for text, `.join(", ")`. `len(list)` gives the length, and `list[0]` reads or sets an item.'),
    (['Does Kel have classes?', 'does kel have structs?', 'how do i make a class in kel', 'objects in .lus', 'how do i make my own type in kel', 'kel type syntax'],
     'Use `type`, not `class`. Build a value by calling the type, with or without field names:\n\n```lus\ntype Point { x: float, y: float }\n\nfn main {\n    var p = Point(x: 1.0, y: 2.0)\n    p.x = 5.0\n    say p\n}\n```'),
    (['How do I write a method in Kel?', 'kel methods', 'how do methods work in .lus', 'what is self in kel?', 'how do i add a function to a type in lus'],
     'Write `fn TypeName.method`. Inside it, `self` is the value it was called on, and it can read `self` but not change it. To change something, return a new value:\n\n```lus\ntype Point { x: float, y: float }\n\nfn Point.moved(dx: float) -> Point {\n    return Point(x: self.x + dx, y: self.y)\n}\n\nfn main {\n    var p = Point(x: 1.0, y: 2.0)\n    p = p.moved(3.0)\n    say p\n}\n```'),
    (['Does var b = a copy a list in Kel?', 'are kel lists references or values?', 'why did my list not change in my kel function?', 'can a function change a list in .lus', "why can't i change a parameter in kel"],
     'Lists and your own types act like values. `var b = a` makes a copy, so changing `b` leaves `a` alone. A function can read what it is given but not change it, so it returns a new value: `items = with_extra(items)`.'),
    (['How do I read input in Kel?', 'how do i ask the user for their name in kel', 'does kel have input?', 'read from keyboard in lus', 'how does ask work in .lus?'],
     'Use `ask`, which shows a question and gives back the answer as text. For a number, wrap it in `int(...)` or `float(...)`:\n\n```lus\nfn main {\n    let name = ask("Your name?")\n    let age = int(ask("Your age?"))\n    say "{name} is {age}"\n}\n```'),
    (['What can text do in Kel?', 'kel string methods', 'how do i make text uppercase in kel', 'how do i split a string in .lus', 'how do i get a letter from text in lus'],
     'Text has `.upper()`, `.lower()`, `.trim()`, `.contains(t)`, `.starts_with(t)`, `.ends_with(t)`, `.replace(a, b)`, `.split(",")` and `.slice(0, 5)`. `s[0]` gives one letter, `len(s)` counts letters, and `for letter in s { }` goes through them.'),
    (['What math functions does Kel have?', 'does kel have sqrt?', 'how do i get a random number in .lus', 'does kel have random numbers?', 'how do i round a number in kel'],
     'Kel has `abs(x)`, `min(a, b)`, `max(a, b)`, `pow(a, b)`, `sqrt(x)`, `floor(x)`, `ceil(x)`, `round(x)` and `random(1, 6)`, which includes both ends. `sqrt` takes a float, and `floor`, `ceil` and `round` give back an int.'),
    (['Does Kel have break?', 'how do i stop a loop in kel', 'does kel have continue?', 'how do i skip a loop turn in .lus'],
     'Yes. `break` leaves the loop, and `continue` skips to the next turn. They work in every kind of loop.'),
    (['Can Kel read files?', 'how do i import a library in kel', 'does kel have imports?', 'can kel make web requests?', 'does kel have dictionaries?', 'does kel have maps?', 'does kel have try catch?'],
     'Not yet. Kel has no imports, files, networking, dictionaries or error catching so far. It has functions, your own types, lists, text, `ask`, `say`, loops and choices.'),
    (['Does Kel have null?', 'what is None in kel?', 'null in .lus'], 'No. Kel has no `None` or `null`. A function that returns a value must always return one.'),
    (['Is Kel fast?', 'how fast is kel?', 'is kel faster than python?'],
     'On small number-heavy code, yes. Kel compiles to WebAssembly, and `fib(38)` takes about 0.12 s, close to C at 0.10 s and well ahead of Python at about 3.8 s. Bigger programs will show a gap, because Kel has no optimizer of its own yet.'),
    (['How does a for loop range work in Kel?', 'is 1..10 inclusive in kel?', 'does 1..5 include 5 in .lus?'], 'Yes, both ends count, so `for i in 1..5 { }` runs with 1, 2, 3, 4 and 5. The counter `i` can\'t be changed inside the loop.'),
    (['Why does Kel say my function can reach its end without returning?', 'kel error: can reach its end without returning', 'what does "can reach its end" mean in .lus'],
     'A function with `-> type` must return on every path. Add a `return` as its last line, or make sure both sides of an `if` / `else` at the end return.'),
    (['Where does a Kel program start?', 'does kel need a main function?', 'what is fn main in .lus'], 'At `fn main { }`, and every program needs one. `main` takes nothing and returns nothing. All code goes inside functions.'),
]
CASUAL = ['', '', 'hey ', 'ok ', 'quick question: ']


def fenced(code):
    return f'```lus\n{code}```'


def check(programs, run):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
        json.dump(programs, handle)
    try:
        out = subprocess.run(['node', str(ROOT / 'data/check_lus.mjs'), handle.name, *(['--run'] if run else [])], capture_output=True, text=True, timeout=900, check=True).stdout
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return {r['id']: r for r in json.loads(out)}


def samples(rng, per_template):
    """Programs from every template, verified: compiled, run, and the output equal to the answer worked out here."""
    found, seen = [], set()
    for make in TEMPLATES:
        for _ in range(per_template):
            s = make(rng)
            if s['kel'] not in seen:
                seen.add(s['kel'])
                found.append({**s, 'id': len(found), 'name': make.__name__})
    results = check([{'id': s['id'], 'code': s['kel'], 'answers': s.get('answers', [])} for s in found], run=True)
    good = []
    for s in found:
        r = results[s['id']]
        if not r['ok']:
            raise SystemExit(f"{s['name']} does not compile: {r['error']}\n{s['kel']}")
        if s['out'] is not None and r['lines'] != s['out']:  # None: uses random
            raise SystemExit(f"{s['name']} printed {r['lines'][:6]}, expected {s['out'][:6]}\n{s['kel']}")
        good.append(s)
    return good


def build(seed=2438, per_template=30):
    rng = random.Random(seed)
    programs = samples(rng, per_template)
    rows, copies = [], collections.Counter()

    def add(family, messages):
        code = messages[-1]['content']
        if copies[code] < MAX_COPIES:
            copies[code] += 1
            rows.append({'family': family, 'source': 'make_lus', 'language': 'lus', 'messages': messages})

    def user(text):
        return {'role': 'user', 'content': text}

    def assistant(text):
        return {'role': 'assistant', 'content': text}

    def answer(s):
        tail = rng.choice(SAY_RUN)
        return fenced(s['kel']) + (f'\n\n{tail}' if tail else '')

    # Fresh names for the second program in a chat, and a pool for "now do something else".
    for s in programs:
        for ask in rng.sample(PROGRAM_ASKS, 2):
            add('lus_program', [user(ask.format(t=s['task'])), assistant(answer(s))])
        if s.get('sig') and rng.random() < 0.7:
            add('lus_function', [user(rng.choice(FUNCTION_ASKS).format(s=s['sig'], t=s['task'])), assistant(fenced(s['kel']))])
        if s.get('py') and rng.random() < 0.8:
            add('lus_convert', [user(rng.choice(CONVERT_ASKS).format(p=s['py'])), assistant(fenced(s['kel']))])
        if rng.random() < 0.15:
            add('lus_explain', [user(rng.choice(EXPLAIN_ASKS).format(c=s['kel'])), assistant(f"It {s['task']}.")])
    # Later in a chat: a new request, or a change to the program just written. Changes are checked by the compiler below.
    pending = []
    for s in rng.sample(programs, min(900, len(programs))):
        first = [user(rng.choice(PROGRAM_ASKS).format(t=s['task'])), assistant(fenced(s['kel']))]
        kind = rng.random()
        if kind < 0.4:
            other = rng.choice(programs)
            add('lus_next', [*first, user(rng.choice(['now write one that {t}', 'also make one that {t}', 'next, a program that {t}', 'great, now one that {t}']).format(t=other['task'])), assistant(fenced(other['kel']))])
        elif kind < 0.7:
            word = rng.choice(['Done', 'Finished', 'All done', 'Bye'])
            changed = re.sub(r'(fn main \{\n(?:.*\n)*?)\}\n$', lambda m: f'{m.group(1)}    say "{word}"\n}}\n', s['kel'])  # main is always last
            pending.append(('lus_change', [*first, user(rng.choice(['also say "{w}" at the end', 'add a line at the end that prints "{w}"', 'make it say "{w}" when it finishes']).format(w=word))], changed))
        else:
            word = rng.choice(['Starting', 'Here we go', 'Hello'])
            changed = s['kel'].replace('fn main {\n', f'fn main {{\n    say "{word}"\n', 1)
            pending.append(('lus_change', [*first, user(rng.choice(['print "{w}" first', 'add "{w}" at the very start', 'say "{w}" before anything else']).format(w=word))], changed))
    verdict = check([{'id': i, 'code': code} for i, (_, _, code) in enumerate(pending)], run=False)
    for i, (family, messages, code) in enumerate(pending):
        if not verdict[i]['ok']:
            raise SystemExit(f'A changed program does not compile: {verdict[i]["error"]}\n{code}')
        add(family, [*messages, {'role': 'assistant', 'content': fenced(code)}])
    # Fixes: the error text comes from the real compiler; a break that still compiles is not used.
    broken = []
    for s in rng.sample(programs, len(programs)):
        for transform, why in rng.sample(BREAKS, len(BREAKS)):
            bad = transform(s['kel'])
            if bad != s['kel']:
                broken.append({'id': len(broken), 'code': bad, 'why': why, 'good': s['kel']})
                if rng.random() < 0.3:
                    continue
                break
    verdict = check([{'id': b['id'], 'code': b['code']} for b in broken], run=False)
    made = collections.Counter()
    for b in broken:
        r = verdict[b['id']]
        if not r['ok'] and made[b['why']] < 120:
            made[b['why']] += 1
            add('lus_fix', [user(rng.choice(FIX_ASKS).format(c=b['code'], e=r['error'])), assistant(f"{b['why']} Here it is fixed:\n\n{fenced(b['good'])}")])
    copies.clear()
    for asks, reply in FACTS:
        for ask in asks:
            for _ in range(2):
                rows.append({'family': 'lus_fact', 'source': 'make_lus', 'language': 'lus', 'messages': [user(rng.choice(CASUAL) + ask), assistant(reply)]})
    seen, unique = set(), []
    for row in rows:
        key = json.dumps(row['messages'])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    # The test's wordings stay out; its task kinds are not templates here.
    held = {' '.join(p.lower().split()) for p in lus_eval.prompts()}
    unique = [r for r in unique if ' '.join(r['messages'][0]['content'].lower().split()) not in held]
    rng.shuffle(unique)
    return [{**r, 'split': 'validation'} for r in unique[:HELD_OUT]] + [{**r, 'split': 'train'} for r in unique[HELD_OUT:]]


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/lus-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows from {len(TEMPLATES)} kinds of program; {collections.Counter(r["family"] for r in train).most_common()}')
