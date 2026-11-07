import { test } from "node:test";
import assert from "node:assert/strict";
import { tokenizeLus } from "../src/lus/highlight.ts";
import { LUS_EXAMPLES } from "../src/lus/examples.ts";

const has = (tokens, kind, text) =>
  tokens.some((token) => token.kind === kind && token.text.includes(text));
test("highlights Lus declarations, calls, types, numbers and comments", () => {
  const source =
    "type Player { name: str, score: int }\nfn square(n: int) -> int { return n * n }\nfn main {\n    let score = square(4)\n    say true\n    // say 999 is only a comment\n}";
  const tokens = tokenizeLus(source);
  for (const [kind, text] of [
    ["keyword", "fn"],
    ["type", "Player"],
    ["type", "int"],
    ["function", "square"],
    ["number", "4"],
    ["number", "true"],
    ["comment", "say 999"],
    ["operator", "->"],
  ])
    assert.ok(has(tokens, kind, text), `${kind}: ${text}`);
});
test("highlights expressions inside strings and respects escaped braces, quotes and comment text", () => {
  const source = String.raw`say "Hi {name}, score {square(4) + 1}: \{literal}, \"quoted\", https://example.com"`;
  const tokens = tokenizeLus(source);
  assert.ok(has(tokens, "interpolation", "{"));
  assert.ok(has(tokens, "function", "square"));
  assert.ok(has(tokens, "number", "4"));
  assert.ok(has(tokens, "string", "literal"));
  assert.ok(has(tokens, "string", "https://example.com"));
  assert.ok(!tokens.some((token) => token.kind === "comment"));
  assert.equal(tokens.map((token) => token.text).join(""), source);
});
test("preserves every character of real examples and incomplete or hostile source", () => {
  const sources = LUS_EXAMPLES.map((example) => example.code).concat([
    "",
    "\n\n",
    '\t say "안녕 🦦 {name}"\n',
    'fn main { say "unfinished {square(4',
    'say "text\nlet score = 1',
    "<script>alert(1)</script>",
    '"{'.repeat(2000),
    "let n = 1..10\r\n// comment\r\n",
  ]);
  for (const source of sources)
    assert.equal(
      tokenizeLus(source)
        .map((token) => token.text)
        .join(""),
      source,
    );
});
test("ranges remain distinct from decimal numbers", () => {
  const tokens = tokenizeLus("for n in 1..10 { say 3.14 }");
  assert.ok(has(tokens, "operator", ".."));
  assert.ok(has(tokens, "number", "3.14"));
});
