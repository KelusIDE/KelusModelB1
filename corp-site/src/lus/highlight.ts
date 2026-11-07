export type LusToken = {
  text: string;
  kind?:
    | "keyword"
    | "string"
    | "number"
    | "type"
    | "function"
    | "comment"
    | "operator"
    | "interpolation";
};

const keywords = new Set([
  "fn",
  "let",
  "var",
  "if",
  "else",
  "while",
  "repeat",
  "for",
  "in",
  "return",
  "say",
  "type",
  "break",
  "continue",
]);
const types = new Set(["int", "float", "bool", "str"]);
const words =
  /\s+|\/\/[^\r\n]*|[A-Za-z_][A-Za-z0-9_]*|[0-9][0-9_]*(?:\.[0-9][0-9_]*)?|->|\.\.|==|!=|<=|>=|&&|\|\||[+*\/%=!-]=?|[-<>]=?|[^]/y;

type Context = { mode: "code"; depth: number } | { mode: "string" };

export function tokenizeLus(source: string): LusToken[] {
  const tokens: LusToken[] = [];
  const contexts: Context[] = [{ mode: "code", depth: 0 }];
  let position = 0;
  let previousWord = "";
  const add = (text: string, kind?: LusToken["kind"]) => {
    if (!text) return;
    const last = tokens.at(-1);
    if (last && last.kind === kind) last.text += text;
    else tokens.push({ text, kind });
  };

  while (position < source.length) {
    const context = contexts.at(-1)!;
    const char = source[position];
    if (context.mode === "string") {
      if (char === "\n" || char === "\r") {
        contexts.splice(1);
        add(char);
        position++;
      } else if (char === "\\") {
        const length =
          position + 1 < source.length && !/[\r\n]/.test(source[position + 1])
            ? 2
            : 1;
        add(source.slice(position, position + length), "string");
        position += length;
      } else if (char === '"') {
        add(char, "string");
        position++;
        contexts.pop();
      } else if (char === "{") {
        add(char, "interpolation");
        position++;
        contexts.push({ mode: "code", depth: 1 });
      } else {
        const start = position++;
        while (
          position < source.length &&
          !'"\\{\r\n'.includes(source[position])
        )
          position++;
        add(source.slice(start, position), "string");
      }
      continue;
    }

    if (char === '"') {
      add(char, "string");
      position++;
      contexts.push({ mode: "string" });
      continue;
    }
    if (context.depth > 0 && char === "}") {
      context.depth--;
      add(char, context.depth === 0 ? "interpolation" : undefined);
      position++;
      if (context.depth === 0) contexts.pop();
      continue;
    }
    if (context.depth > 0 && char === "{") context.depth++;
    words.lastIndex = position;
    const text = words.exec(source)![0];
    let kind: LusToken["kind"];
    if (text.startsWith("//")) kind = "comment";
    else if (/^[0-9]/.test(text) || text === "true" || text === "false")
      kind = "number";
    else if (keywords.has(text)) kind = "keyword";
    else if (
      types.has(text) ||
      (previousWord === "type" && /^[A-Za-z_]/.test(text)) ||
      /^[A-Z][A-Za-z0-9_]*$/.test(text)
    )
      kind = "type";
    else if (
      /^[A-Za-z_]/.test(text) &&
      (previousWord === "fn" ||
        /^\s*\(/.test(source.slice(position + text.length)))
    )
      kind = "function";
    else if (
      /^(?:->|\.\.|[+*\/%=!-]=?|[-<>]=?|==|!=|<=|>=|&&|\|\|)$/.test(text)
    )
      kind = "operator";
    add(text, kind);
    if (/^[A-Za-z_]/.test(text)) previousWord = text;
    else if (!/^\s+$/.test(text) && kind !== "comment") previousWord = "";
    position += text.length;
  }
  return tokens;
}
