"""Local chat UI using Python's HTTP server; no web or AI framework."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from .model import Model
from .tokenizer import conversation
from .runtime import prepare_reply

WEB = Path(__file__).resolve().parent.parent / "web"


def validate_chat(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not (1 <= len(messages) <= 31):
        raise ValueError("Provide 1–31 alternating chat messages")
    if any(not isinstance(message, dict) or not isinstance(message.get("content"), str) or len(message["content"]) > 8000 for message in messages):
        raise ValueError("Each chat message must contain at most 8000 characters")
    conversation(messages, generation=True)
    max_tokens = payload.get("max_tokens", 240)
    temperature = payload.get("temperature", 0)
    if type(max_tokens) is not int or not (1 <= max_tokens <= 512):
        raise ValueError("Maximum output must be between 1 and 512 tokens")
    if type(temperature) not in (int, float) or not (0 <= temperature <= 2):
        raise ValueError("Temperature must be between 0 and 2")
    return messages, max_tokens, float(temperature)


def make_handler(model, header, checkpoint):
    generation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def allowed_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def send_json(self, status, payload):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if not self.allowed_host():
                self.send_json(403, {"error": "Local requests only"}); return
            if self.path == "/api/info":
                self.send_json(200, {"name": model.name, "parameters": model.parameter_count,
                                     "config": header["config"], "training": header.get("metadata", {}),
                                     "tokenizer": model.tokenizer.kind,
                                     "checkpoint": Path(checkpoint).name, "engine": "Own transformer + autograd / NumPy arithmetic"})
                return
            names = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/styles.css": ("styles.css", "text/css"),
                     "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            if self.path not in names:
                self.send_json(404, {"error": "Not found"}); return
            filename, content_type = names[self.path]
            data = (WEB / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.allowed_host():
                self.send_json(403, {"error": "Local requests only"}); return
            if self.path != "/api/chat":
                self.send_json(404, {"error": "Not found"}); return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                self.send_json(403, {"error": "Cross-origin requests are refused"}); return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json(415, {"error": "Send application/json"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not (0 < length <= 65536):
                    raise ValueError("Request must contain 1–65536 bytes")
                messages, max_tokens, temperature = validate_chat(json.loads(self.rfile.read(length)))
                retained, dropped, tool = prepare_reply(model, messages, max_tokens)
            except (ValueError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)}); return
            if not generation_lock.acquire(blocking=False):
                self.send_json(429, {"error": "B1 is answering another message. Try again when it finishes."}); return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                started = time.monotonic()
                chunks = 0
                source = tool[1] if tool else 'model'
                output = [tool[0]] if tool else model.generate(retained, max_tokens=max_tokens, temperature=temperature)
                self.wfile.write((json.dumps({"source": source}) + "\n").encode("utf-8"))
                for text in output:
                    self.wfile.write((json.dumps({"text": text}) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    chunks += 1
                self.wfile.write((json.dumps({"done": True, "chunks": chunks, "history_messages_omitted": dropped, "seconds": round(time.monotonic() - started, 2)}) + "\n").encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                try:
                    self.wfile.write(b'{"error":"Generation failed. Check the local server log."}\n')
                except OSError:
                    pass
                raise
            finally:
                generation_lock.release()

        def log_message(self, format, *args):
            # Paths/status only; no user prompts are logged or saved.
            print(f"B1 HTTP: {format % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve Kelus Model B1 on localhost, without a web framework")
    parser.add_argument("--checkpoint", default="checkpoints/b1-v3.best.npz")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (1024 <= args.port <= 65535):
        parser.error("Choose a port between 1024 and 65535")
    try:
        model, header, _ = Model.load(args.checkpoint)
    except FileNotFoundError:
        parser.error("Train first: python3 -m kelus.train")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, header, args.checkpoint))
    server.daemon_threads = True
    print(f"{model.name} · {model.parameter_count:,} parameters · http://127.0.0.1:{args.port}", flush=True)
    print("Small educational model. Generated code is text only and is never executed.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
