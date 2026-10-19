"""Interactive terminal chat with the actual B1 model."""
import argparse
from .model import Model
from .runtime import prepare_reply


def main():
    parser = argparse.ArgumentParser(description="Chat with Kelus Model B1 locally")
    parser.add_argument("--checkpoint", default="checkpoints/b1-v3.best.npz")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=240)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    print(f"{model.name} · {model.parameter_count:,} parameters · {header.get('metadata', {}).get('steps', 0)} training steps")
    print("Small educational model. Expect mistakes. /reset clears context; /quit exits.")
    messages = []
    try:
        while True:
            prompt = input("\nYou: ").strip()
            if prompt == "/quit":
                break
            if prompt == "/reset":
                messages.clear(); print("Context cleared."); continue
            if not prompt:
                continue
            messages.append({"role": "user", "content": prompt})
            retained, omitted, tool = prepare_reply(model, messages[-15:], args.max_tokens)
            if omitted: print("[Older turns omitted; B1 uses at most one prior exchange.]")
            print((tool[1] if tool else "B1") + ": ", end="", flush=True)
            response = ""
            output = [tool[0]] if tool else model.generate(retained, max_tokens=args.max_tokens, temperature=args.temperature)
            for text in output:
                response += text
                print(text, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": response})
            messages = messages[-16:]
    except (KeyboardInterrupt, EOFError):
        print("\nChat ended.")


if __name__ == "__main__":
    main()
