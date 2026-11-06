import { TICKER } from "../products";

export function Marquee() {
  return (
    <div className="idea-strip" aria-hidden="true">
      <div className="wrap">
        {TICKER.map((name, i) => (
          <span key={name}>
            {name}
            {i < TICKER.length - 1 && <i>✳</i>}
          </span>
        ))}
      </div>
    </div>
  );
}
