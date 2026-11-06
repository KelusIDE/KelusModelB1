import { TICKER } from '../products';


export function Marquee() {
  return (
    <div className="marquee" aria-hidden="true">
      <div className="marquee-track">
        {[...TICKER, ...TICKER].map((name, i) => <span key={i}>{name}<i>✦</i></span>)}
      </div>
    </div>
  );
}
