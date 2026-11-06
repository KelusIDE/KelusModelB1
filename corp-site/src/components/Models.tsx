import { MODELS } from "../products";
import { CountUp } from "./CountUp";

export function Models() {
  return (
    <ul className="models">
      {MODELS.map((model, i) => (
        <li key={model.name}>
          <span className="model-index">0{i + 1}</span>
          <div>
            <h3>{model.name}</h3>
            <p>{model.about}</p>
          </div>
          {"count" in model.stat ? (
            <span
              className="stat"
              aria-label={`${model.stat.count} million parameters`}
            >
              <span>
                <CountUp to={model.stat.count} />M
              </span>
              <small>PARAMETERS</small>
            </span>
          ) : (
            <span className="stat">{model.stat.label}</span>
          )}
        </li>
      ))}
      <li className="models-bottom">
        <span className="live-dot" />
        Own weights. Open experiments.
      </li>
    </ul>
  );
}
