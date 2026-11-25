// Semicircle gauge for the model's probability. The scale runs 0-50% because real scores sit
// between ~1% and ~60%; a 0-100% dial would make every caption look the same. Markers show the
// typical post (base rate) and the "likely" cut-off. The number is always printed as text too.
export const GAUGE_MAX = 0.5;

const R = 80;
const CX = 100;
const CY = 96;

function point(frac) {
  const a = Math.PI * (1 - Math.min(Math.max(frac, 0), 1));
  return [CX + R * Math.cos(a), CY - R * Math.sin(a)];
}

function arc(from, to) {
  const [x1, y1] = point(from);
  const [x2, y2] = point(to);
  return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`;
}

function Tick({ frac, label, className }) {
  const [x1, y1] = point(frac);
  const a = Math.PI * (1 - frac);
  const x2 = CX + (R + 19) * Math.cos(a);
  const y2 = CY - (R + 19) * Math.sin(a);
  return (
    <g className={className}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} />
      <text x={x2} y={y2 - 4} textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

export default function ScoreGauge({ probability, baseRate, threshold }) {
  const frac = Math.min(probability / GAUGE_MAX, 1);
  const pct = Math.round(probability * 1000) / 10;
  return (
    <figure className="gauge">
      <svg
        viewBox="0 -8 200 126"
        role="img"
        aria-label={`Estimated chance ${pct}%. Typical post ${Math.round(baseRate * 100)}%.`}
      >
        <path d={arc(0, 1)} className="gauge-track" />
        {frac > 0.002 && <path d={arc(0, frac)} className="gauge-fill" />}
        <Tick frac={baseRate / GAUGE_MAX} label="typical" className="gauge-tick" />
        <Tick
          frac={threshold / GAUGE_MAX}
          label="likely"
          className="gauge-tick gauge-tick-strong"
        />
        <text x={CX} y={CY - 14} textAnchor="middle" className="gauge-value">
          {pct}%
        </text>
        <text x={CX} y={CY + 6} textAnchor="middle" className="gauge-caption">
          chance of top 10%
        </text>
        <text x={CX - R} y={CY + 16} textAnchor="middle" className="gauge-scale">
          0%
        </text>
        <text x={CX + R} y={CY + 16} textAnchor="middle" className="gauge-scale">
          50%+
        </text>
      </svg>
    </figure>
  );
}
