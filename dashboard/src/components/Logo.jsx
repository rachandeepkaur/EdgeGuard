// EdgeGuard mark: a shield (the guard) over a CAN bus line with three ECU
// nodes. Two nodes are normal (blue/cyan); the middle one is the intruder
// the defender catches (red).
export default function Logo({ size = 40 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className="app-logo">
      <defs>
        <linearGradient id="egShield" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <path
        d="M20 3 L34 8.5 V19 C34 27.5 28.2 33.6 20 37 C11.8 33.6 6 27.5 6 19 V8.5 Z"
        fill="rgba(59,130,246,0.14)"
        stroke="url(#egShield)"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
      <line x1="10" y1="20" x2="30" y2="20" stroke="#22d3ee" strokeWidth="1.8" strokeLinecap="round" />
      <line x1="13" y1="20" x2="13" y2="14.5" stroke="#22d3ee" strokeWidth="1.4" />
      <line x1="27" y1="20" x2="27" y2="14.5" stroke="#22d3ee" strokeWidth="1.4" />
      <line x1="20" y1="20" x2="20" y2="25.5" stroke="#f04452" strokeWidth="1.4" />
      <circle cx="13" cy="13.5" r="2.4" fill="#3b82f6" />
      <circle cx="27" cy="13.5" r="2.4" fill="#3b82f6" />
      <circle cx="20" cy="26.5" r="2.6" fill="#f04452" className="app-logo-intruder" />
    </svg>
  );
}
