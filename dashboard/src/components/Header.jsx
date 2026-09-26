import Logo from "./Logo";

export default function Header({ dataSource, sourceLabel, onOpenUpload }) {
  return (
    <header className="app-header">
      <div className="app-header-title">
        <Logo size={48} />
        <div>
          <span className="eyebrow">Edge AI &middot; Secure AI track</span>
          <h1>EdgeGuard</h1>
          <p className="hint">CAN-bus intrusion detection</p>
        </div>
      </div>
      <div className="app-header-meta">
        {onOpenUpload && (
          <button className="btn header-upload" onClick={onOpenUpload}>
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 11 V13 H13 V11" /><path d="M8 10 V3" /><path d="M5 6 L8 3 L11 6" />
            </svg>
            Analyze capture
          </button>
        )}
        <span className="pill pill-source mono" title="Data now playing">
          {sourceLabel ? sourceLabel : "Demo replay"}
        </span>
        <span className={`pill ${dataSource === "mock" ? "pill-mock" : "pill-live"}`}>
          {dataSource === "mock" ? "MOCK DATA" : "LIVE"}
        </span>
      </div>
    </header>
  );
}
