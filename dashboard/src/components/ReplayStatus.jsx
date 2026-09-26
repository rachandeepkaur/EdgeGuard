import { REAL_TIME } from "../utils/useReplay";
import { useClock, driveSeconds, formatHms } from "../utils/clock";

// Compact replay control bar: play/pause, reset, replay indicator + clock,
// position and drive time, scrubbable progress, speed and alert sound, all
// on one small row.
const SPEEDS = [
  { value: REAL_TIME, label: "Real-time", title: "1 window per second: the rate data actually arrives from the car" },
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" },
];

export default function ReplayStatus({
  captureId, windowId, index, total, isPlaying, speed,
  onPlay, onPause, onReset, onSpeed, onSeek, soundOn, onToggleSound,
}) {
  const now = useClock();
  function scrub(e) {
    if (!onSeek || !total) return;
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - rect.left) / rect.width) * (total - 1));
  }
  const pct = total ? Math.round(((index + 1) / total) * 100) : 0;

  return (
    <div className="replay-bar" role="group" aria-label="Replay controls">
      <button
        onClick={isPlaying ? onPause : onPlay}
        className="icon-btn icon-btn-primary"
        aria-label={isPlaying ? "Pause" : "Play"}
        title={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? (
          <svg viewBox="0 0 16 16" width="14" height="14"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
        ) : (
          <svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 2.5 L13.5 8 L4 13.5 Z" /></svg>
        )}
      </button>
      <button onClick={onReset} className="icon-btn" aria-label="Reset" title="Reset">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
          <path d="M3 8 A5 5 0 1 0 5 4" /><path d="M2.5 2.5 V5.5 H5.5" />
        </svg>
      </button>

      <span
        className={`replay-indicator ${isPlaying ? "replay-indicator-on" : ""}`}
        title="Replaying recorded Defender output; the clock is your local time"
      >
        <span className="replay-dot" aria-hidden="true" />
        REPLAY
        <span className="mono replay-clock">{now.toLocaleTimeString([], { hour12: false })}</span>
      </span>

      <span className="replay-pos mono">
        {captureId} &middot; <span title="Drive time into this recording (1 window = 1 s)">{formatHms(driveSeconds(windowId, index))}</span> &middot; {index + 1}/{total}
      </span>

      <div
        className="progress-track progress-scrub"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        title="Click to jump"
        onClick={scrub}
      >
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="speed-group" aria-label="Playback speed">
        {SPEEDS.map((s) => (
          <button
            key={s.label}
            className={`btn btn-chip ${speed === s.value ? "btn-chip-active" : ""}`}
            onClick={() => onSpeed(s.value)}
            title={s.title}
          >
            {s.label}
          </button>
        ))}
      </div>

      {onToggleSound && (
        <button
          onClick={onToggleSound}
          className={`icon-btn ${soundOn ? "icon-btn-on" : ""}`}
          aria-label={soundOn ? "Mute alert sound" : "Turn on alert sound"}
          aria-pressed={soundOn}
          title={soundOn ? "Alert sound on" : "Alert sound off"}
        >
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2.5 6 H5 L8.5 3 V13 L5 10 H2.5 Z" fill="currentColor" stroke="none" />
            {soundOn ? (
              <><path d="M10.5 5.5 A3.5 3.5 0 0 1 10.5 10.5" /><path d="M12.3 3.8 A6 6 0 0 1 12.3 12.2" /></>
            ) : (
              <><path d="M10.5 6 L14 10" /><path d="M14 6 L10.5 10" /></>
            )}
          </svg>
        </button>
      )}
    </div>
  );
}
