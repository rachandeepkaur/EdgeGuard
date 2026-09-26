// Soft two-tone "ping" for the alert flash, synthesized with Web Audio (no
// audio file). Off by default; the AudioContext is only created after the
// viewer turns sound on, which browsers require to be a user gesture.
let ctx = null;

export function unlockAlertSound() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return false;
  ctx = ctx ?? new Ctx();
  if (ctx.state === "suspended") ctx.resume();
  return true;
}

export function playAlertPing() {
  if (!ctx) return;
  const t = ctx.currentTime;
  [880, 1320].forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    const start = t + i * 0.09;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.06, start + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.25);
    osc.connect(gain).connect(ctx.destination);
    osc.start(start);
    osc.stop(start + 0.26);
  });
}
