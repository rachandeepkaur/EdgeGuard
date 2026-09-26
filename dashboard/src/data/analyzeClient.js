// Talks to the local analysis server (integration/analyze_server.py) through
// Vite's /api proxy. The uploaded file only travels to that local process.

/**
 * Upload a .log capture and receive the real Defender output as it streams.
 * onEvent gets: {type:"upload", loaded, total} while uploading, then every
 * NDJSON event from the server ("meta" | "window" | "progress" | "done" | "error").
 * Returns an abort() function.
 */
export function analyzeCapture(file, onEvent) {
  const xhr = new XMLHttpRequest();
  let seen = 0;
  let buffer = "";

  function drain(final = false) {
    const text = xhr.responseText || "";
    buffer += text.slice(seen);
    seen = text.length;
    const lines = buffer.split("\n");
    buffer = final ? "" : lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        onEvent(JSON.parse(line));
      } catch {
        /* ignore a partial line */
      }
    }
  }

  xhr.open("POST", `/api/analyze?name=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (e) => onEvent({ type: "upload", loaded: e.loaded, total: e.total || file.size });
  xhr.onprogress = () => drain();
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      drain(true);
    } else {
      let message = `Server returned ${xhr.status}`;
      let fromServer = false;
      try {
        message = JSON.parse(xhr.responseText).detail || message;
        fromServer = true;
      } catch {
        /* not JSON: the Vite proxy could not reach the server */
      }
      const offline = !fromServer && xhr.status >= 500;
      onEvent({ type: "error", message: offline ? "Analysis server is not running." : message, offline });
    }
  };
  xhr.onerror = () => onEvent({ type: "error", message: "Analysis server is not running.", offline: true });
  xhr.send(file);
  return () => xhr.abort();
}

export async function fetchSystem(signal) {
  // Give up quickly when the server is down so the card can say "Offline".
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), 2500);
  signal?.addEventListener("abort", () => timeout.abort());
  const res = await fetch("/api/system", { signal: timeout.signal }).finally(() => clearTimeout(timer));
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}
