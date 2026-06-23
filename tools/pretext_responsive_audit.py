"""Browser-backed Pretext checks for responsive text fit.

Pretext needs a browser canvas context for measurement, so this script serves
the repo root, opens the observer in Chromium, and imports the installed
@chenglou/pretext module from node_modules.
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PRETEXT_AUDIT_OUT", "/tmp/persuasion-arena-pretext-audit.json"))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _fmt: str, *_args: object) -> None:
        return


def main() -> int:
    handler = partial(QuietHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            results = []
            for name, width, height in [
                ("desktop", 1365, 900),
                ("tablet", 900, 900),
                ("mobile", 390, 900),
            ]:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(f"{origin}/web/observer.html", wait_until="domcontentloaded")
                page.evaluate(
                    """() => {
                      document.querySelectorAll('.screen').forEach(el => el.classList.remove('on'));
                      document.querySelector('#screen-newrun').classList.add('on');
                      window.scrollTo(0, 0);
                    }"""
                )
                result = page.evaluate(
                    """async () => {
                      const { prepare, layout } = await import('/node_modules/@chenglou/pretext/dist/layout.js');
                      const px = value => {
                        const parsed = Number.parseFloat(value);
                        return Number.isFinite(parsed) ? parsed : 0;
                      };
                      const textMetrics = (el, text, maxWidth) => {
                        const style = getComputedStyle(el);
                        const fontSize = px(style.fontSize) || 13;
                        const lineHeight = px(style.lineHeight) || fontSize * 1.5;
                        const letterSpacing = style.letterSpacing === 'normal' ? 0 : px(style.letterSpacing);
                        const prepared = prepare(text, style.font, { letterSpacing });
                        const measured = layout(prepared, Math.max(1, Math.floor(maxWidth)), lineHeight);
                        return {
                          text,
                          width: Math.round(maxWidth),
                          lineCount: measured.lineCount,
                          height: Math.round(measured.height * 100) / 100,
                          font: style.font,
                        };
                      };

                      const labels = [...document.querySelectorAll('.run-settings label')].map(label => {
                        const control = label.closest('.control');
                        const box = control.getBoundingClientRect();
                        const metrics = textMetrics(label, label.textContent.trim().toUpperCase(), box.width);
                        return { ...metrics, kind: 'label', pass: metrics.lineCount <= 1 };
                      });

                      const buttons = [...document.querySelectorAll('.run-actions .btn')].map(button => {
                        const box = button.getBoundingClientRect();
                        const metrics = textMetrics(button, button.textContent.trim(), box.width - 18);
                        return { ...metrics, kind: 'button', pass: metrics.lineCount <= 1 };
                      });

                      const doc = document.documentElement;
                      return {
                        viewport: { width: window.innerWidth, height: window.innerHeight },
                        overflow: {
                          pass: doc.scrollWidth <= doc.clientWidth,
                          scrollWidth: doc.scrollWidth,
                          clientWidth: doc.clientWidth,
                        },
                        checks: [...labels, ...buttons],
                      };
                    }"""
                )
                result["name"] = name
                results.append(result)
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")

    failures = []
    for result in results:
        if not result["overflow"]["pass"]:
            failures.append(f"{result['name']}: document overflows horizontally")
        for check in result["checks"]:
            if not check["pass"]:
                failures.append(f"{result['name']}: {check['kind']} wraps: {check['text']}")

    print(f"wrote {OUT}")
    for result in results:
        print(
            f"{result['name']}: overflow={result['overflow']['pass']} "
            f"checks={sum(1 for c in result['checks'] if c['pass'])}/{len(result['checks'])}"
        )
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PRETEXT AUDIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
