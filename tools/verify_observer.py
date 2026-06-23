"""Headless render-verify of the observer against the live server.

Loads the page, drills index -> run -> game -> player dossier, screenshots each, and fails on any
console error or missing real content.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/private/tmp/claude-501/-Users-vishaltandale-Convergence-2026-PersuationArena/edc9e026-7826-4f48-8af2-5c463cde5a7e/scratchpad")
URL = "http://localhost:8000/observer.html"


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1380, "height": 900})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))

        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_selector(".rt-row", timeout=8000)
        pg.screenshot(path=str(OUT / "v1_index.png"))
        print("index: run rows =", pg.locator(".rt-row").count())

        pg.locator(".rt-row").first.click()
        pg.wait_for_selector(".grow2", timeout=8000)
        pg.screenshot(path=str(OUT / "v2_run.png"))
        print("run overview: agents =", pg.locator(".agrow").count(), "| games =", pg.locator(".grow2").count())

        pg.locator(".grow2").first.click()
        pg.wait_for_selector(".pcard", timeout=8000)
        # step to the end so every phase renders
        for _ in range(40):
            nxt = pg.locator("#next")
            if nxt.is_disabled():
                break
            nxt.click()
        pg.screenshot(path=str(OUT / "v3_game.png"))
        msgs = pg.locator(".turn .say").count()
        print("game detail: player cards =", pg.locator(".pcard").count(), "| turns rendered =", msgs)

        # go to the Discussion phase (has per-player reasoning) and reveal a dossier
        pg.locator(".pchip").nth(1).click()
        pg.locator(".pcard").first.click()
        pg.wait_for_selector(".leadthink, .think .t", timeout=8000)
        pg.screenshot(path=str(OUT / "v4_reasoning.png"))
        if pg.locator(".leadthink").count():
            reasoning = pg.locator(".leadthink").first.inner_text()
        else:
            reasoning = pg.locator(".think .t .tx").first.inner_text()
        print("private reasoning shown:", reasoning[:90].replace("\n", " "))

        b.close()

    if errors:
        print("CONSOLE/PAGE ERRORS:")
        for e in errors[:20]:
            print("  ", e)
        sys.exit(1)
    print("VERIFY OK — no console errors; real data rendered.")


if __name__ == "__main__":
    main()
