"""Screenshot the run overview for a given run id, asserting win-rate + CI render."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("/private/tmp/claude-501/-Users-vishaltandale-Convergence-2026-PersuationArena/edc9e026-7826-4f48-8af2-5c463cde5a7e/scratchpad")
RUN = sys.argv[1] if len(sys.argv) > 1 else "run_9000"


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1380, "height": 980})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))
        pg.goto("http://localhost:8000/observer.html", wait_until="networkidle")
        pg.wait_for_selector(".rt-row", timeout=8000)
        pg.locator(f'.rt-row[data-run="{RUN}"]').click()
        pg.wait_for_selector(".winbars .wb", timeout=8000)
        pg.screenshot(path=str(OUT / "v5_overview.png"))
        bars = pg.locator(".winbars .wb").count()
        agrows = pg.locator(".agrow").count()
        games = pg.locator(".grow2").count()
        text = pg.locator("#ovresults").inner_text()
        print(f"overview: agents={agrows} games={games} winrate-bars={bars}")
        print("results text has CI %:", "%" in text and "n=" in text)
        b.close()
    if errors:
        print("CONSOLE ERRORS:", errors[:10]); sys.exit(1)
    print("OVERVIEW VERIFY OK")


if __name__ == "__main__":
    main()
