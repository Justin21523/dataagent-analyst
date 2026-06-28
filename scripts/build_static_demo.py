#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the static GitHub Pages demo for DataAgent Analyst.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dist" / "github-pages",
        help="Directory where the static demo should be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    frontend_dir = PROJECT_ROOT / "frontend"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copytree(frontend_dir, output_dir)
    inject_demo_script(output_dir / "index.html")
    write_backtest_screenshot_fixtures(output_dir)
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Static demo written to {output_dir}")


def inject_demo_script(index_path: Path) -> None:
    html = index_path.read_text(encoding="utf-8")
    bootstrap_tag = '<script type="module" src="./js/bootstrap.js"></script>'
    demo_tag = '<script type="module" src="./js/demo/mockApi.js"></script>'

    if demo_tag in html:
        return

    if bootstrap_tag not in html:
        raise RuntimeError("Unable to locate bootstrap module script in frontend/index.html")

    html = html.replace(bootstrap_tag, f"{demo_tag}\n    {bootstrap_tag}")
    index_path.write_text(html, encoding="utf-8")


def write_backtest_screenshot_fixtures(output_dir: Path) -> None:
    screenshot_dir = (
        output_dir / "api" / "backtests" / "runs" / "playwright-demo-run" / "screenshots"
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5"
        "rgXJYQAAAABJRU5ErkJggg=="
    )
    for name in ["upload", "schema", "ml-workbench", "guide-tour"]:
        (screenshot_dir / f"{name}.png").write_bytes(png_bytes)


if __name__ == "__main__":
    main()
