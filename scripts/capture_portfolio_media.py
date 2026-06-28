#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_support import ManagedServices, ensure_demo_dataset  # noqa: E402

SCREENSHOTS = [
    ("01-overview", "data-upload", "#uploadForm"),
    ("02-guide-sidebar", "data-upload", "[data-guide-tour-panel]"),
    ("03-guide-preview", "data-preview", "#previewTable"),
    ("04-dataset-preview", "data-preview", "#previewTable"),
    ("05-dataset-versions", "data-versions", "#datasetVersionsPanel"),
    ("06-schema", "analyze-schema", "#schemaSummary"),
    ("07-eda", "analyze-eda", "#edaQualitySummary"),
    ("08-visualization-lab", "analyze-visualization", "#visualizationLabSummary"),
    ("09-ml-workbench", "model-workbench", "#mlTaskTypeSelect"),
    ("10-model-registry", "model-registry", "#modelLeaderboard"),
    ("11-prediction", "model-prediction", "#predictionJsonInput"),
    ("12-explainability", "model-explainability", "#explainabilityOverview"),
    ("13-drift-center", "lifecycle-drift", "#driftReportOutput"),
    ("14-reports", "lifecycle-reports", "#reportViewer"),
    ("15-backtests", "lifecycle-backtests", "#backtestRunList"),
    ("16-agent-jobs", "agent-workflow", "#agentGoalInput"),
    ("17-ai-insights", "agent-insights", "#aiInsightOutput"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture screenshots and guide-tour video for the portfolio.",
    )
    parser.add_argument(
        "--portfolio-dir",
        type=Path,
        default=Path("../justin-portfolio"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:5174")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path("data/samples/customer_churn_demo.csv"),
    )
    parser.add_argument("--no-start", action="store_true")
    args = parser.parse_args()

    ensure_demo_dataset(args.sample_file, args.python)

    portfolio_project_dir = (
        args.portfolio_dir.resolve() / "public" / "projects" / "dataagent-analyst"
    )
    screenshot_dir = portfolio_project_dir / "screenshots"
    video_dir = portfolio_project_dir / "videos"
    poster_dir = video_dir / "posters"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    poster_dir.mkdir(parents=True, exist_ok=True)

    with ManagedServices(
        base_url=args.base_url,
        frontend_url=args.frontend_url,
        python=args.python,
        no_start=args.no_start,
        keep_servers=False,
    ):
        asyncio.run(
            capture(
                args.frontend_url,
                args.base_url,
                args.sample_file,
                screenshot_dir,
                video_dir,
            ),
        )

    print(f"Portfolio screenshots written to {screenshot_dir}")
    print(f"Portfolio video written to {video_dir / 'playwright-external-live-demo.webm'}")


async def capture(
    frontend_url: str,
    base_url: str,
    sample_file: Path,
    screenshot_dir: Path,
    video_dir: Path,
) -> None:
    from playwright.async_api import async_playwright

    with tempfile.TemporaryDirectory() as temp_dir:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1000},
                record_video_dir=temp_dir,
                record_video_size={"width": 1440, "height": 1000},
            )
            page = await context.new_page()
            page.set_default_timeout(60000)
            await run_demo_flow(page, frontend_url, base_url, sample_file, screenshot_dir)
            video = page.video
            await context.close()
            await browser.close()

            if video:
                video_path = await video.path()
                shutil.copyfile(video_path, video_dir / "playwright-external-live-demo.webm")


async def run_demo_flow(
    page: Any,
    frontend_url: str,
    base_url: str,
    sample_file: Path,
    screenshot_dir: Path,
) -> None:
    await page.goto(f"{frontend_url}/?apiBaseUrl={base_url}#data-upload", wait_until="networkidle")
    await page.wait_for_selector("#uploadForm")
    await page.wait_for_selector("[data-guide-tour-panel]")
    await page.screenshot(path=screenshot_dir / "01-overview.png", full_page=True)
    await page.screenshot(path=screenshot_dir / "02-guide-sidebar.png", full_page=True)

    for _ in range(4):
        await page.click('[data-guide-action="next"]')
    await page.wait_for_selector("#previewTable")
    await page.screenshot(path=screenshot_dir / "03-guide-preview.png", full_page=True)
    await page.click('[data-guide-action="skip"]')
    await page.wait_for_selector("[data-guide-tour-panel]", state="hidden")

    await page.click('[data-view-target="data-upload"]')
    await page.wait_for_selector("#uploadForm")
    await page.set_input_files("#csvFileInput", str(sample_file.resolve()))
    await page.click("#uploadForm button[type='submit']")
    await page.wait_for_function(
        """() => document.querySelector('#contextDatasetName')?.textContent?.trim()
          !== 'No dataset selected'""",
    )

    for name, route, selector in SCREENSHOTS:
        if name in {"01-overview", "02-guide-sidebar", "03-guide-preview"}:
            continue
        await page.click(f'[data-view-target="{route}"]')
        await page.wait_for_selector(selector)
        await stabilize_route(page, route)
        await trigger_route_action(page, route)
        await page.screenshot(path=screenshot_dir / f"{name}.png", full_page=True)

    await page.set_viewport_size({"width": 390, "height": 900})
    await page.click("[data-guide-tour-launcher]")
    await page.wait_for_selector("[data-guide-tour-panel]")
    await page.screenshot(path=screenshot_dir / "18-mobile-guide.png", full_page=True)


async def stabilize_route(page: Any, route: str) -> None:
    await page.wait_for_function(
        """(route) => {
            return document.querySelector('.sidebar-nav-button.active')
              ?.dataset.viewTarget === route;
        }""",
        arg=route,
    )
    await page.wait_for_timeout(500)


async def trigger_route_action(page: Any, route: str) -> None:
    actions = {
        "model-workbench": "#generateMlPlanButton",
        "model-registry": "#promoteModelButton",
        "model-prediction": "#runPredictionButton",
        "model-explainability": "#runExplainabilityButton",
        "lifecycle-drift": "#runDriftButton",
        "lifecycle-reports": "#generateReportButton",
        "lifecycle-backtests": "#startBacktestJobButton",
        "agent-workflow": "#runAgentWorkflowButton",
        "agent-insights": "#generateEdaInsightButton",
    }
    selector = actions.get(route)

    if not selector:
        return

    button = page.locator(selector)
    if await button.count() == 0:
        return

    await button.first.click()
    await page.wait_for_timeout(900)


if __name__ == "__main__":
    main()
