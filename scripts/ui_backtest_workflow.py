#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_support import (  # noqa: E402
    ArtifactStore,
    AssertionResult,
    ManagedServices,
    StepResult,
    add_common_arguments,
    ensure_demo_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Playwright UI workflow backtest.")
    add_common_arguments(parser)
    args = parser.parse_args()

    ensure_demo_dataset(args.sample_file, args.python)
    seed_backtest_viewer_fixture(args.artifact_dir)
    store = ArtifactStore(args.artifact_dir, args.run_id)
    metadata: dict[str, Any] = {
        "base_url": args.base_url,
        "frontend_url": args.frontend_url,
        "sample_file": str(args.sample_file),
    }

    try:
        with ManagedServices(
            base_url=args.base_url,
            frontend_url=args.frontend_url,
            python=args.python,
            no_start=args.no_start,
            keep_servers=args.keep_servers,
        ):
            asyncio.run(run_ui_workflow(args.frontend_url, args.base_url, args.sample_file, store))
    finally:
        store.write_run_json(metadata)
        store.write_summary(metadata)

    print(f"UI backtest artifacts: {store.run_dir}")


async def run_ui_workflow(
    frontend_url: str,
    base_url: str,
    sample_file: Path,
    store: ArtifactStore,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Python Playwright is not installed. "
            "Run: .venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    async with async_playwright() as playwright:
        reset_workspace_state(base_url)
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        requests: list[str] = []
        responses: list[dict[str, Any]] = []
        page_errors: list[str] = []
        console_errors: list[str] = []

        page.on("request", lambda request: requests.append(request.url))
        page.on(
            "response",
            lambda response: responses.append({"url": response.url, "status": response.status}),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )

        try:
            await run_step(
                store,
                "ui_open_upload",
                lambda: open_upload_page(page, frontend_url, base_url, store),
            )
            await assert_next_action(page, store, "upload-dataset", "required")
            assert_lazy_initial_state(store, requests)

            await run_step(
                store,
                "ui_upload_dataset",
                lambda: upload_dataset(page, sample_file, store),
            )
            await assert_next_action(page, store, "inspect-schema", "required")
            await run_step(
                store,
                "ui_schema_route",
                lambda: navigate_and_capture(page, "analyze-schema", "#schemaSummary", store),
            )
            await assert_next_action(page, store, "run-ml-experiment", "required")
            await run_step(
                store,
                "ui_model_workbench_route",
                lambda: navigate_and_capture(page, "model-workbench", "#mlTaskTypeSelect", store),
            )
            await assert_next_action(page, store, "run-ml-experiment", "required")
            await run_step(
                store,
                "ui_lifecycle_route",
                lambda: navigate_and_capture(page, "lifecycle-drift", "#driftReportOutput", store),
            )
            await run_step(
                store,
                "ui_agent_route",
                lambda: navigate_and_capture(page, "agent-workflow", "#agentGoalInput", store),
            )
            await run_step(
                store,
                "ui_guided_workflow_states",
                lambda: assert_guided_workflow_states(page, store),
            )
            await run_step(
                store,
                "ui_backtest_viewer_route",
                lambda: assert_backtest_viewer(page, store),
            )

            partials = [url.split("/partials/")[1] for url in requests if "/partials/" in url]
            feature_modules = [
                url.split("/js/features/")[1] for url in requests if "/js/features/" in url
            ]
            network_errors = [response for response in responses if response["status"] >= 400]
            errors = filter_console_errors(console_errors)
            store.write_payload(
                "ui_network_summary",
                {
                    "partials": partials,
                    "feature_modules": feature_modules,
                    "page_errors": page_errors,
                    "console_errors": errors,
                    "network_errors": network_errors,
                },
            )

            store.assert_that(
                "ui_has_no_page_errors",
                not page_errors,
                expected=[],
                actual=page_errors,
            )
            store.assert_that(
                "ui_has_no_console_errors",
                not errors,
                expected=[],
                actual=errors,
            )
            store.assert_that(
                "ui_has_no_network_errors",
                not network_errors,
                expected=[],
                actual=network_errors[:10],
            )
            for screenshot in (
                "upload",
                "uploaded",
                "analyze-schema",
                "model-workbench",
                "lifecycle-drift",
                "agent-workflow",
                "lifecycle-backtests",
            ):
                assert_screenshot_file(store, screenshot)
        finally:
            await browser.close()


async def run_step(store: ArtifactStore, name: str, action: Any) -> None:
    import time

    started_at = time.perf_counter()
    try:
        await action()
        store.add_step(
            StepResult(
                name=name, status="success", duration_seconds=time.perf_counter() - started_at
            )
        )
    except Exception as exc:
        store.add_step(
            StepResult(
                name=name,
                status="failed",
                duration_seconds=time.perf_counter() - started_at,
                error=str(exc),
            )
        )
        raise


async def open_upload_page(
    page: Any, frontend_url: str, base_url: str, store: ArtifactStore
) -> None:
    await page.goto(f"{frontend_url}/?apiBaseUrl={base_url}#data-upload", wait_until="networkidle")
    await page.wait_for_selector("#uploadForm")
    await assert_guide_tour(page, store)
    await page.click('[data-view-target="data-upload"]')
    await page.wait_for_selector("#uploadForm")
    await page.screenshot(path=store.screenshot_path("upload"), full_page=True)
    assert_screenshot_file(store, "upload")
    state = await page.evaluate(
        """() => ({
          title: document.querySelector('#routeTitle')?.textContent?.trim(),
          dataLoaded: Boolean(document.querySelector('#uploadForm')),
          modelLoaded: Boolean(document.querySelector('#mlTaskTypeSelect')),
          timelineSteps: document.querySelectorAll('#workflowTimeline .workflow-step').length,
          bootstrapErrors: document.querySelectorAll('.bootstrap-error-panel').length,
        })"""
    )
    if state["title"] != "Upload" or state["bootstrapErrors"]:
        raise RuntimeError(f"Unexpected initial UI state: {state}")


async def assert_guide_tour(page: Any, store: ArtifactStore) -> None:
    await page.wait_for_selector("[data-guide-tour-panel]", timeout=15000)
    initial_state = await page.evaluate(
        """() => {
          const stepCountText = document.querySelector(
            '[data-guide-step-count]'
          )?.textContent || '';
          const totalSteps = Number(stepCountText.split('/')[1]?.trim() || 0);
          return {
            visible: Boolean(document.querySelector('[data-guide-tour-panel]')),
            title: document.querySelector('[data-guide-title]')?.textContent?.trim(),
            totalSteps,
          };
        }"""
    )
    store.assert_that(
        "ui_guide_tour_auto_starts",
        initial_state["visible"] and initial_state["totalSteps"] >= 17,
        expected="visible guide with at least 17 steps",
        actual=initial_state,
    )

    for _ in range(4):
        await page.click('[data-guide-action="next"]')

    await page.wait_for_function(
        """() => document.querySelector('.sidebar-nav-button.active')?.dataset.viewTarget
          === 'data-preview'""",
        timeout=30000,
    )
    routed_state = await page.evaluate(
        """() => ({
          route: document.querySelector('.sidebar-nav-button.active')?.dataset.viewTarget,
          title: document.querySelector('[data-guide-title]')?.textContent?.trim(),
          frameWidth: document.querySelector('[data-guide-tour-frame]')
            ?.getBoundingClientRect().width,
        })"""
    )
    store.assert_that(
        "ui_guide_tour_cross_route_step",
        routed_state["route"] == "data-preview" and routed_state["frameWidth"] > 20,
        expected={"route": "data-preview", "frameWidth": "> 20"},
        actual=routed_state,
    )

    await page.click('[data-guide-action="skip"]')
    await page.wait_for_selector("[data-guide-tour-panel]", state="hidden", timeout=10000)
    await page.wait_for_selector("[data-guide-tour-launcher]", timeout=10000)
    await page.click("[data-guide-tour-launcher]")
    await page.wait_for_selector("[data-guide-tour-panel]", timeout=10000)
    await page.click('[data-guide-action="skip"]')
    await page.wait_for_selector("[data-guide-tour-panel]", state="hidden", timeout=10000)


async def upload_dataset(page: Any, sample_file: Path, store: ArtifactStore) -> None:
    await page.set_input_files("#csvFileInput", str(sample_file.resolve()))
    await page.click("#uploadForm button[type='submit']")
    dataset_loaded_expression = """
      () => {
        const datasetName = document.querySelector('#contextDatasetName')?.textContent?.trim();
        return datasetName && datasetName !== 'No dataset selected';
      }
    """
    await page.wait_for_function(
        dataset_loaded_expression,
        timeout=60000,
    )
    await page.screenshot(path=store.screenshot_path("uploaded"), full_page=True)
    assert_screenshot_file(store, "uploaded")
    dataset_name = await page.text_content("#contextDatasetName")
    if not dataset_name or dataset_name.strip() == "No dataset selected":
        raise RuntimeError("Dataset context did not update after upload.")


async def navigate_and_capture(page: Any, route: str, selector: str, store: ArtifactStore) -> None:
    await page.click(f'[data-view-target="{route}"]')
    await page.wait_for_selector(selector, timeout=60000)
    route_active_expression = f"""
      () => {{
        const title = document.querySelector('#routeTitle')?.textContent?.trim();
        const activeRoute = document.querySelector(
          '.sidebar-nav-button.active'
        )?.dataset.viewTarget;
        return title && activeRoute === '{route}';
      }}
    """
    await page.wait_for_function(
        route_active_expression,
        timeout=60000,
    )
    await page.screenshot(path=store.screenshot_path(route), full_page=True)
    assert_screenshot_file(store, route)


async def assert_guided_workflow_states(page: Any, store: ArtifactStore) -> None:
    await render_context_for_test(
        page,
        {
            "dataset": dataset_context(),
            "version": {"version_id": "v1"},
            "targetColumn": "churn",
            "models": [model_context("production")],
            "selectedModelId": "model-1",
            "workflowFlags": {
                "schemaReviewed": True,
                "trainingCompleted": True,
                "evaluationCompleted": True,
                "explainabilityCompleted": True,
            },
            "driftStatus": "Not checked",
        },
    )
    await assert_next_action(page, store, "run-drift-check", "recommended")

    await render_context_for_test(
        page,
        {
            "dataset": dataset_context(),
            "version": {"version_id": "v2"},
            "targetColumn": "churn",
            "models": [model_context("production")],
            "selectedModelId": "model-1",
            "workflowFlags": {
                "schemaReviewed": True,
                "trainingCompleted": True,
                "evaluationCompleted": True,
                "explainabilityCompleted": True,
                "driftChecked": True,
            },
            "driftStatus": "drift",
            "driftReportId": "drift-1",
        },
    )
    await assert_next_action(page, store, "build-retrain-plan", "required")
    await assert_next_action(page, store, "retrain-challenger", "recommended")

    await render_context_for_test(
        page,
        {
            "dataset": dataset_context(),
            "version": {"version_id": "v2"},
            "targetColumn": "churn",
            "models": [
                model_context("production"),
                {
                    **model_context("candidate"),
                    "id": "candidate-1",
                    "model_name": "logistic_regression_challenger",
                },
            ],
            "selectedModelId": "model-1",
            "workflowFlags": {
                "schemaReviewed": True,
                "trainingCompleted": True,
                "evaluationCompleted": True,
                "explainabilityCompleted": True,
                "driftChecked": True,
                "retrainCandidateCreated": True,
            },
            "driftStatus": "stable",
            "driftReportId": "drift-1",
            "retrainCandidateId": "candidate-1",
        },
    )
    await assert_next_action(page, store, "run-migration-check", "recommended")
    await assert_next_action(page, store, "promote-challenger", "optional")


async def assert_backtest_viewer(page: Any, store: ArtifactStore) -> None:
    await page.click('[data-view-target="lifecycle-backtests"]')
    await page.wait_for_selector("#backtestRunList", timeout=60000)
    await page.wait_for_selector(
        '[data-backtest-run-id="viewer-fixture-run"]',
        timeout=60000,
    )
    await page.click('[data-backtest-run-id="viewer-fixture-run"]')
    await page.wait_for_selector("#backtestRunDetail", timeout=60000)
    await page.wait_for_selector(
        '[data-backtest-payload-name="viewer_payload.json"]',
        timeout=60000,
    )
    await page.click('[data-backtest-payload-name="viewer_payload.json"]')
    await page.wait_for_function(
        """() => document.querySelector('#backtestPayloadViewer')?.textContent
          ?.includes('fixture_payload')""",
        timeout=60000,
    )
    await page.screenshot(path=store.screenshot_path("lifecycle-backtests"), full_page=True)
    assert_screenshot_file(store, "lifecycle-backtests")
    state = await page.evaluate(
        """() => ({
          title: document.querySelector('#routeTitle')?.textContent?.trim(),
          selectedRun: document.querySelector('.backtest-run-card.active')?.dataset.backtestRunId,
          payloadLoaded: document.querySelector('#backtestPayloadViewer')?.textContent
            ?.includes('fixture_payload'),
          runnerSuite: document.querySelector('#backtestSuiteTypeSelect')?.value,
          runnerButton: Boolean(document.querySelector('#startBacktestJobButton')),
          jobLog: Boolean(document.querySelector('#backtestJobLog')),
        })"""
    )
    store.assert_that(
        "ui_backtest_viewer_loaded",
        state["title"] == "Backtest Runs" and state["selectedRun"] == "viewer-fixture-run",
        expected={"title": "Backtest Runs", "selectedRun": "viewer-fixture-run"},
        actual=state,
    )
    store.assert_that(
        "ui_backtest_payload_loaded",
        bool(state["payloadLoaded"]),
        expected=True,
        actual=state,
    )
    store.assert_that(
        "ui_backtest_runner_controls_visible",
        state["runnerSuite"] == "api" and state["runnerButton"] and state["jobLog"],
        expected={"runnerSuite": "api", "runnerButton": True, "jobLog": True},
        actual=state,
    )


async def render_context_for_test(page: Any, context: dict[str, Any]) -> None:
    await page.evaluate(
        """(context) => {
          window.dispatchEvent(new CustomEvent('dataagent:context-changed', { detail: context }));
        }""",
        context,
    )


async def assert_next_action(
    page: Any,
    store: ArtifactStore,
    action_id: str,
    priority: str,
) -> None:
    selector = f'[data-next-action-id="{action_id}"]'
    await page.wait_for_selector(selector, timeout=10000)
    action = await page.eval_on_selector(
        selector,
        """(element) => ({
          id: element.dataset.nextActionId,
          priority: element.dataset.nextActionPriority,
          status: element.dataset.nextActionStatus,
          route: element.dataset.viewTarget,
          label: element.querySelector('span')?.textContent?.trim(),
        })""",
    )
    store.assert_that(
        f"ui_next_action_{action_id}",
        action["priority"] == priority,
        expected=priority,
        actual=action,
    )


def dataset_context() -> dict[str, Any]:
    return {
        "id": "dataset-1",
        "name": "guided-workflow.csv",
        "row_count": 120,
        "column_count": 14,
        "latest_version_id": "v2",
        "status": "ready",
    }


def model_context(lifecycle_status: str) -> dict[str, Any]:
    return {
        "id": "model-1",
        "model_name": "logistic_regression",
        "task_type": "classification",
        "target_column": "churn",
        "lifecycle_status": lifecycle_status,
    }


def assert_lazy_initial_state(store: ArtifactStore, requests: list[str]) -> None:
    partials = [url.split("/partials/")[1] for url in requests if "/partials/" in url]
    store.assert_that(
        "ui_lazy_loaded_overview_partial",
        "overview.html" in partials,
        expected="overview.html",
        actual=partials,
    )
    store.assert_that(
        "ui_lazy_did_not_preload_model_partial",
        "machine-learning.html" not in partials,
        expected="machine-learning.html absent",
        actual=partials,
    )


def assert_screenshot_file(store: ArtifactStore, name: str) -> None:
    path = store.screenshot_path(name)
    store.assert_that(
        f"ui_screenshot_{name}_exists",
        path.exists() and path.stat().st_size > 0,
        expected="non-empty screenshot",
        actual=str(path),
    )


def filter_console_errors(errors: list[str]) -> list[str]:
    ignored = ("favicon", "ERR_CONNECTION_REFUSED", "Failed to load resource")
    return [error for error in errors if not any(pattern in error for pattern in ignored)]


def reset_workspace_state(base_url: str) -> None:
    response = httpx.patch(
        f"{base_url.rstrip('/')}/api/workspaces/default/state",
        json={
            "active_route": "data-upload",
            "dataset_id": None,
            "dataset_version_id": None,
            "target_column": None,
            "selected_model_id": None,
            "drift_report_id": None,
            "drift_status": "Not checked",
            "workflow_flags": {},
            "retrain_candidate_id": None,
        },
        timeout=10,
    )
    response.raise_for_status()


def seed_backtest_viewer_fixture(artifact_dir: Path) -> None:
    fixture = ArtifactStore(artifact_dir, run_id="viewer-fixture-run")
    payload_path = fixture.write_payload(
        "viewer_payload",
        {
            "fixture_payload": True,
            "purpose": "Backtest Runs viewer regression fixture",
        },
    )
    fixture.screenshot_path("viewer_upload").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJ5"
            "rgXJYQAAAABJRU5ErkJggg=="
        )
    )
    fixture.add_step(
        StepResult(
            name="viewer_fixture_step",
            status="success",
            duration_seconds=0.01,
            payload_path=payload_path,
            metadata={"route": "lifecycle-backtests"},
        )
    )
    fixture.add_assertion(
        AssertionResult(
            name="viewer_fixture_assertion",
            status="success",
            expected=True,
            actual=True,
            message="Fixture assertion visible in UI.",
        )
    )
    metadata = {
        "suite": "ui_viewer_fixture",
        "description": "Deterministic artifact for the Backtest Runs UI route.",
    }
    fixture.write_run_json(metadata)
    fixture.write_summary(metadata)


if __name__ == "__main__":
    main()
