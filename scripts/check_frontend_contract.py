import re
from collections import Counter
from pathlib import Path

FRONTEND_DIR = Path("frontend")


def collect_required_ids() -> set[str]:
    # 掃描所有 frontend JS modules，避免 feature module 的 DOM ID 漏檢。
    required_ids: set[str] = set()

    for javascript_path in sorted((FRONTEND_DIR / "js").rglob("*.js")):
        source = javascript_path.read_text(encoding="utf-8")

        required_ids.update(
            re.findall(
                r'document\.querySelector\(["\']#([^"\']+)["\']\)',
                source,
            )
        )
        required_ids.update(
            re.findall(
                r':\s*["\']#([A-Za-z][A-Za-z0-9_-]+)["\']',
                source,
            )
        )

    return required_ids


def collect_defined_ids() -> list[str]:
    html_paths = [
        FRONTEND_DIR / "index.html",
        *sorted((FRONTEND_DIR / "partials").glob("*.html")),
    ]

    defined_ids: list[str] = []

    for html_path in html_paths:
        html = html_path.read_text(encoding="utf-8")
        defined_ids.extend(re.findall(r'\bid="([^"]+)"', html))

    return defined_ids


def main() -> None:
    required_ids = collect_required_ids()
    defined_ids = collect_defined_ids()
    defined_id_set = set(defined_ids)

    missing_ids = sorted(required_ids - defined_id_set)

    duplicate_ids = sorted(
        element_id for element_id, count in Counter(defined_ids).items() if count > 1
    )

    if missing_ids:
        print("Missing frontend DOM IDs:")

        for element_id in missing_ids:
            print(f"  - {element_id}")

    if duplicate_ids:
        print("Duplicate frontend DOM IDs:")

        for element_id in duplicate_ids:
            print(f"  - {element_id}")

    if missing_ids or duplicate_ids:
        raise SystemExit(1)

    print(
        f"Frontend contract OK: {len(required_ids)} queried IDs, {len(defined_id_set)} defined IDs."
    )


if __name__ == "__main__":
    main()
