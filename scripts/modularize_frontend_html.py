import re
from pathlib import Path

INDEX_PATH = Path("frontend/index.html")
PARTIALS_DIR = Path("frontend/partials")


def extract_top_level_sections(html: str) -> list[str]:
    # 只抽取 main 直屬 section；內層 section 會保留在父區塊中。
    token_pattern = re.compile(r"<section\b[^>]*>|</section>")
    depth = 0
    block_start: int | None = None
    sections: list[str] = []

    for match in token_pattern.finditer(html):
        token = match.group()

        if token.startswith("<section"):
            if depth == 0:
                block_start = match.start()

            depth += 1
            continue

        depth -= 1

        if depth == 0 and block_start is not None:
            sections.append(html[block_start : match.end()].strip())
            block_start = None

    if depth != 0:
        raise RuntimeError("Unbalanced <section> tags were detected.")

    return sections


def get_section_classes(section: str) -> set[str]:
    opening_tag_match = re.search(r'<section\s+class="([^"]+)"', section)

    if not opening_tag_match:
        return set()

    return set(opening_tag_match.group(1).split())


def classify_section(section: str) -> str:
    classes = get_section_classes(section)

    if "hero-card" in classes:
        return "overview"

    if classes == {"workspace-grid"}:
        return "overview"

    if "dataset-grid" in classes:
        return "overview"

    if "profile-grid" in classes:
        return "analysis"

    if "eda-section" in classes:
        return "analysis"

    if "visualization-section" in classes:
        return "analysis"

    if "ml-section" in classes:
        return "machine-learning"

    if "model-evaluation-section" in classes:
        return "machine-learning"

    if "prediction-section" in classes:
        return "machine-learning"

    if "agent-section" in classes:
        return "intelligence"

    if "ai-insight-section" in classes:
        return "intelligence"

    if "report-section" in classes:
        return "intelligence"

    raise RuntimeError(f"Unable to classify section with classes: {sorted(classes)}")


def build_navigation() -> str:
    return """
    <nav class="workspace-nav" aria-label="Workspace navigation">
      <div class="workspace-nav-inner">
        <button class="workspace-nav-button active" type="button" data-view-target="overview">
          Overview
        </button>
        <button class="workspace-nav-button" type="button" data-view-target="analysis">
          Analysis
        </button>
        <button class="workspace-nav-button" type="button" data-view-target="machine-learning">
          Machine Learning
        </button>
        <button class="workspace-nav-button" type="button" data-view-target="intelligence">
          Agent & AI
        </button>
      </div>
    </nav>
""".strip()


def build_main_shell() -> str:
    return """
    <main class="app-shell">
      <div id="overviewView" class="app-view" data-view-mount="overview"></div>
      <div
        id="analysisView"
        class="app-view"
        data-view-mount="analysis"
        hidden
      ></div>
      <div
        id="machineLearningView"
        class="app-view"
        data-view-mount="machine-learning"
        hidden
      ></div>
      <div
        id="intelligenceView"
        class="app-view"
        data-view-mount="intelligence"
        hidden
      ></div>
    </main>

    <div
      id="toastRegion"
      class="toast-region"
      aria-live="polite"
      aria-atomic="true"
    ></div>
""".strip()


def main() -> None:
    text = INDEX_PATH.read_text(encoding="utf-8")

    if 'data-view-mount="overview"' in text:
        print("Frontend HTML is already modularized.")
        return

    main_opening = '<main class="app-shell">'
    main_start = text.find(main_opening)

    if main_start == -1:
        raise RuntimeError('Unable to find <main class="app-shell">.')

    inner_start = main_start + len(main_opening)
    main_end = text.find("</main>", inner_start)

    if main_end == -1:
        raise RuntimeError("Unable to find closing </main>.")

    main_content = text[inner_start:main_end]
    sections = extract_top_level_sections(main_content)

    grouped_sections: dict[str, list[str]] = {
        "overview": [],
        "analysis": [],
        "machine-learning": [],
        "intelligence": [],
    }

    for section in sections:
        group_name = classify_section(section)
        grouped_sections[group_name].append(section)

    for group_name, group_sections in grouped_sections.items():
        if not group_sections:
            raise RuntimeError(f"No sections were assigned to partial: {group_name}")

    PARTIALS_DIR.mkdir(parents=True, exist_ok=True)

    for group_name, group_sections in grouped_sections.items():
        partial_path = PARTIALS_DIR / f"{group_name}.html"
        partial_path.write_text(
            "\n\n".join(group_sections) + "\n",
            encoding="utf-8",
        )
        print(f"Created {partial_path}")

    prefix = text[:main_start].rstrip()
    suffix = text[main_end + len("</main>") :].lstrip()

    prefix = prefix.replace(
        'src="./js/app.js"',
        'src="./js/bootstrap.js"',
    )

    new_index = prefix + "\n\n" + build_navigation() + "\n\n" + build_main_shell() + "\n\n" + suffix

    INDEX_PATH.write_text(new_index, encoding="utf-8")

    print("Updated frontend/index.html")
    print("Frontend HTML modularization completed.")


if __name__ == "__main__":
    main()
