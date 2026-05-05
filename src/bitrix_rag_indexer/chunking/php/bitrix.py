

from pathlib import Path


def build_bitrix_component_context_lines(path: Path) -> list[str]:
    component = detect_bitrix_component_context(path)
    if component is None:
        return []

    lines = [
        f"Bitrix component: {component['vendor']}:{component['name']}",
        f"Bitrix component path: {component['component_path']}",
    ]

    site_template = component.get("site_template")
    if site_template:
        lines.append(f"Bitrix site template: {site_template}")

    component_template = component.get("component_template")
    if component_template:
        lines.append(f"Bitrix component template: {component_template}")

    return lines

def detect_bitrix_component_context(path: Path) -> dict[str, str] | None:
    parts = path.as_posix().split("/")

    if len(parts) >= 3 and parts[0] == "components":
        return {
            "vendor": parts[1],
            "name": parts[2],
            "component_path": "/".join(parts[:3]),
        }

    if len(parts) >= 6 and parts[0] == "templates":
        try:
            components_index = parts.index("components")
        except ValueError:
            return None

        if len(parts) <= components_index + 2:
            return None

        component_template = (
            parts[components_index + 3]
            if len(parts) > components_index + 3
            else None
        )

        result = {
            "vendor": parts[components_index + 1],
            "name": parts[components_index + 2],
            "component_path": "/".join(parts[: components_index + 3]),
            "site_template": parts[1],
        }

        if component_template:
            result["component_template"] = component_template

        return result

    return None

