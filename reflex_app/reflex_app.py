"""Polished Reflex frontend for Eye-Heart Connection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import time

import tempfile

import httpx
import reflex as rx

API_BASE = os.getenv("EHC_API_BASE", "http://localhost:8000")
HF_MEDIA_BASE = "https://huggingface.co/datasets/ayushsainime/eye_heart_connect_media/resolve/main"
ANIMATION_SRC = f"{HF_MEDIA_BASE}/animation_reflex.mp4"

# Vibrant Color Palette
PURPLE_VIBRANT = "#8B5CF6"
BLUE_VIBRANT = "#3B82F6"
TEAL_VIBRANT = "#14B8A6"
GREEN_VIBRANT = "#10B981"
ORANGE_VIBRANT = "#F97316"
PINK_VIBRANT = "#EC4899"
RED_VIBRANT = "#EF4444"
INDIGO_VIBRANT = "#6366F1"
CYAN_VIBRANT = "#06B6D4"
AMBER_VIBRANT = "#F59E0B"

# Gradient backgrounds for cards
GRADIENT_PURPLE = "linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%)"
GRADIENT_BLUE = "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%)"
GRADIENT_GREEN = "linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(20, 184, 166, 0.08) 100%)"
GRADIENT_ORANGE = "linear-gradient(135deg, rgba(249, 115, 22, 0.12) 0%, rgba(245, 158, 11, 0.08) 100%)"
GRADIENT_PINK = "linear-gradient(135deg, rgba(236, 72, 153, 0.12) 0%, rgba(244, 63, 94, 0.08) 100%)"
GRADIENT_TEAL = "linear-gradient(135deg, rgba(20, 184, 166, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%)"

# Text colors
TEXT_MAIN = rx.color("gray", 12)
TEXT_MUTED = rx.color("gray", 11)
TEXT_WHITE = "#FFFFFF"

# Card backgrounds with vibrancy
CARD_BG_GLASS = "rgba(255, 255, 255, 0.85)"
CARD_BG_WHITE = "rgba(255, 255, 255, 0.95)"

LABEL_DESCRIPTIONS = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other",
}

INDICATOR_INFO = [
    ("N", "Normal", "Represents baseline retinal structure with no strong pathological signal."),
    ("D", "Diabetes", "Diabetic retinal damage can indicate broader vascular stress in the body."),
    ("G", "Glaucoma", "Optic nerve pressure-related changes can impact long-term visual health."),
    ("C", "Cataract", "Lens opacity can reduce image quality and indicate age-related degeneration."),
    ("A", "Age-related Macular Degeneration", "Macular damage can reflect microvascular decline."),
    ("H", "Hypertension", "Retinal vessel narrowing and hemorrhage can track blood pressure burden."),
    ("M", "Myopia", "Elongated eye shape may alter retinal structure and vessel geometry."),
    ("O", "Other Findings", "Captures additional retinal anomalies not grouped in the core classes."),
]


def _discover_sample_pairs(max_cases: int = 7) -> dict[str, tuple[str, str]]:
    """Download sample image pairs from HF Dataset and store locally.

    Falls back to local assets/sample_cases if available.
    """
    # Try local first (for development)
    local_root = Path("assets/sample_cases")
    if local_root.exists() and any(local_root.glob("*_left.*")):
        left_map: dict[str, Path] = {}
        right_map: dict[str, Path] = {}
        for left_path in local_root.glob("*_left.*"):
            left_map[left_path.stem.removesuffix("_left")] = left_path
        for right_path in local_root.glob("*_right.*"):
            right_map[right_path.stem.removesuffix("_right")] = right_path
        common_ids = sorted(
            set(left_map).intersection(right_map),
            key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
        )
        if common_ids:
            lookup: dict[str, tuple[str, str]] = {}
            for case_id in common_ids[:max_cases]:
                lookup[f"Case {case_id}"] = (str(left_map[case_id]), str(right_map[case_id]))
            return lookup

    # Download from HF Dataset
    sample_ids = ["1", "4", "5", "6", "7", "8", "9"]
    cache_dir = Path(tempfile.gettempdir()) / "ehc_sample_cases"
    cache_dir.mkdir(parents=True, exist_ok=True)

    lookup = {}
    for case_id in sample_ids[:max_cases]:
        left_name = f"{case_id}_left.jpg"
        right_name = f"{case_id}_right.jpg"
        left_local = cache_dir / left_name
        right_local = cache_dir / right_name

        try:
            for name, local_path in [(left_name, left_local), (right_name, right_local)]:
                if not local_path.exists():
                    resp = httpx.get(f"{HF_MEDIA_BASE}/{name}", follow_redirects=True, timeout=30.0)
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
            lookup[f"Case {case_id}"] = (str(left_local), str(right_local))
        except Exception:
            continue

    return lookup


SAMPLE_LOOKUP = _discover_sample_pairs()
SAMPLE_OPTIONS = list(SAMPLE_LOOKUP.keys())


class AppState(rx.State):
    """Application state for uploads, samples, and prediction workflow."""

    current_step: int = 1  # 1 = Input, 2 = Processing, 3 = Results
    age: int = 55

    selected_sample: str = ""
    using_sample: bool = False
    left_file_name: str = ""
    right_file_name: str = ""

    risk_band: str = ""
    cv_proxy: float = 0.0
    risk_message: str = ""
    explanation_text: str = ""
    related_conditions_text: str = ""
    chart_data: list[dict[str, Any]] = []

    error_message: str = ""
    is_loading: bool = False

    def set_age(self, value: list[int] | int | str) -> None:
        parsed_age = value[0] if isinstance(value, list) else value
        try:
            age_value = int(parsed_age)
        except (TypeError, ValueError):
            return
        self.age = max(1, min(120, age_value))

    def _persist_upload(self, side: str, original_name: str, payload: bytes) -> str:
        """Store upload bytes in Reflex upload dir and return stored name."""
        suffix = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"
        stored_name = f"{side}_{int(time.time() * 1000)}{suffix}"
        target = rx.get_upload_dir() / stored_name
        target.write_bytes(payload)
        return stored_name

    async def handle_left_upload(self, files: list[rx.UploadFile]) -> None:
        if not files:
            return
        file = files[0]
        data = await file.read()
        self.left_file_name = self._persist_upload("left", file.filename or "left.jpg", data)
        self.using_sample = False
        self.selected_sample = ""
        self.current_step = 1
        self.error_message = ""

    async def handle_right_upload(self, files: list[rx.UploadFile]) -> None:
        if not files:
            return
        file = files[0]
        data = await file.read()
        self.right_file_name = self._persist_upload("right", file.filename or "right.jpg", data)
        self.using_sample = False
        self.selected_sample = ""
        self.current_step = 1
        self.error_message = ""

    def set_sample_case(self, selected: str) -> None:
        """Load a selected left-right sample pair into upload storage."""
        self.selected_sample = selected
        self.error_message = ""
        self.current_step = 1

        if selected not in SAMPLE_LOOKUP:
            return

        left_src, right_src = SAMPLE_LOOKUP[selected]
        left_bytes = Path(left_src).read_bytes()
        right_bytes = Path(right_src).read_bytes()

        self.left_file_name = self._persist_upload("leftsample", Path(left_src).name, left_bytes)
        self.right_file_name = self._persist_upload("rightsample", Path(right_src).name, right_bytes)
        self.using_sample = True

    def reset_flow(self) -> list[Any]:
        self.current_step = 1
        self.error_message = ""
        self.is_loading = False
        self.selected_sample = ""
        self.using_sample = False
        self.left_file_name = ""
        self.right_file_name = ""
        self.risk_band = ""
        self.cv_proxy = 0.0
        self.risk_message = ""
        self.explanation_text = ""
        self.related_conditions_text = ""
        self.chart_data = []
        return [
            rx.clear_selected_files("left_upload"),
            rx.clear_selected_files("right_upload"),
        ]

    async def start_prediction(self) -> None:
        if not self.left_file_name or not self.right_file_name:
            self.error_message = "Please provide both left and right fundus images."
            return

        self.error_message = ""
        self.current_step = 2
        self.is_loading = True
        yield

        import asyncio

        await asyncio.sleep(0.35)
        await self._call_backend()

    async def _call_backend(self) -> None:
        """Call FastAPI backend for prediction and update result state."""
        try:
            upload_dir = rx.get_upload_dir()
            left_path = upload_dir / self.left_file_name
            right_path = upload_dir / self.right_file_name

            if not left_path.exists() or not right_path.exists():
                self.error_message = "Uploaded files are missing. Please upload images again."
                self.current_step = 1
                return

            left_bytes = left_path.read_bytes()
            right_bytes = right_path.read_bytes()

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{API_BASE}/predict",
                    data={"age": str(self.age)},
                    files={
                        "left_image": (left_path.name, left_bytes, "image/jpeg"),
                        "right_image": (right_path.name, right_bytes, "image/jpeg"),
                    },
                )
                response.raise_for_status()
                self._parse_results(response.json())
                self.current_step = 3

        except httpx.HTTPStatusError as exc:
            self.error_message = f"API Error {exc.response.status_code}: {exc.response.text}"
            self.current_step = 1
        except httpx.ConnectError:
            self.error_message = f"Cannot reach FastAPI backend on {API_BASE}. Make sure it is running."
            self.current_step = 1
        except Exception as exc:  # pragma: no cover
            self.error_message = f"Prediction failed: {exc}"
            self.current_step = 1
        finally:
            self.is_loading = False

    def _parse_results(self, body: dict[str, Any]) -> None:
        probs = body.get("probabilities", {})
        cv_summary = body.get("cv_summary", {})

        chart = []
        for key, value in probs.items():
            chart.append(
                {
                    "name": LABEL_DESCRIPTIONS.get(key, key),
                    "Probability": round(float(value) * 100.0, 1),
                }
            )
        self.chart_data = sorted(chart, key=lambda item: item["Probability"], reverse=True)

        self.risk_band = str(cv_summary.get("risk_band", "unknown")).upper()
        self.cv_proxy = round(float(cv_summary.get("overall_cv_proxy", 0.0)) * 100.0, 1)

        non_normal = [item for item in self.chart_data if item["name"] != "Normal"]
        top_related = non_normal[:3] if non_normal else self.chart_data[:3]
        related_names = ", ".join(item["name"] for item in top_related) if top_related else "retinal conditions"
        top_related_prob = top_related[0]["Probability"] if top_related else 0.0

        self.risk_message = f"You have a {self.cv_proxy:.1f}% predicted risk of cardiovascular disease."
        recommendation = ""
        if self.risk_band in {"MEDIUM", "HIGH"}:
            recommendation = " It is recommended to consult a doctor."
        self.explanation_text = (
            f"{self.risk_message}{recommendation} "
            f"Your risk category is {self.risk_band.title()}."
        )
        self.related_conditions_text = (
            f"You have {top_related_prob:.1f}% likelihood of other related conditions such as {related_names}."
        )


def card_shell(*children: rx.Component, **kwargs: Any) -> rx.Component:
    """Reusable rounded, vibrant glass card container with colorful shadows."""
    props: dict[str, Any] = {
        "background": CARD_BG_GLASS,
        "border": "1px solid rgba(255, 255, 255, 0.6)",
        "border_radius": "24px",
        "box_shadow": "0 8px 32px rgba(139, 92, 246, 0.12), 0 4px 16px rgba(59, 130, 246, 0.08)",
        "backdrop_filter": "blur(12px)",
        "transition": "all 250ms ease",
        "_hover": {
            "box_shadow": "0 12px 40px rgba(139, 92, 246, 0.18), 0 6px 20px rgba(59, 130, 246, 0.12)",
            "transform": "translateY(-2px)",
        },
    }
    props.update(kwargs)
    return rx.box(*children, **props)


def vibrant_card(*children: rx.Component, gradient: str, border_color: str, **kwargs: Any) -> rx.Component:
    """Vibrant card with gradient background and colored border."""
    props: dict[str, Any] = {
        "background": gradient,
        "border": f"2px solid {border_color}",
        "border_radius": "20px",
        "box_shadow": f"0 6px 24px {border_color}30",
        "backdrop_filter": "blur(8px)",
        "transition": "all 200ms ease",
        "_hover": {
            "box_shadow": f"0 10px 32px {border_color}45",
            "transform": "translateY(-2px)",
        },
    }
    props.update(kwargs)
    return rx.box(*children, **props)


def info_card(title: str, body: rx.Component, icon: str = "info", color: str = PURPLE_VIBRANT) -> rx.Component:
    icon_colors = {
        PURPLE_VIBRANT: PURPLE_VIBRANT,
        BLUE_VIBRANT: BLUE_VIBRANT,
        TEAL_VIBRANT: TEAL_VIBRANT,
        GREEN_VIBRANT: GREEN_VIBRANT,
        ORANGE_VIBRANT: ORANGE_VIBRANT,
        PINK_VIBRANT: PINK_VIBRANT,
        INDIGO_VIBRANT: INDIGO_VIBRANT,
        CYAN_VIBRANT: CYAN_VIBRANT,
    }
    icon_color = icon_colors.get(color, PURPLE_VIBRANT)
    
    return card_shell(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(icon, size=20, color=icon_color),
                    background=f"{icon_color}15",
                    border_radius="12px",
                    padding="0.5rem",
                ),
                rx.heading(title, size="4", color=color, weight="bold"),
                align="center",
                gap="3",
            ),
            body,
            spacing="4",
            align="start",
            width="100%",
        ),
        padding="1.2rem",
        width="100%",
        border_left=f"4px solid {color}",
    )


def left_sidebar() -> rx.Component:
    # Create colorful indicator badges
    indicator_colors = [GREEN_VIBRANT, RED_VIBRANT, PURPLE_VIBRANT, ORANGE_VIBRANT, 
                        BLUE_VIBRANT, PINK_VIBRANT, CYAN_VIBRANT, INDIGO_VIBRANT]
    
    indicators_block = rx.vstack(
        *[
            rx.hstack(
                rx.badge(code, color_scheme="green" if code == "N" else "red" if code in ["D", "G", "C"] else "blue", 
                        variant="soft", radius="full", size="2"),
                rx.vstack(
                    rx.text(full, size="2", weight="bold", color=TEXT_MAIN),
                    rx.text(description, size="1", color=TEXT_MUTED, line_height="1.4"),
                    spacing="1",
                    align="start",
                ),
                align="start",
                gap="3",
                width="100%",
            )
            for code, full, description in INDICATOR_INFO
        ],
        spacing="3",
        align="start",
        width="100%",
    )

    refs_block = rx.vstack(
        rx.hstack(
            rx.icon("book-open", size=16, color=PURPLE_VIBRANT),
            rx.text(
                "Cheung et al. (2012) Retinal microvasculature as a model to study systemic vascular disease.",
                size="2",
                line_height="1.5",
            ),
            align="start",
            gap="2",
        ),
        rx.hstack(
            rx.icon("book-open", size=16, color=PURPLE_VIBRANT),
            rx.text(
                "Rim et al. (2020) Deep learning for cardiovascular risk factors from retinal fundus photographs.",
                size="2",
                line_height="1.5",
            ),
            align="start",
            gap="2",
        ),
        rx.hstack(
            rx.icon("book-open", size=16, color=PURPLE_VIBRANT),
            rx.text(
                "Poplin et al. (2018) Prediction of cardiovascular risk factors from retinal images via deep learning.",
                size="2",
                line_height="1.5",
            ),
            align="start",
            gap="2",
        ),
        spacing="3",
        align="start",
    )

    glossary_block = rx.vstack(
        rx.hstack(
            rx.icon("eye", size=16, color=BLUE_VIBRANT),
            rx.text("Fundus Image: Photo of the inner back surface of the eye.", size="2"),
            align="start",
            gap="2",
        ),
        rx.hstack(
            rx.icon("heart-pulse", size=16, color=RED_VIBRANT),
            rx.text("CV Risk: Estimated likelihood of cardiovascular disease burden.", size="2"),
            align="start",
            gap="2",
        ),
        rx.hstack(
            rx.icon("circle-dot", size=16, color=ORANGE_VIBRANT),
            rx.text("Microaneurysm: Tiny bulge in retinal blood vessels.", size="2"),
            align="start",
            gap="2",
        ),
        rx.hstack(
            rx.icon("git-branch", size=16, color=TEAL_VIBRANT),
            rx.text("Tortuosity: Twisting or curving of blood vessels.", size="2"),
            align="start",
            gap="2",
        ),
        spacing="3",
        align="start",
    )

    return rx.vstack(
        info_card(
            "About the Project",
            rx.text(
                "EYE HEART CONNECTION analyzes bilateral fundus images with age metadata to estimate cardiovascular risk and associated ophthalmic indicators.",
                size="2",
                line_height="1.6",
            ),
            icon="heart",
            color=PINK_VIBRANT,
        ),
        info_card("8 Ophthalmic Indicators", indicators_block, icon="scan-eye", color=BLUE_VIBRANT),
        info_card(
            "Data Flow",
            rx.vstack(
                rx.hstack(
                    rx.badge("1", color_scheme="purple", variant="solid", radius="full"),
                    rx.text("Input Images + Age", size="2"),
                    align="center",
                    gap="2",
                ),
                rx.hstack(
                    rx.badge("2", color_scheme="blue", variant="solid", radius="full"),
                    rx.text("Feature Processing", size="2"),
                    align="center",
                    gap="2",
                ),
                rx.hstack(
                    rx.badge("3", color_scheme="teal", variant="solid", radius="full"),
                    rx.text("Risk Prediction + Explanations", size="2"),
                    align="center",
                    gap="2",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            icon="workflow",
            color=INDIGO_VIBRANT,
        ),
        info_card(
            "Tech Stack",
            rx.vstack(
                rx.hstack(rx.icon("layout", size=14, color=PURPLE_VIBRANT), rx.text(rx.text.strong("Frontend:"), " Reflex (Radix UI + Recharts)", size="2"), align="center", gap="2"),
                rx.hstack(rx.icon("server", size=14, color=BLUE_VIBRANT), rx.text(rx.text.strong("API:"), " FastAPI + Uvicorn + Pydantic", size="2"), align="center", gap="2"),
                rx.hstack(rx.icon("brain", size=14, color=PINK_VIBRANT), rx.text(rx.text.strong("ML Core:"), " PyTorch + EfficientNet-B4", size="2"), align="center", gap="2"),
                rx.hstack(rx.icon("image", size=14, color=TEAL_VIBRANT), rx.text(rx.text.strong("Image:"), " Albumentations + OpenCV", size="2"), align="center", gap="2"),
                rx.hstack(rx.icon("bar-chart-2", size=14, color=ORANGE_VIBRANT), rx.text(rx.text.strong("Data:"), " NumPy + Pandas + scikit-learn", size="2"), align="center", gap="2"),
                rx.hstack(rx.icon("wifi", size=14, color=CYAN_VIBRANT), rx.text(rx.text.strong("Network:"), " HTTPX", size="2"), align="center", gap="2"),
                spacing="2",
                align="start",
                width="100%",
            ),
            icon="cpu",
            color=ORANGE_VIBRANT,
        ),
        info_card("Research References", refs_block, icon="graduation-cap", color=PURPLE_VIBRANT),
        info_card("Medical Terms Glossary", glossary_block, icon="book", color=TEAL_VIBRANT),
        info_card(
            "Credits",
            rx.vstack(
                rx.hstack(
                    rx.icon("user", size=16, color=PURPLE_VIBRANT),
                    rx.text(rx.text.strong("Made by Ayush Saini"), size="2", weight="medium"),
                    align="center",
                    gap="2",
                ),
                rx.hstack(
                    rx.icon("linkedin", size=16, color=BLUE_VIBRANT),
                    rx.link("LinkedIn: ayush-saini-30a4a0372", href="https://www.linkedin.com/in/ayush-saini-30a4a0372/", is_external=True, size="2", color=BLUE_VIBRANT),
                    align="center",
                    gap="2",
                ),
                rx.hstack(
                    rx.icon("github", size=16, color=INDIGO_VIBRANT),
                    rx.link("GitHub: EYE_HEART_CONNECTION", href="https://github.com/ayushsainime/EYE_HEART_CONNECTION", is_external=True, size="2", color=INDIGO_VIBRANT),
                    align="center",
                    gap="2",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            icon="sparkles",
            color=AMBER_VIBRANT,
        ),
        spacing="4",
        width="100%",
        position=["static", "sticky"],
        top="1rem",
    )


def hero_section() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Colorful badges row
            rx.hstack(
                rx.badge("Medical AI Interface", color_scheme="purple", variant="soft", radius="full", size="2"),
                rx.badge("Deep Learning", color_scheme="blue", variant="soft", radius="full", size="2"),
                rx.badge("CV Risk Analysis", color_scheme="teal", variant="soft", radius="full", size="2"),
                rx.badge("Ophthalmic Insights", color_scheme="pink", variant="soft", radius="full", size="2"),
                spacing="2",
                justify="center",
                flex_wrap="wrap",
            ),
            # Main heading with gradient effect
            rx.hstack(
                rx.icon("heart-pulse", size=50, color=PINK_VIBRANT),
                rx.vstack(
                    rx.heading("EYE HEART", size="9", 
                              background=f"linear-gradient(135deg, {PURPLE_VIBRANT}, {PINK_VIBRANT}, {BLUE_VIBRANT})",
                              background_clip="text",
                              weight="bold"),
                    rx.heading("CONNECTION", size="9", 
                              background=f"linear-gradient(135deg, {TEAL_VIBRANT}, {CYAN_VIBRANT}, {GREEN_VIBRANT})",
                              background_clip="text",
                              weight="bold"),
                    spacing="1",
                    align="center",
                ),
                rx.icon("scan-eye", size=50, color=BLUE_VIBRANT),
                align="center",
                justify="center",
                gap="4",
            ),
            rx.text(
                "Predict cardiovascular risk using left and right retinal fundus images.",
                size="5",
                color=TEXT_MUTED,
                text_align="center",
                max_width="760px",
                weight="medium",
            ),
            # Video with colorful gradient border
            rx.box(
                rx.el.video(
                    src=ANIMATION_SRC,
                    auto_play=True,
                    loop=True,
                    muted=True,
                    plays_inline=True,
                    controls=False,
                    width="100%",
                    style={"display": "block", "borderRadius": "16px"},
                ),
                width="100%",
                max_width="760px",
                padding="0.65rem",
                border_radius="20px",
                background=f"linear-gradient(135deg, {PURPLE_VIBRANT}20, {BLUE_VIBRANT}20, {PINK_VIBRANT}20, {TEAL_VIBRANT}20)",
                border=f"2px solid transparent",
                box_shadow=f"0 8px 32px {PURPLE_VIBRANT}20",
            ),
            # Colorful CTA button
            rx.link(
                rx.box(
                    rx.hstack(
                        rx.text("Get Started", size="3", weight="bold", color=TEXT_WHITE),
                        rx.icon("arrow-down", size=18, color=TEXT_WHITE),
                        align="center",
                        justify="center",
                        gap="2",
                    ),
                    padding="0.75rem 1.5rem",
                    border_radius="9999px",
                    background=f"linear-gradient(135deg, {PURPLE_VIBRANT}, {BLUE_VIBRANT})",
                    box_shadow=f"0 6px 20px {PURPLE_VIBRANT}40",
                    transition="all 200ms ease",
                    _hover={
                        "transform": "translateY(-2px)",
                        "box_shadow": f"0 10px 30px {PURPLE_VIBRANT}50",
                    },
                ),
                href="#input-panel",
                text_decoration="none",
            ),
            spacing="5",
            align="center",
            width="100%",
        ),
        padding=["1.5rem", "2rem"],
        width="100%",
        border_radius="24px",
        background="linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, rgba(255, 255, 255, 0.9) 100%)",
        border="1px solid rgba(255, 255, 255, 0.6)",
        box_shadow="0 8px 32px rgba(139, 92, 246, 0.12)",
    )


def upload_preview_card(
    label: str,
    upload_id: str,
    current_file_name: rx.Var[str],
    on_drop_handler: Any,
    card_color: str = PURPLE_VIBRANT,
) -> rx.Component:
    is_left = "left" in upload_id.lower()
    accent_color = BLUE_VIBRANT if is_left else PINK_VIBRANT
    
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("scan-eye" if is_left else "eye", size=20, color=accent_color),
                rx.text(label, size="3", weight="bold", color=TEXT_MAIN),
                align="center",
                gap="2",
            ),
            rx.upload(
                rx.cond(
                    current_file_name != "",
                    rx.box(
                        rx.image(
                            src=rx.get_upload_url(current_file_name),
                            alt=label,
                            width="100%",
                            height="220px",
                            object_fit="cover",
                            border_radius="14px",
                        ),
                        border=f"3px solid {accent_color}",
                        border_radius="18px",
                        overflow="hidden",
                        box_shadow=f"0 4px 16px {accent_color}30",
                    ),
                    rx.vstack(
                        rx.box(
                            rx.icon("upload", size=32, color=accent_color),
                            background=f"{accent_color}15",
                            border_radius="16px",
                            padding="1rem",
                        ),
                        rx.text("Drag and drop or click to upload", color=TEXT_MUTED, size="2", weight="medium"),
                        rx.badge("Accepted: .jpg, .jpeg, .png", color_scheme="gray", variant="soft", radius="full", size="1"),
                        spacing="3",
                        align="center",
                        justify="center",
                        height="220px",
                        width="100%",
                        border=f"2px dashed {accent_color}50",
                        border_radius="18px",
                        background=f"{accent_color}05",
                    ),
                ),
                id=upload_id,
                max_files=1,
                accept={"image/*": [".jpg", ".jpeg", ".png"]},
                on_drop=on_drop_handler,
                width="100%",
            ),
            spacing="4",
            width="100%",
            align="start",
        ),
        padding="1.2rem",
        width="100%",
        border_radius="20px",
        background=CARD_BG_GLASS,
        border=f"1px solid {accent_color}30",
        border_left=f"4px solid {accent_color}",
        box_shadow=f"0 4px 20px {accent_color}15",
        transition="all 200ms ease",
        _hover={
            "box_shadow": f"0 8px 30px {accent_color}25",
            "transform": "translateY(-2px)",
        },
    )


def input_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Vibrant header
            rx.hstack(
                rx.box(
                    rx.icon("settings-2", size=24, color=PURPLE_VIBRANT),
                    background=f"{PURPLE_VIBRANT}15",
                    border_radius="14px",
                    padding="0.6rem",
                ),
                rx.vstack(
                    rx.heading("Input Panel", size="6", color=PURPLE_VIBRANT, weight="bold"),
                    rx.text("Upload fundus images and configure patient data", size="2", color=TEXT_MUTED),
                    spacing="1",
                    align="start",
                ),
                align="center",
                gap="3",
            ),
            rx.divider(border_color=f"{PURPLE_VIBRANT}30"),
            # Upload cards
            rx.flex(
                upload_preview_card(
                    "Left Fundus Image",
                    "left_upload",
                    AppState.left_file_name,
                    AppState.handle_left_upload(rx.upload_files(upload_id="left_upload")),
                ),
                upload_preview_card(
                    "Right Fundus Image",
                    "right_upload",
                    AppState.right_file_name,
                    AppState.handle_right_upload(rx.upload_files(upload_id="right_upload")),
                ),
                flex_direction=["column", "row"],
                gap="4",
                width="100%",
            ),
            # Age card with orange accent
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("user", size=18, color=ORANGE_VIBRANT),
                        rx.text("Patient Age", size="3", weight="bold", color=TEXT_MAIN),
                        align="center",
                        gap="2",
                    ),
                    rx.flex(
                        rx.text("Selected Age:", size="2", color=TEXT_MUTED),
                        rx.badge(AppState.age, color_scheme="orange", variant="solid", radius="full", size="2"),
                        align="center",
                        gap="2",
                    ),
                    rx.slider(
                        min=1,
                        max=120,
                        default_value=[55],
                        on_change=AppState.set_age,
                        width="100%",
                        color_scheme="orange",
                    ),
                    rx.input(
                        type="number",
                        min=1,
                        max=120,
                        step=1,
                        value=AppState.age,
                        on_change=AppState.set_age,
                        placeholder="Enter patient age",
                        width="180px",
                        variant="surface",
                        border=f"2px solid {ORANGE_VIBRANT}40",
                    ),
                    spacing="3",
                    width="100%",
                    align="start",
                ),
                padding="1.2rem",
                width="100%",
                border_radius="18px",
                background=GRADIENT_ORANGE,
                border=f"2px solid {ORANGE_VIBRANT}30",
                border_left=f"4px solid {ORANGE_VIBRANT}",
            ),
            # Sample images card with teal accent
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.icon("image", size=18, color=TEAL_VIBRANT),
                        rx.text("Sample Images", size="3", weight="bold", color=TEXT_MAIN),
                        align="center",
                        gap="2",
                    ),
                    rx.text(
                        "Load a ready left-right sample pair from assets/sample_cases.",
                        size="2",
                        color=TEXT_MUTED,
                    ),
                    rx.select(
                        SAMPLE_OPTIONS,
                        placeholder="Select a sample case",
                        value=AppState.selected_sample,
                        on_change=AppState.set_sample_case,
                        width="100%",
                        color_scheme="teal",
                        variant="surface",
                        radius="large",
                    ),
                    rx.cond(
                        AppState.selected_sample != "",
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=GREEN_VIBRANT),
                            rx.text(
                                "Loaded: ",
                                AppState.selected_sample,
                                size="2",
                                color=GREEN_VIBRANT,
                                weight="medium",
                            ),
                            align="center",
                            gap="1",
                        ),
                    ),
                    spacing="3",
                    width="100%",
                    align="start",
                ),
                padding="1.2rem",
                width="100%",
                border_radius="18px",
                background=GRADIENT_TEAL,
                border=f"2px solid {TEAL_VIBRANT}30",
                border_left=f"4px solid {TEAL_VIBRANT}",
            ),
            # Prediction button with gradient
            rx.button(
                rx.cond(
                    AppState.is_loading,
                    rx.hstack(
                        rx.spinner(size="2", color=TEXT_WHITE),
                        rx.text("Running Prediction...", weight="bold", color=TEXT_WHITE),
                        align="center",
                        gap="2",
                    ),
                    rx.hstack(
                        rx.icon("play", size=20, color=TEXT_WHITE),
                        rx.text("Run Prediction", weight="bold", color=TEXT_WHITE, size="4"),
                        align="center",
                        gap="2",
                    ),
                ),
                on_click=AppState.start_prediction,
                width="100%",
                size="4",
                radius="full",
                disabled=AppState.is_loading,
                background=f"linear-gradient(135deg, {PURPLE_VIBRANT}, {PINK_VIBRANT})",
                box_shadow=f"0 6px 24px {PURPLE_VIBRANT}35",
                border="none",
                _hover={
                    "background": f"linear-gradient(135deg, {PINK_VIBRANT}, {PURPLE_VIBRANT})",
                    "box_shadow": f"0 8px 30px {PURPLE_VIBRANT}45",
                    "transform": "translateY(-2px)",
                } if not AppState.is_loading else {},
            ),
            rx.cond(
                AppState.error_message != "",
                rx.box(
                    rx.hstack(
                        rx.icon("alert-triangle", size=20, color=RED_VIBRANT),
                        rx.text(AppState.error_message, size="2", color=RED_VIBRANT, weight="medium"),
                        align="center",
                        gap="2",
                    ),
                    padding="1rem",
                    width="100%",
                    border_radius="14px",
                    background=f"{RED_VIBRANT}10",
                    border=f"1px solid {RED_VIBRANT}40",
                    border_left=f"4px solid {RED_VIBRANT}",
                ),
            ),
            spacing="5",
            align="start",
            width="100%",
        ),
        padding=["1.2rem", "1.5rem"],
        width="100%",
        id="input-panel",
        border_radius="24px",
        background=CARD_BG_GLASS,
        border="1px solid rgba(255, 255, 255, 0.6)",
        box_shadow="0 8px 32px rgba(139, 92, 246, 0.12)",
    )


def risk_band_badge() -> rx.Component:
    return rx.badge(
        AppState.risk_band,
        color_scheme=rx.match(
            AppState.risk_band,
            ("LOW", "green"),
            ("MEDIUM", "orange"),
            ("HIGH", "red"),
            "gray",
        ),
        variant="solid",
        radius="full",
        size="3",
        padding="0.45rem 0.9rem",
    )


def results_section() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Vibrant header
            rx.hstack(
                rx.box(
                    rx.icon("activity", size=24, color=CYAN_VIBRANT),
                    background=f"{CYAN_VIBRANT}15",
                    border_radius="14px",
                    padding="0.6rem",
                ),
                rx.vstack(
                    rx.heading("Prediction Results", size="6", color=CYAN_VIBRANT, weight="bold"),
                    rx.text("AI-powered cardiovascular risk analysis", size="2", color=TEXT_MUTED),
                    spacing="1",
                    align="start",
                ),
                align="center",
                gap="3",
            ),
            rx.divider(border_color=f"{CYAN_VIBRANT}30"),
            rx.match(
                AppState.current_step,
                (
                    2,
                    # Loading state with vibrant styling
                    rx.box(
                        rx.vstack(
                            rx.box(
                                rx.spinner(size="3", color=PURPLE_VIBRANT),
                                background=f"{PURPLE_VIBRANT}10",
                                border_radius="20px",
                                padding="1.5rem",
                            ),
                            rx.heading("Analyzing retinal biomarkers...", size="5", color=PURPLE_VIBRANT),
                            rx.text(
                                "Running multimodal inference on bilateral fundus images and age input.",
                                size="3",
                                color=TEXT_MUTED,
                                text_align="center",
                            ),
                            rx.box(
                                rx.progress(value=72, max=100, color_scheme="purple", width="300px"),
                                padding="0.5rem 1rem",
                                border_radius="14px",
                                background=f"{PURPLE_VIBRANT}08",
                            ),
                            align="center",
                            spacing="4",
                            width="100%",
                            padding="2rem 1rem",
                        ),
                        width="100%",
                        border_radius="20px",
                        background=GRADIENT_PURPLE,
                        border=f"2px solid {PURPLE_VIBRANT}30",
                    ),
                ),
                (
                    3,
                    rx.vstack(
                        # Risk stats row
                        rx.flex(
                            # Risk Band card
                            rx.box(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("shield", size=18, color=INDIGO_VIBRANT),
                                        rx.text("Risk Band", size="2", color=TEXT_MUTED, weight="medium"),
                                        align="center",
                                        gap="2",
                                    ),
                                    risk_band_badge(),
                                    spacing="3",
                                    align="start",
                                ),
                                padding="1.2rem",
                                width="100%",
                                border_radius="18px",
                                background=GRADIENT_PURPLE,
                                border=f"2px solid {INDIGO_VIBRANT}30",
                                border_top=f"4px solid {INDIGO_VIBRANT}",
                            ),
                            # CV Risk card
                            rx.box(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("heart-pulse", size=18, color=PINK_VIBRANT),
                                        rx.text("Predicted CV Risk", size="2", color=TEXT_MUTED, weight="medium"),
                                        align="center",
                                        gap="2",
                                    ),
                                    rx.heading(
                                        f"{AppState.cv_proxy:.1f}%", 
                                        size="8", 
                                        background=f"linear-gradient(135deg, {PINK_VIBRANT}, {RED_VIBRANT})",
                                        background_clip="text",
                                        weight="bold"
                                    ),
                                    spacing="3",
                                    align="start",
                                ),
                                padding="1.2rem",
                                width="100%",
                                border_radius="18px",
                                background=GRADIENT_PINK,
                                border=f"2px solid {PINK_VIBRANT}30",
                                border_top=f"4px solid {PINK_VIBRANT}",
                            ),
                            flex_direction=["column", "row"],
                            gap="4",
                            width="100%",
                        ),
                        # Chart card
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("bar-chart-2", size=20, color=TEAL_VIBRANT),
                                    rx.heading("Disease Probability Chart", size="4", color=TEAL_VIBRANT),
                                    align="center",
                                    gap="2",
                                ),
                                rx.recharts.bar_chart(
                                    rx.recharts.bar(
                                        data_key="Probability",
                                        fill=TEAL_VIBRANT,
                                        radius=[8, 8, 0, 0],
                                    ),
                                    rx.recharts.x_axis(data_key="name", stroke=TEXT_MUTED, tick={"fontSize": 11}),
                                    rx.recharts.y_axis(stroke=TEXT_MUTED, tick={"fontSize": 11}),
                                    rx.recharts.tooltip(
                                        cursor={"fill": f"{TEAL_VIBRANT}15"},
                                        content_style={
                                            "backgroundColor": "rgba(255,255,255,0.95)",
                                            "border": f"1px solid {TEAL_VIBRANT}30",
                                            "borderRadius": "12px",
                                        }
                                    ),
                                    data=AppState.chart_data,
                                    width="100%",
                                    height=330,
                                ),
                                spacing="4",
                                width="100%",
                                align="start",
                            ),
                            padding="1.35rem",
                            width="100%",
                            border_radius="18px",
                            background=GRADIENT_TEAL,
                            border=f"2px solid {TEAL_VIBRANT}30",
                            border_left=f"4px solid {TEAL_VIBRANT}",
                        ),
                        # Clinical Explanation card
                        rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("file-text", size=20, color=AMBER_VIBRANT),
                                    rx.heading("Clinical Explanation", size="4", color=AMBER_VIBRANT),
                                    align="center",
                                    gap="2",
                                ),
                                rx.box(
                                    rx.text(AppState.explanation_text, size="3", line_height="1.8", color=TEXT_MAIN),
                                    padding="1rem",
                                    border_radius="12px",
                                    background=f"{AMBER_VIBRANT}08",
                                    border=f"1px solid {AMBER_VIBRANT}20",
                                ),
                                rx.box(
                                    rx.text(AppState.related_conditions_text, size="3", line_height="1.8", color=TEXT_MAIN),
                                    padding="1rem",
                                    border_radius="12px",
                                    background=f"{BLUE_VIBRANT}08",
                                    border=f"1px solid {BLUE_VIBRANT}20",
                                ),
                                spacing="4",
                                width="100%",
                                align="start",
                            ),
                            padding="1.35rem",
                            width="100%",
                            border_radius="18px",
                            background=GRADIENT_ORANGE,
                            border=f"2px solid {AMBER_VIBRANT}30",
                            border_left=f"4px solid {AMBER_VIBRANT}",
                        ),
                        # Reset button
                        rx.button(
                            rx.hstack(
                                rx.icon("refresh-cw", size=18),
                                rx.text("Reset Results", weight="bold"),
                                align="center",
                                gap="2",
                            ),
                            on_click=AppState.reset_flow,
                            variant="outline",
                            color_scheme="purple",
                            radius="full",
                            border=f"2px solid {PURPLE_VIBRANT}",
                            _hover={
                                "background": f"{PURPLE_VIBRANT}10",
                            },
                        ),
                        spacing="5",
                        width="100%",
                        align="start",
                    ),
                ),
                # Default state
                rx.box(
                    rx.vstack(
                        rx.box(
                            rx.icon("bar-chart-2", size=48, color=f"{CYAN_VIBRANT}40"),
                            background=f"{CYAN_VIBRANT}10",
                            border_radius="20px",
                            padding="1.5rem",
                        ),
                        rx.text(
                            "Run prediction to view risk band, disease probabilities, and explanation.",
                            size="3",
                            color=TEXT_MUTED,
                            text_align="center",
                        ),
                        spacing="3",
                        align="center",
                        padding="2rem",
                    ),
                    width="100%",
                    border_radius="18px",
                    background=GRADIENT_TEAL,
                    border=f"2px dashed {CYAN_VIBRANT}40",
                ),
            ),
            spacing="5",
            width="100%",
            align="start",
        ),
        padding=["1.2rem", "1.5rem"],
        width="100%",
        border_radius="24px",
        background=CARD_BG_GLASS,
        border="1px solid rgba(255, 255, 255, 0.6)",
        box_shadow="0 8px 32px rgba(6, 182, 212, 0.12)",
    )


def main_content() -> rx.Component:
    return rx.vstack(
        hero_section(),
        input_panel(),
        results_section(),
        spacing="5",
        width="100%",
        align="start",
        id="main-content",
    )


def index() -> rx.Component:
    return rx.box(
        rx.flex(
            rx.box(
                left_sidebar(),
                width=["100%", "360px"],
                min_width=["100%", "320px"],
                flex_shrink="0",
            ),
            rx.box(
                main_content(),
                width="100%",
                min_width="0",
            ),
            flex_direction=["column", "row"],
            gap="5",
            align="start",
            width="100%",
            max_width="1450px",
            margin="0 auto",
            padding=["1rem", "1.25rem"],
        ),
        min_height="100vh",
        background=(
            "radial-gradient(circle at 0% 0%, rgba(72, 187, 120, 0.16) 0%, transparent 30%), "
            "radial-gradient(circle at 100% 10%, rgba(132, 226, 172, 0.22) 0%, transparent 34%), "
            f"{BG_TINT}"
        ),
        color=TEXT_MAIN,
        scroll_behavior="smooth",
        padding_bottom="2rem",
    )


app = rx.App(
    theme=rx.theme(
        appearance="light",
        radius="large",
        accent_color="green",
        gray_color="sage",
    ),
    stylesheets=[
        "/frontend.css",
    ],
)
app.add_page(index, title="EYE HEART CONNECTION")
