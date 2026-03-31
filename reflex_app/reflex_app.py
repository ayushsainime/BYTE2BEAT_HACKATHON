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

GREEN_MAIN = rx.color("green", 10)
GREEN_SOFT = rx.color("green", 3)
GREEN_BORDER = rx.color("green", 6)
BG_TINT = rx.color("green", 1)
TEXT_MAIN = rx.color("gray", 12)
TEXT_MUTED = rx.color("gray", 10)
CARD_BG = "rgba(255,255,255,0.92)"

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
    """Reusable rounded, soft medical card container."""
    props: dict[str, Any] = {
        "background": CARD_BG,
        "border": f"1px solid {GREEN_BORDER}",
        "border_radius": "20px",
        "box_shadow": "0 18px 45px -28px rgba(20, 120, 80, 0.45)",
        "backdrop_filter": "blur(8px)",
        "transition": "transform 180ms ease, box-shadow 180ms ease",
        "_hover": {"box_shadow": "0 20px 48px -28px rgba(20, 120, 80, 0.55)"},
    }
    props.update(kwargs)
    return rx.box(*children, **props)


def info_card(title: str, body: rx.Component) -> rx.Component:
    return card_shell(
        rx.vstack(
            rx.heading(title, size="4", color=GREEN_MAIN),
            body,
            spacing="3",
            align="start",
            width="100%",
        ),
        padding="1.1rem",
        width="100%",
    )


def left_sidebar() -> rx.Component:
    indicators_block = rx.vstack(
        *[
            rx.text(f"{code} - {full}: {description}", size="2", color=TEXT_MAIN, line_height="1.5")
            for code, full, description in INDICATOR_INFO
        ],
        spacing="2",
        align="start",
        width="100%",
    )

    refs_block = rx.vstack(
        rx.text(
            "Cheung et al. (2012) Retinal microvasculature as a model to study systemic vascular disease.",
            size="2",
            line_height="1.5",
        ),
        rx.text(
            "Rim et al. (2020) Deep learning for cardiovascular risk factors from retinal fundus photographs.",
            size="2",
            line_height="1.5",
        ),
        rx.text(
            "Poplin et al. (2018) Prediction of cardiovascular risk factors from retinal images via deep learning.",
            size="2",
            line_height="1.5",
        ),
        spacing="2",
        align="start",
    )

    glossary_block = rx.vstack(
        rx.text("Fundus Image: Photo of the inner back surface of the eye.", size="2"),
        rx.text("CV Risk: Estimated likelihood of cardiovascular disease burden.", size="2"),
        rx.text("Microaneurysm: Tiny bulge in retinal blood vessels.", size="2"),
        rx.text("Tortuosity: Twisting or curving of blood vessels.", size="2"),
        spacing="2",
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
        ),
        info_card("8 Ophthalmic Indicators", indicators_block),
        info_card(
            "Data Flow",
            rx.text(rx.text.strong("Input Images + Age → Feature Processing → Risk Prediction + Explanations"), size="2"),
        ),
        info_card(
            "Tech Stack",
            rx.vstack(
                rx.text(rx.text.strong("Frontend:"), " Reflex (Radix UI + Recharts)", size="2"),
                rx.text(rx.text.strong("API:"), " FastAPI + Uvicorn + Pydantic + python-multipart", size="2"),
                rx.text(rx.text.strong("ML Core:"), " PyTorch + Torchvision (EfficientNet-B4)", size="2"),
                rx.text(rx.text.strong("Image Pipeline:"), " Albumentations + OpenCV + Pillow", size="2"),
                rx.text(rx.text.strong("Data/Metrics:"), " NumPy + Pandas + scikit-learn + Matplotlib", size="2"),
                rx.text(rx.text.strong("Client/Networking:"), " HTTPX", size="2"),
                spacing="1",
                align="start",
            ),
        ),
        info_card("Research References", refs_block),
        info_card("Medical Terms Glossary", glossary_block),
        info_card(
            "Credits",
            rx.vstack(
                rx.text(rx.text.strong("Made by Ayush Saini"), size="2", weight="medium"),
                rx.link("LinkedIn: ayush-saini-30a4a0372", href="https://www.linkedin.com/in/ayush-saini-30a4a0372/", is_external=True, size="2", color=GREEN_MAIN),
                rx.link("GitHub: EYE_HEART_CONNECTION", href="https://github.com/ayushsainime/EYE_HEART_CONNECTION", is_external=True, size="2", color=GREEN_MAIN),
                spacing="1",
                align="start",
            ),
        ),
        spacing="4",
        width="100%",
        position=["static", "sticky"],
        top="1rem",
    )


def hero_section() -> rx.Component:
    return card_shell(
        rx.vstack(
            rx.badge("Medical AI Interface", color_scheme="green", variant="soft", radius="full"),
            rx.heading("EYE HEART CONNECTION", size="9", text_align="center", color=GREEN_MAIN),
            rx.text(
                "Predict cardiovascular risk using left and right retinal fundus images.",
                size="4",
                color=TEXT_MUTED,
                text_align="center",
                max_width="760px",
            ),
            card_shell(
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
                border_radius="18px",
                background="linear-gradient(145deg, rgba(23, 149, 85, 0.12), rgba(211, 249, 216, 0.35))",
            ),
            rx.link(
                card_shell(
                    rx.hstack(
                        rx.text("Try it out", size="3", weight="bold", color=GREEN_MAIN),
                        rx.icon("arrow-down", size=18, color=GREEN_MAIN),
                        align="center",
                        justify="center",
                        gap="2",
                    ),
                    padding="0.6rem 1.1rem",
                    border_radius="9999px",
                    background=GREEN_SOFT,
                ),
                href="#input-panel",
                text_decoration="none",
            ),
            spacing="4",
            align="center",
            width="100%",
        ),
        padding=["1rem", "1.5rem"],
        width="100%",
    )


def upload_preview_card(
    label: str,
    upload_id: str,
    current_file_name: rx.Var[str],
    on_drop_handler: Any,
) -> rx.Component:
    return card_shell(
        rx.vstack(
            rx.text(label, size="3", weight="bold", color=TEXT_MAIN),
            rx.upload(
                rx.cond(
                    current_file_name != "",
                    rx.image(
                        src=rx.get_upload_url(current_file_name),
                        alt=label,
                        width="100%",
                        height="220px",
                        object_fit="cover",
                        border_radius="14px",
                    ),
                    rx.vstack(
                        rx.icon("upload", size=26, color=GREEN_MAIN),
                        rx.text("Drag and drop or click to upload", color=TEXT_MUTED, size="2"),
                        rx.text("Accepted: .jpg, .jpeg, .png", color=TEXT_MUTED, size="1"),
                        spacing="2",
                        align="center",
                        justify="center",
                        height="220px",
                        width="100%",
                        border=f"2px dashed {GREEN_BORDER}",
                        border_radius="14px",
                        background=rx.color("green", 2),
                    ),
                ),
                id=upload_id,
                max_files=1,
                accept={"image/*": [".jpg", ".jpeg", ".png"]},
                on_drop=on_drop_handler,
                width="100%",
            ),
            spacing="3",
            width="100%",
            align="start",
        ),
        padding="1rem",
        width="100%",
    )


def input_panel() -> rx.Component:
    return card_shell(
        rx.vstack(
            rx.heading("Input Panel", size="5", color=GREEN_MAIN),
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
            card_shell(
                rx.vstack(
                    rx.text("Patient Age", size="3", weight="bold"),
                    rx.flex(
                        rx.text("Selected Age:", size="2", color=TEXT_MUTED),
                        rx.badge(AppState.age, color_scheme="green", variant="soft", radius="full"),
                        align="center",
                        gap="2",
                    ),
                    rx.slider(
                        min=1,
                        max=120,
                        default_value=[55],
                        on_change=AppState.set_age,
                        width="100%",
                        color_scheme="green",
                    ),
                    rx.input(
                        type="number",
                        min=1,
                        max=120,
                        step=1,
                        value=AppState.age,
                        on_change=AppState.set_age,
                        placeholder="Enter patient age",
                        width="220px",
                        variant="surface",
                    ),
                    spacing="3",
                    width="100%",
                    align="start",
                ),
                padding="1rem",
                width="100%",
                border_radius="16px",
            ),
            card_shell(
                rx.vstack(
                    rx.text("Sample Images Dropdown", size="3", weight="bold"),
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
                        color_scheme="green",
                        variant="surface",
                        radius="large",
                    ),
                    rx.cond(
                        AppState.selected_sample != "",
                        rx.text(
                            "Loaded sample pair: ",
                            AppState.selected_sample,
                            size="2",
                            color=GREEN_MAIN,
                            weight="medium",
                        ),
                    ),
                    spacing="3",
                    width="100%",
                    align="start",
                ),
                padding="1rem",
                width="100%",
                border_radius="16px",
            ),
            rx.button(
                rx.cond(
                    AppState.is_loading,
                    rx.hstack(
                        rx.spinner(size="2"),
                        rx.text("Running Prediction...", weight="bold"),
                        align="center",
                        gap="2",
                    ),
                    rx.text("Run Prediction", weight="bold"),
                ),
                on_click=AppState.start_prediction,
                width="100%",
                size="4",
                color_scheme="green",
                radius="large",
                disabled=AppState.is_loading,
            ),
            rx.cond(
                AppState.error_message != "",
                rx.callout(
                    AppState.error_message,
                    icon="triangle_alert",
                    color_scheme="red",
                    width="100%",
                ),
            ),
            spacing="4",
            align="start",
            width="100%",
        ),
        padding=["1rem", "1.35rem"],
        width="100%",
        id="input-panel",
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
    return card_shell(
        rx.vstack(
            rx.heading("Prediction Results", size="5", color=GREEN_MAIN),
            rx.match(
                AppState.current_step,
                (
                    2,
                    card_shell(
                        rx.vstack(
                            rx.spinner(size="3"),
                            rx.heading("Analyzing retinal biomarkers...", size="4"),
                            rx.text(
                                "Running multimodal inference on bilateral fundus images and age input.",
                                size="2",
                                color=TEXT_MUTED,
                            ),
                            rx.progress(value=72, max=100, color_scheme="green", width="70%"),
                            align="center",
                            spacing="4",
                            width="100%",
                            padding="2rem 1rem",
                        ),
                        width="100%",
                        border_radius="16px",
                    ),
                ),
                (
                    3,
                    rx.vstack(
                        rx.flex(
                            card_shell(
                                rx.vstack(
                                    rx.text("Risk Band", size="2", color=TEXT_MUTED),
                                    risk_band_badge(),
                                    spacing="2",
                                    align="start",
                                ),
                                padding="1rem",
                                width="100%",
                                border_radius="16px",
                            ),
                            card_shell(
                                rx.vstack(
                                    rx.text("Predicted CV Risk", size="2", color=TEXT_MUTED),
                                    rx.heading(f"{AppState.cv_proxy:.1f}%", size="8", color=GREEN_MAIN),
                                    spacing="2",
                                    align="start",
                                ),
                                padding="1rem",
                                width="100%",
                                border_radius="16px",
                            ),
                            flex_direction=["column", "row"],
                            gap="4",
                            width="100%",
                        ),
                        card_shell(
                            rx.vstack(
                                rx.heading("Disease Probability Chart", size="4"),
                                rx.recharts.bar_chart(
                                    rx.recharts.bar(
                                        data_key="Probability",
                                        fill=GREEN_MAIN,
                                        radius=[6, 6, 0, 0],
                                    ),
                                    rx.recharts.x_axis(data_key="name", stroke=TEXT_MUTED),
                                    rx.recharts.y_axis(stroke=TEXT_MUTED),
                                    rx.recharts.tooltip(cursor={"fill": rx.color("green", 2)}),
                                    data=AppState.chart_data,
                                    width="100%",
                                    height=330,
                                ),
                                spacing="3",
                                width="100%",
                                align="start",
                            ),
                            padding="1.15rem",
                            width="100%",
                            border_radius="16px",
                        ),
                        card_shell(
                            rx.vstack(
                                rx.heading("Clinical Explanation", size="4"),
                                rx.text(AppState.explanation_text, size="3", line_height="1.7"),
                                rx.text(AppState.related_conditions_text, size="3", line_height="1.7"),
                                spacing="3",
                                width="100%",
                                align="start",
                            ),
                            padding="1.15rem",
                            width="100%",
                            border_radius="16px",
                        ),
                        rx.button(
                            "Reset Results",
                            on_click=AppState.reset_flow,
                            variant="outline",
                            color_scheme="green",
                            radius="large",
                        ),
                        spacing="4",
                        width="100%",
                        align="start",
                    ),
                ),
                rx.text(
                    "Run prediction to view risk band, disease probabilities, and explanation.",
                    size="2",
                    color=TEXT_MUTED,
                ),
            ),
            spacing="4",
            width="100%",
            align="start",
        ),
        width="100%",
        padding=["1rem", "1.35rem"],
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
