"""Eye-Heart Connection — Reflex frontend (Phase 1: simple starter)."""

from __future__ import annotations

import httpx
import reflex as rx


API_BASE = "http://localhost:8000"


class PredictionState(rx.State):
    """Reactive state that holds uploaded files, age, and prediction results."""

    # --- inputs ---
    left_files: list[rx.UploadFile] = []
    right_files: list[rx.UploadFile] = []
    age: int = 55

    # --- outputs ---
    risk_band: str = ""
    cv_proxy: str = ""
    risk_message: str = ""
    prediction_json: str = ""
    is_loading: bool = False
    error_message: str = ""

    def set_age(self, value: list[int]) -> None:
        """Handle slider change (Reflex sliders emit a list)."""
        self.age = value[0] if isinstance(value, list) else int(value)

    async def handle_left_upload(self, files: list[rx.UploadFile]) -> None:
        self.left_files = files

    async def handle_right_upload(self, files: list[rx.UploadFile]) -> None:
        self.right_files = files

    async def run_prediction(self) -> None:
        """POST to FastAPI /predict and populate result fields."""
        self.error_message = ""
        self.risk_band = ""
        self.cv_proxy = ""
        self.risk_message = ""
        self.prediction_json = ""

        if not self.left_files or not self.right_files:
            self.error_message = "Please upload both left and right fundus images."
            return

        self.is_loading = True
        yield  # update UI to show loading

        try:
            left_file = self.left_files[0]
            right_file = self.right_files[0]

            left_bytes = await left_file.read()
            right_bytes = await right_file.read()

            left_name = left_file.filename or "left.jpg"
            right_name = right_file.filename or "right.jpg"

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{API_BASE}/predict",
                    data={"age": str(self.age)},
                    files={
                        "left_image": (left_name, left_bytes, "image/jpeg"),
                        "right_image": (right_name, right_bytes, "image/jpeg"),
                    },
                )
                resp.raise_for_status()
                body = resp.json()

            # Parse response
            cv_summary = body.get("cv_summary", {})
            risk_band_val = str(cv_summary.get("risk_band", "unknown")).upper()
            overall_cv = float(cv_summary.get("overall_cv_proxy", 0.0))
            chance_pct = overall_cv * 100.0

            guidance = {
                "LOW": "You currently appear to have a low cardiovascular risk profile.",
                "MEDIUM": "You appear to have a moderate cardiovascular risk profile. Consider a clinical checkup.",
                "HIGH": "You appear to have a high cardiovascular risk profile. Please consult a doctor soon.",
            }.get(risk_band_val, "Please consult a healthcare professional for interpretation.")

            self.risk_band = risk_band_val
            self.cv_proxy = f"{overall_cv:.4f}"
            self.risk_message = f"{chance_pct:.1f}% predicted CV risk · {risk_band_val} · {guidance}"

            import json
            self.prediction_json = json.dumps(body, indent=2)

        except httpx.HTTPStatusError as exc:
            self.error_message = f"API error {exc.response.status_code}: {exc.response.text}"
        except httpx.ConnectError:
            self.error_message = (
                "Cannot reach FastAPI backend. Make sure it is running on "
                f"{API_BASE} (uvicorn api.main:app --port 8000)."
            )
        except Exception as exc:
            self.error_message = f"Prediction failed: {exc}"
        finally:
            self.is_loading = False


def _hero() -> rx.Component:
    return rx.box(
        rx.heading("🩺 Eye-Heart Connection", size="7"),
        rx.text(
            "Multimodal retinal AI — upload bilateral fundus images + age to "
            "estimate cardiovascular risk indicators.",
            color_scheme="gray",
        ),
        padding="24px",
        margin_bottom="20px",
        border_radius="14px",
        background="linear-gradient(120deg, #dcfce7 0%, #bbf7d0 100%)",
        border="1px solid #86efac",
    )


def _upload_card(label: str, upload_id: str, handler: str) -> rx.Component:
    return rx.box(
        rx.text(label, weight="bold", margin_bottom="8px"),
        rx.upload(
            rx.box(
                rx.text("Drag & drop or click to upload", color="gray"),
                padding="40px",
                border="2px dashed #86efac",
                border_radius="12px",
                text_align="center",
                cursor="pointer",
            ),
            id=upload_id,
            accept={"image/*": [".jpg", ".jpeg", ".png"]},
            max_files=1,
            on_drop=getattr(PredictionState, handler)(rx.upload_files(upload_id=upload_id)),
        ),
        flex="1",
    )


def _results() -> rx.Component:
    return rx.cond(
        PredictionState.risk_band != "",
        rx.box(
            rx.heading("Results", size="5", margin_bottom="12px"),
            rx.flex(
                rx.box(
                    rx.text("Risk Band", color="gray", size="2"),
                    rx.heading(PredictionState.risk_band, size="6"),
                    padding="16px",
                    border_radius="12px",
                    background="#f0fdf4",
                    border="1px solid #bbf7d0",
                    flex="1",
                ),
                rx.box(
                    rx.text("CV Proxy Score", color="gray", size="2"),
                    rx.heading(PredictionState.cv_proxy, size="6"),
                    padding="16px",
                    border_radius="12px",
                    background="#f0fdf4",
                    border="1px solid #bbf7d0",
                    flex="1",
                ),
                gap="16px",
                width="100%",
            ),
            rx.box(
                rx.text(PredictionState.risk_message, size="3"),
                padding="16px",
                margin_top="12px",
                border_radius="12px",
                background="#ecfdf5",
                border="1px solid #a7f3d0",
            ),
            rx.box(
                rx.text("Raw JSON Response", weight="bold", margin_bottom="8px"),
                rx.code_block(
                    PredictionState.prediction_json,
                    language="json",
                ),
                margin_top="16px",
            ),
            padding="20px",
            border_radius="14px",
            border="1px solid #e5e7eb",
            margin_top="20px",
        ),
    )


def index() -> rx.Component:
    return rx.box(
        rx.box(
            _hero(),
            # Upload section
            rx.box(
                rx.flex(
                    _upload_card("Left Eye Fundus", "left_upload", "handle_left_upload"),
                    _upload_card("Right Eye Fundus", "right_upload", "handle_right_upload"),
                    gap="20px",
                    width="100%",
                ),
                # Age slider
                rx.box(
                    rx.text(
                        rx.text.strong("Patient Age: "),
                        PredictionState.age,
                        size="3",
                    ),
                    rx.slider(
                        min=0,
                        max=120,
                        step=1,
                        default_value=[55],
                        on_change=PredictionState.set_age,
                    ),
                    margin_top="20px",
                    margin_bottom="20px",
                ),
                # Predict button
                rx.button(
                    rx.cond(
                        PredictionState.is_loading,
                        rx.spinner(size="3"),
                        rx.text("Run Prediction"),
                    ),
                    on_click=PredictionState.run_prediction,
                    size="3",
                    width="100%",
                    color_scheme="green",
                    loading=PredictionState.is_loading,
                ),
                # Error message
                rx.cond(
                    PredictionState.error_message != "",
                    rx.callout(
                        PredictionState.error_message,
                        icon="alert_triangle",
                        color_scheme="red",
                        margin_top="12px",
                    ),
                ),
                # Results
                _results(),
                padding="20px",
                border_radius="14px",
                border="1px solid #e5e7eb",
                background="white",
            ),
            max_width="900px",
            margin="0 auto",
            padding="24px",
        ),
        min_height="100vh",
        background="#f8fafc",
    )


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="green"),
)
app.add_page(index, title="Eye-Heart Connection")
