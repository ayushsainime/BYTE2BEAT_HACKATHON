from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from inference.predictor import Predictor
from utils.config import (
    load_api_config,
    load_cv_proxy_config,
    load_data_config,
    load_inference_config,
    load_model_config,
)

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

APP_THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
)

APP_JS = """
() => {
  try {
    localStorage.setItem("theme", "light");
  } catch (e) {}
  document.documentElement.classList.remove("dark");
}
"""

APP_CSS = """
.app-wrap { max-width: 1200px !important; margin: 0 auto; }
* {
  font-family: "Courier New", Courier, monospace !important;
}
.left-pane {
  background: linear-gradient(180deg, #ecfdf5 0%, #dcfce7 100%);
  border: 1px solid #86efac;
  border-radius: 22px;
  padding: 12px;
  box-shadow: 0 10px 22px rgba(16, 185, 129, 0.14);
}
.left-pane .gr-group {
  background: transparent !important;
  border-color: #a7f3d0 !important;
}
.sidebar-card {
  background: linear-gradient(180deg, #f0fff4 0%, #dcfce7 100%);
  border: 1px solid #b7ebc6;
  border-radius: 14px;
  padding: 16px;
}
.reference-card {
  margin-top: 12px;
  background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
  border: 1px solid #bfdbfe;
}
.sidebar-card,
.sidebar-card * {
  color: #111827 !important;
}
[data-theme="dark"] .left-pane,
[data-theme="dark"] .sidebar-card,
[data-theme="dark"] .sidebar-card * {
  color: #111827 !important;
}
.hero-card {
  background: linear-gradient(120deg, #dcfce7 0%, #bbf7d0 100%);
  border: 1px solid #86efac;
  border-radius: 14px;
  padding: 14px 18px;
  color: #111827;
  margin-bottom: 12px;
}
.hero-card h1,
.hero-card p {
  color: #111827 !important;
}
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_api_config(project_root: Path) -> Path:
    env_path = os.getenv("API_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    space_cfg = project_root / "configs" / "api_space.yaml"
    if space_cfg.exists():
        return space_cfg
    return project_root / "configs" / "api.yaml"


def _build_predictor() -> Predictor:
    project_root = _project_root()
    api_config_path = _resolve_api_config(project_root)
    api_config = load_api_config(api_config_path, project_root=project_root)

    return Predictor(
        checkpoint_path=api_config.checkpoint_path,
        model_config=load_model_config(api_config.model_config_path),
        data_config=load_data_config(api_config.data_config_path, project_root=project_root),
        inference_config=load_inference_config(api_config.inference_config_path, project_root=project_root),
        cv_proxy_config=load_cv_proxy_config(api_config.cv_proxy_config_path),
    )


def _probability_plot(probabilities: dict[str, float]) -> plt.Figure:
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{code} ({LABEL_DESCRIPTIONS.get(code, code)})" for code, _ in sorted_items]
    values = [value for _, value in sorted_items]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(labels, values, color="#1982a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Per-label Predicted Probability")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    return fig


def _example_rows(project_root: Path) -> list[list[str | int]]:
    images_dir = project_root / "preprocessed_images"
    if not images_dir.exists():
        return []

    age_map: dict[str, int] = {}
    patients_csv = project_root / "data" / "processed" / "patients.csv"
    if patients_csv.exists():
        try:
            patients_df = pd.read_csv(patients_csv)
            for _, row in patients_df.iterrows():
                patient_id = str(row.get("patient_id", "")).strip()
                age_value = row.get("age")
                if patient_id and pd.notna(age_value):
                    age_map[patient_id] = int(round(float(age_value)))
        except Exception:
            age_map = {}

    fallback_ages = [42, 51, 63, 74]
    examples: list[list[str | int]] = []
    for left_path in sorted(images_dir.glob("*_left.jpg"))[:80]:
        patient_id = left_path.stem.replace("_left", "")
        right_path = images_dir / f"{patient_id}_right.jpg"
        if not right_path.exists():
            continue

        default_age = fallback_ages[len(examples) % len(fallback_ages)]
        age = age_map.get(patient_id, default_age)
        examples.append([str(left_path), str(right_path), age])

        if len(examples) >= 4:
            break
    return examples


def create_demo(predictor: Predictor) -> gr.Blocks:
    with gr.Blocks(title="Eye-Heart Connection") as demo:
        with gr.Column(elem_classes=["app-wrap"]):
            gr.HTML(
                """
                <div class='hero-card'>
                  <h1 style='margin:0;'>Eye-Heart Connection</h1>
                  <p style='margin:6px 0 0;'>Multimodal retinal AI to estimate cardiovascular-risk-related indicators from bilateral fundus images + age.</p>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=2, elem_classes=["left-pane"]):
                    with gr.Group(elem_classes=["sidebar-card"]):
                        gr.Markdown(
                            """
                            ### Project Brief
                            This system predicts **8 ophthalmic diagnostic indicators** from:
                            - Left fundus image
                            - Right fundus image
                            - Patient age

                            The predicted probabilities are converted into a **cardiovascular risk proxy summary**:
                            - Hypertension proxy
                            - Diabetes proxy
                            - Atherosclerotic proxy
                            - Overall cardiovascular proxy + risk band

                            #### Why this works
                            Retinal microvasculature reflects systemic vascular health. Patterns in vessel caliber,
                            tortuosity, and retinal pathology can correlate with cardiometabolic disease signals.

                            ### Tech Stack
                            - Python 3.10+
                            - PyTorch + Torchvision
                            - EfficientNet-B4 backbone
                            - FastAPI (serving API)
                            - Gradio (interactive frontend)
                            - Docker (deployment)

                            ### Core Libraries
                            - `torch`, `torchvision`
                            - `albumentations`, `opencv-python`
                            - `fastapi`, `uvicorn`
                            - `gradio`, `pandas`, `matplotlib`
                            """
                        )

                    with gr.Group(elem_classes=["sidebar-card", "reference-card"]):
                        gr.Markdown(
                            """
                            ### Evidence & References
                            Selected literature supporting retina-cardiovascular linkage and prediction feasibility:

                            1. **Poplin et al. (2018), Nature Biomedical Engineering**
                            Deep learning from retinal fundus images predicted cardiovascular risk factors.
                            https://www.nature.com/articles/s41551-018-0195-0

                            2. **Cheung, Ikram, Sabanayagam, Wong (2012), Nature Reviews Cardiology**
                            Retinal microvasculature as a model for coronary/systemic vascular disease study.
                            https://www.nature.com/articles/nrcardio.2012.191

                            3. **Wong et al. population retinal vascular studies**
                            Retinal vessel changes associated with hypertension/stroke risk.
                            https://pubmed.ncbi.nlm.nih.gov/11377643/

                            4. **Nusinovici et al. cohort epidemiology**
                            Retinal biomarkers associated with cardiovascular outcomes.
                            https://pubmed.ncbi.nlm.nih.gov/31199366/

                            Note: This app is a research-oriented screening aid, not a diagnostic medical device.
                            """
                        )

                with gr.Column(scale=3):
                    with gr.Row():
                        left_image = gr.Image(type="numpy", label="Left Eye Fundus Image")
                        right_image = gr.Image(type="numpy", label="Right Eye Fundus Image")

                    age = gr.Slider(minimum=0, maximum=120, step=1, value=55, label="Patient Age")

                    with gr.Row():
                        run_btn = gr.Button("Run Prediction", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                    with gr.Row():
                        risk_band = gr.Textbox(label="Predicted Risk Band", interactive=False)
                        cardiovascular_proxy = gr.Number(
                            label="Overall Cardiovascular Proxy",
                            interactive=False,
                            precision=4,
                        )

                    risk_message = gr.Textbox(
                        label="Cardiovascular Risk Message",
                        interactive=False,
                        lines=3,
                    )

                    probability_plot = gr.Plot(label="Prediction Profile")
                    probability_table = gr.Dataframe(
                        headers=["Label", "Description", "Probability", "Predicted"],
                        datatype=["str", "str", "number", "number"],
                        label="Detailed Predictions",
                        interactive=False,
                    )
                    cv_summary_json = gr.JSON(label="Cardiovascular Summary Details")

                    examples = _example_rows(_project_root())
                    if examples:
                        gr.Examples(
                            examples=examples,
                            inputs=[left_image, right_image, age],
                            label="Quick Examples",
                        )

        def predict(
            left: object,
            right: object,
            patient_age: float,
        ) -> tuple[str, float, str, plt.Figure, pd.DataFrame, dict[str, float | str]]:
            if left is None or right is None:
                raise gr.Error("Please upload both left and right fundus images.")

            result = predictor.predict_single(left_image=left, right_image=right, age=float(patient_age))

            rows = []
            for label, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                rows.append(
                    {
                        "Label": label,
                        "Description": LABEL_DESCRIPTIONS.get(label, label),
                        "Probability": round(prob, 4),
                        "Predicted": int(result.labels[label]),
                    }
                )

            risk_band_value = str(result.cv_summary.get("risk_band", "unknown")).lower()
            chance = float(result.cv_summary.get("overall_cv_proxy", 0.0))
            chance_pct = chance * 100.0
            guidance = {
                "low": "Don't worry, you currently appear to have a low cardiovascular risk profile.",
                "medium": "You appear to have a moderate cardiovascular risk profile. Please monitor regularly and consider a clinical checkup.",
                "high": "You appear to have a high cardiovascular risk profile. Please consult a doctor soon for proper evaluation.",
            }.get(
                risk_band_value,
                "Please consult a healthcare professional for complete clinical interpretation.",
            )
            message = (
                f"You have {chance_pct:.1f}% predicted chance of cardiovascular disease. "
                f"Risk band: {risk_band_value.upper()}. {guidance}"
            )

            table = pd.DataFrame(rows)
            fig = _probability_plot(result.probabilities)
            return (
                risk_band_value.upper(),
                chance,
                message,
                fig,
                table,
                result.cv_summary,
            )

        run_btn.click(
            fn=predict,
            inputs=[left_image, right_image, age],
            outputs=[
                risk_band,
                cardiovascular_proxy,
                risk_message,
                probability_plot,
                probability_table,
                cv_summary_json,
            ],
        )

        clear_btn.click(
            fn=lambda: (
                None,
                None,
                55,
                "",
                0.0,
                "",
                None,
                pd.DataFrame(columns=["Label", "Description", "Probability", "Predicted"]),
                {},
            ),
            outputs=[
                left_image,
                right_image,
                age,
                risk_band,
                cardiovascular_proxy,
                risk_message,
                probability_plot,
                probability_table,
                cv_summary_json,
            ],
        )

    return demo


def main() -> None:
    predictor = _build_predictor()
    demo = create_demo(predictor)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=APP_THEME,
        css=APP_CSS,
        js=APP_JS,
    )


if __name__ == "__main__":
    main()

