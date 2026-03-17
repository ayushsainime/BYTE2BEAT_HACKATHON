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

    examples: list[list[str | int]] = []
    for left_path in sorted(images_dir.glob("*_left.jpg"))[:40]:
        patient_id = left_path.stem.replace("_left", "")
        right_path = images_dir / f"{patient_id}_right.jpg"
        if right_path.exists():
            examples.append([str(left_path), str(right_path), 55])
        if len(examples) >= 4:
            break
    return examples


def create_demo(predictor: Predictor) -> gr.Blocks:
    css = """
    .app-wrap { max-width: 1200px !important; margin: 0 auto; }
    .sidebar-card {
      background: linear-gradient(180deg, #f6fbff 0%, #eef7ff 100%);
      border: 1px solid #d5e9f7;
      border-radius: 14px;
      padding: 16px;
    }
    .hero-card {
      background: linear-gradient(120deg, #0b4f6c 0%, #1f7a8c 100%);
      border-radius: 14px;
      padding: 14px 18px;
      color: white;
      margin-bottom: 12px;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="sky",
            neutral_hue="slate",
            radius_size="lg",
        ),
        css=css,
        title="Eye-Heart Connection",
    ) as demo:
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
                with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                    gr.Markdown(
                        """
                        ### Project Brief
                        This system predicts **8 ophthalmic diagnostic indicators** from:
                        - Left fundus image
                        - Right fundus image
                        - Patient age

                        The predicted probabilities are converted to a **CV risk proxy summary**.

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

                with gr.Column(scale=2):
                    with gr.Row():
                        left_image = gr.Image(type="numpy", label="Left Eye Fundus Image")
                        right_image = gr.Image(type="numpy", label="Right Eye Fundus Image")

                    age = gr.Slider(minimum=0, maximum=120, step=1, value=55, label="Patient Age")

                    with gr.Row():
                        run_btn = gr.Button("Run Prediction", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                    with gr.Row():
                        risk_band = gr.Textbox(label="Predicted Risk Band", interactive=False)
                        cv_proxy = gr.Number(label="Overall CV Proxy", interactive=False, precision=4)

                    probability_plot = gr.Plot(label="Prediction Profile")
                    probability_table = gr.Dataframe(
                        headers=["Label", "Description", "Probability", "Predicted"],
                        datatype=["str", "str", "number", "number"],
                        label="Detailed Predictions",
                        interactive=False,
                    )
                    cv_summary_json = gr.JSON(label="CV Summary Details")

                    examples = _example_rows(_project_root())
                    if examples:
                        gr.Examples(
                            examples=examples,
                            inputs=[left_image, right_image, age],
                            label="Quick Examples",
                        )

        def predict(left: object, right: object, patient_age: float) -> tuple[str, float, plt.Figure, pd.DataFrame, dict[str, float | str]]:
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

            table = pd.DataFrame(rows)
            fig = _probability_plot(result.probabilities)
            return (
                str(result.cv_summary.get("risk_band", "unknown")),
                float(result.cv_summary.get("overall_cv_proxy", 0.0)),
                fig,
                table,
                result.cv_summary,
            )

        run_btn.click(
            fn=predict,
            inputs=[left_image, right_image, age],
            outputs=[risk_band, cv_proxy, probability_plot, probability_table, cv_summary_json],
        )

        clear_btn.click(
            fn=lambda: (None, None, 55, "", 0.0, None, pd.DataFrame(columns=["Label", "Description", "Probability", "Predicted"]), {}),
            outputs=[left_image, right_image, age, risk_band, cv_proxy, probability_plot, probability_table, cv_summary_json],
        )

    return demo


def main() -> None:
    predictor = _build_predictor()
    demo = create_demo(predictor)
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()

