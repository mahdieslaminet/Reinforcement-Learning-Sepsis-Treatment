import gradio as gr
import matplotlib.pyplot as plt
import tempfile
import os
from argparse import Namespace

from train import train_main
from data_loader import load_trajectories_npz
from eval_offpolicy import wis_estimate, fit_behavior_model
from agent import A2CAgent

DEFAULT_DATASET = "synthetic_clinical_trajectories.npz"


def run_training_ui(data_file, epochs, lr, gamma, seed):
    if data_file is None:
        data_path = DEFAULT_DATASET
        status_msg = "ℹ No dataset uploaded — using default synthetic dataset."
    else:
        data_path = data_file.name
        status_msg = "✅ Using uploaded dataset."

    args = Namespace(
        data=data_path,
        out_dir="checkpoints",
        input_dim=379,
        epochs=int(epochs),
        lr=float(lr),
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=float(gamma),
        val_every=5,
        seed=int(seed),
        cpu=False
    )

    train_losses, val_losses = train_main(args, return_metrics=True)

    # Plot losses
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    return status_msg + " Training finished.", fig



def run_evaluation_ui(data_file):
    if data_file is None:
        data_path = DEFAULT_DATASET
    else:
        data_path = data_file.name

    trajs = load_trajectories_npz(data_path)

    rf, prob_fn = fit_behavior_model(trajs)

    agent = A2CAgent(input_dim=379)
    agent.load("checkpoints/best_model.pth")

    results = wis_estimate(agent, trajs, behavior_prob_fn=prob_fn)

    return (
        f"WIS Estimate: {results['wis']:.4f}\n"
        f"95% Lower Bound: {results['lower95']:.4f}"
    )



with gr.Blocks(title="Off-policy RL Clinical Decision UI") as demo:
    gr.Markdown("## 🧠 Off-policy RL Training & Evaluation")
    gr.Markdown("📁 If no dataset is uploaded, a default synthetic dataset will be used automatically.")


    with gr.Row():
        data_file = gr.File(label="Upload Trajectory Dataset (.npz)")

    with gr.Row():
        epochs = gr.Slider(10, 300, value=50, step=10, label="Epochs")
        lr = gr.Number(value=3e-4, label="Learning Rate")
        gamma = gr.Slider(0.9, 0.999, value=0.99, label="Gamma")
        seed = gr.Number(value=42, label="Random Seed")

    train_btn = gr.Button("🚀 Train Model")
    status = gr.Textbox(label="Status")
    loss_plot = gr.Plot(label="Training Curves")

    train_btn.click(
        run_training_ui,
        inputs=[data_file, epochs, lr, gamma, seed],
        outputs=[status, loss_plot]
    )

    gr.Markdown("## 📊 Off-policy Evaluation")

    eval_btn = gr.Button("Run WIS Evaluation")
    eval_out = gr.Textbox(label="Evaluation Result")

    eval_btn.click(
        run_evaluation_ui,
        inputs=[data_file],
        outputs=eval_out
    )

demo.launch(share=True, debug=True)

