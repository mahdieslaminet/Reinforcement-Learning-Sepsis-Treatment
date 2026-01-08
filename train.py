import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from data_loader import load_trajectories_npz, build_synthetic_demo
from agent import A2CAgent


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_main(args, return_metrics=False):
    """
    Main training function.
    If return_metrics=True, returns (train_losses, val_losses)
    for UI visualization.
    """

    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # -----------------------------
    # Load dataset
    # -----------------------------
    if args.data is None:
        print("⚠ No dataset provided. Using synthetic demo dataset.")
        trajs = build_synthetic_demo(
            num_traj=500,
            n_features=args.input_dim,
            seed=args.seed
        )
    else:
        trajs = load_trajectories_npz(args.data)

    # -----------------------------
    # Train / Val / Test split
    # -----------------------------
    random.shuffle(trajs)
    n = len(trajs)

    train_trajs = trajs[:int(0.7 * n)]
    val_trajs = trajs[int(0.7 * n):int(0.9 * n)]
    test_trajs = trajs[int(0.9 * n):]

    print(f"Trajectories → Train: {len(train_trajs)}, "
          f"Val: {len(val_trajs)}, Test: {len(test_trajs)}")

    # -----------------------------
    # Initialize agent
    # -----------------------------
    agent = A2CAgent(
        input_dim=args.input_dim,
        n_actions=5,
        device=device,
        lr=args.lr,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma
    )

    os.makedirs(args.out_dir, exist_ok=True)

    best_val_loss = float('inf')
    train_losses = []
    val_losses_all = []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_trajs)
        epoch_losses = []

        agent.model.train()
        for traj in tqdm(train_trajs, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            metrics = agent.update_from_trajectory(traj)
            epoch_losses.append(metrics['loss'])

        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        print(f"[Epoch {epoch}] Train Loss: {mean_train_loss:.6f}")

        # -----------------------------
        # Validation
        # -----------------------------
        if epoch % args.val_every == 0:
            agent.model.eval()
            val_losses = []

            with torch.no_grad():
                for traj in val_trajs:
                    states = torch.tensor(
                        traj['states'],
                        dtype=torch.float32,
                        device=device
                    )
                    _, values = agent.model(states)

                    values_np = values.cpu().numpy()
                    next_v = 0.0
                    returns, _ = agent.compute_returns_and_advantages(
                        traj['rewards'],
                        traj['dones'],
                        values_np,
                        next_v
                    )

                    val_loss = np.mean((values_np - returns) ** 2)
                    val_losses.append(val_loss)

            mean_val_loss = float(np.mean(val_losses))
            val_losses_all.append(mean_val_loss)

            print(f"           Validation Value MSE: {mean_val_loss:.6f}")

            # Save checkpoint
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pth")
            agent.save(ckpt_path)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                agent.save(os.path.join(args.out_dir, "best_model.pth"))
                print("           ⭐ Best model updated")

    print("✅ Training completed")
    print("Best validation loss:", best_val_loss)

    if return_metrics:
        return train_losses, val_losses_all


# ----------------------------------------------------
# CLI ENTRY POINT (kept for backward compatibility)
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Training Script")

    parser.add_argument('--data', type=str, default=None,
                        help='Path to .npz trajectory dataset')
    parser.add_argument('--out_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--input_dim', type=int, default=379,
                        help='Number of input features')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')

    args = parser.parse_args()

    train_main(args)
