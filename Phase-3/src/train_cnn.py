import sys
from pathlib import Path
import csv
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import get_data_loaders
from src.models.skin_cancer_cnn import SkinCancerCNN, count_parameters
from src.utils.seed import seed_everything


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def calculate_gradient_norm(model: nn.Module) -> float:
    """Calculate gradient norm for logging."""
    total_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is not None:
            param_norm = parameter.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def run_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> tuple[float, float, float]:
    """Run one training or validation epoch."""

    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    grad_steps = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                grad_norm = calculate_gradient_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()

                total_grad_norm += grad_norm
                grad_steps += 1

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    avg_grad_norm = total_grad_norm / grad_steps if grad_steps > 0 else 0.0

    return epoch_loss, epoch_accuracy, avg_grad_norm


def train_cnn(config_path: str) -> None:
    """Train CNN model using config file."""

    config = load_config(config_path)

    seed = config["training"]["seed"]
    seed_everything(seed)

    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]
    val_split = config["data"]["val_split"]

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]
    gradient_clip = config["training"]["gradient_clip"]

    dropout = config["model"]["dropout"]
    use_batchnorm = config["model"]["use_batchnorm"]
    use_residual = config["model"]["use_residual"]

    log_dir = PROJECT_ROOT / config["paths"]["log_dir"]
    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "cnn_v1_log.csv"
    checkpoint_path = checkpoint_dir / "cnn_v1_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        val_split=val_split,
        seed=seed,
    )

    model = SkinCancerCNN(
        num_classes=len(class_names),
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        use_residual=use_residual,
    ).to(device)

    print("\nModel Architecture:")
    print(model)
    print("\nTrainable Parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.5,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "gradient_norm",
        ])

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc, grad_norm = run_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gradient_clip=gradient_clip,
            )

            val_loss, val_acc, _ = run_one_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                gradient_clip=gradient_clip,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                grad_norm,
            ])

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Grad Norm: {grad_norm:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "class_names": class_names,
                        "config": config,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    checkpoint_path,
                )

                print("Best CNN model saved.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            scheduler.step()

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining completed.")
    print("Best checkpoint saved at:", checkpoint_path)
    print("Training log saved at:", log_path)


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "configs" / "cnn_v1.yaml"
    train_cnn(str(config_file))