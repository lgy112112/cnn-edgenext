import torch
from tqdm import tqdm
import wandb

__all__ = ["train_one_epoch", "validate_one_epoch", "train_and_validate"]

def train_one_epoch(model, dataloader, criterion, optimizer, device, wandb_log=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 创建 tqdm 进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        images, labels = batch
        cwt_images = images['CWT'].to(device)
        stft_images = images['STFT'].to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(cwt_images, stft_images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 更新 tqdm 的附加信息
        avg_loss = total_loss / (total // labels.size(0))
        accuracy = correct / total
        progress_bar.set_postfix(loss=avg_loss, acc=accuracy)

        # 记录到 wandb
        if wandb_log:
            wandb.log({"Train Loss (Batch)": loss.item(), "Train Accuracy (Batch)": accuracy})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    # print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}")

    # 记录到 wandb
    if wandb_log:
        wandb.log({"Train Loss (Epoch)": avg_loss, "Train Accuracy (Epoch)": accuracy})
    
    return avg_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device, wandb_log=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 创建 tqdm 进度条
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            images, labels = batch
            cwt_images = images['CWT'].to(device)
            stft_images = images['STFT'].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(cwt_images, stft_images)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 更新 tqdm 的附加信息
            avg_loss = total_loss / (total // labels.size(0))
            accuracy = correct / total
            progress_bar.set_postfix(loss=avg_loss, acc=accuracy)

            # 记录到 wandb
            if wandb_log:
                wandb.log({"Validation Loss (Batch)": loss.item(), "Validation Accuracy (Batch)": accuracy})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    # print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    # 记录到 wandb
    if wandb_log:
        wandb.log({"Validation Loss (Epoch)": avg_loss, "Validation Accuracy (Epoch)": accuracy})
    
    return avg_loss, accuracy


def train_and_validate(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10, 
    save_path="best_model.pth",
    wandb_log=False
):
    if wandb_log:
        wandb.init(project="radar", name="model_training", config={
            "epochs": num_epochs,
            "learning_rate": optimizer.defaults["lr"],
            "device": device.type
        })

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, wandb_log)

        # Validate for one epoch
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device, wandb_log)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with Validation Accuracy: {val_acc:.4f}")

        # Print summary for this epoch
        # print(f"Epoch [{epoch + 1}/{num_epochs}] Summary:")
        # print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    if wandb_log:
        wandb.finish()
