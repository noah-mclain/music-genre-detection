import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ModelComparison:
    def __init__(self, device, results_dir="results"):
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Track metrics for both models
        self.metrics = {
            "simple": {
                "train_losses": [],
                "train_accs": [],
                "val_losses": [],
                "val_accs": [],
                "best_val_acc": 0.0,
                "best_epoch": 0,
                "total_params": 0,
                "trainable_params": 0,
            },
            "complex": {
                "train_losses": [],
                "train_accs": [],
                "val_losses": [],
                "val_accs": [],
                "best_val_acc": 0.0,
                "best_epoch": 0,
                "total_params": 0,
                "trainable_params": 0,
            },
        }

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def log_model_info(self, model, model_name):
        total_params, trainable_params = self.count_parameters(model)

        logger.info(f"\n{'-'*70}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'-'*70}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Model Architecture:\n{model}")
        logger.info(f"{'-'*70}\n")

        # Store metrics
        metric_key = "simple" if "simple" in model_name.lower() else "complex"
        self.metrics[metric_key]["total_params"] = total_params
        self.metrics[metric_key]["trainable_params"] = trainable_params

    def train_epoch(self, model, train_loader, criterion, optimizer, model_name):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"{model_name} | Training epoch...")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 2 == 0:
                batch_loss = running_loss / total
                batch_acc = correct / total
                logger.debug(
                    f"{model_name} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        return val_loss, val_acc

    def train_model(
        self,
        model,
        model_name,
        train_loader,
        val_loader,
        num_epochs,
        learning_rate,
        weight_decay,
        save_path,
        pretrained_path=None,
    ):
        if pretrained_path and Path(pretrained_path).exists():
            try:
                model.load_state_dict(torch.load(pretrained_path, weights_only=True))
                logger.info(f"Loaded pretrained {model_name} from {pretrained_path}")
            except Exception:
                logger.info(f"No pretrained model found for {model_name}, training from scratch")

        logger.info(f"\n{'-'*70}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'-'*70}\n")

        self.log_model_info(model, model_name)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_val_acc = 0.0
        best_epoch = 0
        patience = 30
        patience_counter = 0

        epoch_pbar = tqdm(total=num_epochs, desc=f"{model_name} Epochs", leave=True)

        for epoch in range(num_epochs):
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
            running_loss = 0.0
            correct = 0
            total = 0
            
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Running metrics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Live batch metrics for progress bar
                batch_loss = loss.item()
                batch_acc = (predicted == labels).float().mean().item()
                
                # Update batch progress bar
                train_pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'Acc': f'{batch_acc:.3f}'
                })
            
            train_pbar.close()
            
            # Epoch training metrics
            train_loss = running_loss / total
            train_acc = correct / total

            # Validation (silent)
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            # Track metrics
            metric_key = "simple" if "simple" in model_name.lower() else "complex"
            self.metrics[metric_key]["train_losses"].append(train_loss)
            self.metrics[metric_key]["train_accs"].append(train_acc)
            self.metrics[metric_key]["val_losses"].append(val_loss)
            self.metrics[metric_key]["val_accs"].append(val_acc)
            self.metrics[metric_key]["best_val_acc"] = best_val_acc
            self.metrics[metric_key]["best_epoch"] = best_epoch

            logger.info(
                f"{model_name} | Epoch {epoch+1} COMPLETE | Progress: {epoch+1}/{num_epochs}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), f"{save_path}_best.pth")
                logger.info(
                    f"{model_name} | Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.2e} | ✓ BEST"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"{model_name} | Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"{model_name} | Early stopping at epoch {epoch+1}")
                break

        epoch_pbar.set_postfix({
            'TrainLoss': f'{train_loss:.4f}',
            'TrainAcc': f'{train_acc:.3f}',
            'ValLoss': f'{val_loss:.4f}',
            'ValAcc': f'{val_acc:.3f}'
        })
        epoch_pbar.update(1)

        epoch_pbar.close()

        # Store final metrics
        metric_key = "simple" if "simple" in model_name.lower() else "complex"
        self.metrics[metric_key]["best_val_acc"] = best_val_acc
        self.metrics[metric_key]["best_epoch"] = best_epoch

        logger.info(f"\n{'-'*70}")
        logger.info(f"{model_name} Training Complete")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        logger.info(f"{'-'*70}\n")

        torch.save(model.state_dict(), save_path)
        return model

    def generate_comparison_report(self):
        logger.info(f"\n{'-'*70}")
        logger.info("MODEL COMPARISON RESULTS")
        logger.info(f"{'-'*70}\n")

        # Parameter comparison
        logger.info("PARAMETER COMPARISON:")
        logger.info(f"{'Metric':<30} {'SimpleModel':<20} {'ComplexModel':<20}")
        logger.info(f"{'-'*70}")

        simple_total = self.metrics["simple"]["total_params"]
        complex_total = self.metrics["complex"]["total_params"]
        param_ratio = complex_total / simple_total if simple_total > 0 else 0

        logger.info(f"{'Total Parameters':<30} {simple_total:<20,} {complex_total:<20,}")
        logger.info(
            f"{'Trainable Parameters':<30} {self.metrics['simple']['trainable_params']:<20,} {self.metrics['complex']['trainable_params']:<20,}"
        )
        logger.info(f"{'Complexity Ratio':<30} {'1.0x':<20} {f'{param_ratio:.2f}x':<20}")

        logger.info(f"\n{'PERFORMANCE COMPARISON:'}")
        logger.info(f"{'Metric':<30} {'SimpleModel':<20} {'ComplexModel':<20}")
        logger.info(f"{'-'*70}")

        simple_best = self.metrics["simple"]["best_val_acc"]
        complex_best = self.metrics["complex"]["best_val_acc"]
        accuracy_diff = complex_best - simple_best

        logger.info(f"{'Best Val Accuracy':<30} {simple_best:.4f}{'':<14} {complex_best:.4f}")
        logger.info(
            f"{'Best Epoch':<30} {self.metrics['simple']['best_epoch']:<20} {self.metrics['complex']['best_epoch']:<20}"
        )

        if simple_best > 0:
            improvement_pct = (accuracy_diff / simple_best) * 100
            logger.info(
                f"{'Accuracy Improvement':<30} {'':<20} {f'+{improvement_pct:.2f}%' if improvement_pct > 0 else f'{improvement_pct:.2f}%':<20}"
            )

        logger.info(f"\n{'CONVERGENCE ANALYSIS:'}")
        logger.info(f"{'Metric':<30} {'SimpleModel':<20} {'ComplexModel':<20}")
        logger.info(f"{'-'*70}")

        simple_final_train_acc = (
            self.metrics["simple"]["train_accs"][-1] if self.metrics["simple"]["train_accs"] else 0
        )
        complex_final_train_acc = (
            self.metrics["complex"]["train_accs"][-1]
            if self.metrics["complex"]["train_accs"]
            else 0
        )

        logger.info(
            f"{'Final Train Accuracy':<30} {simple_final_train_acc:.4f}{'':<14} {complex_final_train_acc:.4f}"
        )
        logger.info(
            f"{'Final Val Accuracy':<30} {self.metrics['simple']['val_accs'][-1]:.4f}{'':<14} {self.metrics['complex']['val_accs'][-1]:.4f}"
        )

        logger.info(f"{'-'*70}\n")

    def save_results_json(self):
        results_file = (
            self.results_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        return results_file
