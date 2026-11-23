import math

import torch


def run_epoch(model, loader, criterion, device, *, optimizer=None):
    """Return sample-weighted mean loss, accuracy in [0, 1], and examples seen."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                raise FloatingPointError(f"Non-finite loss in batch {step}.")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += batch_loss * batch_size
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            seen += batch_size

            if training and step % 100 == 0:
                print(
                    f"Training batch {step}/{len(loader)} | mean loss so far {total_loss / seen:.4f}"
                )

    if seen == 0:
        raise ValueError("The loader produced no examples.")
    return {"loss": total_loss / seen, "accuracy": correct / seen, "seen": seen}
