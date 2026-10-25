from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {'train_loss': '#1f77b4', 'val_loss': '#ff7f0e', 'train_acc': '#d62728', 'val_acc': '#2ca02c'}


def plot_training(
    hist_train_loss, hist_val_loss,
    hist_train_acc, hist_val_acc,
    best_epoch, live=False
):
    if live:
        from IPython.display import clear_output
        clear_output(wait=True)
    epochs = range(1, len(hist_train_loss) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, hist_train_loss, label='train loss', color=COLORS['train_loss'], linewidth=2)
    if hist_val_loss:
        ax_loss.plot(epochs, hist_val_loss, label='val loss', color=COLORS['val_loss'], linewidth=2)
        ax_loss.axvline(best_epoch, linestyle='--', alpha=0.5, label=f'best={best_epoch}')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    ax_acc.plot(epochs, hist_train_acc, label='train acc', color=COLORS['train_acc'], linewidth=2)
    if hist_val_acc:
        ax_acc.plot(epochs, hist_val_acc, label='val acc', color=COLORS['val_acc'], linewidth=2)
        ax_acc.axvline(best_epoch, linestyle='--', alpha=0.5)
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def train(
    model, optimizer, loss_func, dl_train, dl_val, path, device,
    EPOCH_COUNT=20, plot=True, log_every=10, patience=50,
    train_preprocess=None, aim_run=None, aim_url="http://127.0.0.1:43800",
):
    """train_preprocess обрабатывает только train после переноса батча на device.

    dl_val должен выдавать уже подготовленные входы для модели.
    При dl_val=None обучаем все EPOCH_COUNT эпох и сохраняем веса один раз в конце.
    aim_run записывает метрики каждой эпохи и итог
    """

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if aim_run is not None:
        print(f"Метрики Aim: {aim_url} (интерфейс Aim нужно запустить отдельно)")

    best_val_accuracy = -1
    best_epoch = -1
    epochs_without_improvement = 0
    hist_train_loss, hist_val_loss, hist_train_acc, hist_val_acc = [], [], [], []

    for epoch in range(1, EPOCH_COUNT+1):
        model.train()
        train_loss_sum, train_n, train_correct, train_total = 0.0, 0, 0, 0
        for batch_x, batch_y in dl_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if train_preprocess is not None:
                batch_x = train_preprocess(batch_x)
            y_pred = model(batch_x)
            loss = loss_func(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_x.size(0)
            train_n += batch_x.size(0)
            pred_indices = y_pred.argmax(dim=1)
            train_correct += (pred_indices == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss = train_loss_sum / train_n
        train_acc = train_correct / train_total

        hist_train_loss.append(train_loss)
        hist_train_acc.append(train_acc)

        if aim_run is not None:
            aim_run.track(train_loss, name="loss", step=epoch, context={"subset": "train"})
            aim_run.track(train_acc, name="accuracy", step=epoch, context={"subset": "train"})

        stop_training = False

        if dl_val is not None:
            model.eval()
            val_loss_sum, val_n, correct, total = 0.0, 0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in dl_val:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    y_logits = model(batch_x)
                    loss = loss_func(y_logits, batch_y)

                    val_loss_sum += loss.item() * batch_x.size(0)
                    val_n += batch_x.size(0)

                    pred_indices = y_logits.argmax(dim=1)

                    correct += (pred_indices == batch_y).sum().item()
                    total += batch_y.size(0)

            val_loss = val_loss_sum / val_n
            val_acc = correct / total

            hist_val_loss.append(val_loss)
            hist_val_acc.append(val_acc)

            if aim_run is not None:
                aim_run.track(val_loss, name="loss", step=epoch, context={"subset": "val"})
                aim_run.track(val_acc, name="accuracy", step=epoch, context={"subset": "val"})

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), path)
            else:
                epochs_without_improvement += 1

            stop_training = epochs_without_improvement >= patience

        if epoch == 1 or epoch % log_every == 0 or epoch == EPOCH_COUNT or stop_training:
            if plot:
                plot_training(
                    hist_train_loss,
                    hist_val_loss,
                    hist_train_acc,
                    hist_val_acc,
                    best_epoch, 
                    live=True
                )
                if aim_run is not None:
                    print(f"Метрики Aim: {aim_url} (интерфейс Aim нужно запустить отдельно)")

        if stop_training:
            print(f"Ранняя остановка на эпохе {epoch}: val_acc не улучшалась {patience} эпох")
            break

    if dl_val is None:
        torch.save(model.state_dict(), path)
        if aim_run is not None:
            aim_run["results"] = {
                "train_accuracy": train_acc,
                "last_epoch": epoch,
                "weights_path": str(path),
            }
        print(f'Обучение завершено: {EPOCH_COUNT} эпох, train_acc = {train_acc:.4f}')
        return pd.DataFrame([train_acc, epoch])

    if aim_run is not None:
        aim_run["results"] = {
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "last_epoch": epoch,
            "weights_path": str(path),
        }
    print(f'Best epoch = {best_epoch}, best val_acc = {best_val_accuracy:.4f}')

    return pd.DataFrame([best_val_accuracy, best_epoch])

def predict_submission(
    model, dl_test, class_names, sample_path, output_path, device,
    preprocess=None,
):
    """preprocess обрабатывает test после переноса батча на device"""
    model.eval()
    predicted_classes = []

    with torch.inference_mode():
        for batch_x in dl_test:
            batch_x = batch_x.to(device)
            if preprocess is not None:
                batch_x = preprocess(batch_x)
            logits = model(batch_x)
            pred_indices = logits.argmax(dim=1).cpu().tolist()
            predicted_classes.extend(class_names[i] for i in pred_indices)

    submission = pd.read_csv(sample_path)

    if len(predicted_classes) != len(submission):
        raise ValueError("Число предсказаний не совпадает с числом строк submission")

    submission["label"] = predicted_classes
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
