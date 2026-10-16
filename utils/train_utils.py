import torch
import matplotlib.pyplot as plt
import pandas as pd

EPOCH_COUNT = 20

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
    ax_loss.plot(epochs, hist_val_loss, label='val loss', color=COLORS['val_loss'], linewidth=2)
    ax_loss.axvline(best_epoch, linestyle='--', alpha=0.5, label=f'best={best_epoch}')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    ax_acc.plot(epochs, hist_train_acc, label='train acc', color=COLORS['train_acc'], linewidth=2)
    ax_acc.plot(epochs, hist_val_acc, label='val acc', color=COLORS['val_acc'], linewidth=2)
    ax_acc.axvline(best_epoch, linestyle='--', alpha=0.5)
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def train(model, optimizer, loss_func, dl_train, dl_val, path, device, EPOCH_COUNT=EPOCH_COUNT, plot=True, log_every=10):
    best_val_accuracy = -1
    best_epoch = -1
    hist_train_loss, hist_val_loss, hist_train_acc, hist_val_acc = [], [], [], []

    for epoch in range(1, EPOCH_COUNT+1):
        model.train()
        train_loss_sum, train_n, train_correct, train_total = 0.0, 0, 0, 0
        for batch_x, batch_y in dl_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x)
            loss = loss_func(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_x.size(0)
            train_n += batch_x.size(0)
            train_correct += (y_pred.argmax(dim=1) == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss = train_loss_sum / train_n
        train_acc = train_correct / train_total

        model.eval()
        val_loss_sum, val_n, correct, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in dl_val:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_logits = model(batch_x)
                loss = loss_func(y_logits, batch_y)

                val_loss_sum += loss.item() * batch_x.size(0)
                val_n += batch_x.size(0)
                
                pred = y_logits.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss = val_loss_sum / val_n
        val_acc = correct / total

        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_train_acc.append(train_acc)
        hist_val_acc.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), path)

        if epoch == 1 or epoch % log_every == 0 or epoch == EPOCH_COUNT:
            # print(f'[epoch={epoch:3d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
            if plot:
                plot_training(
                            hist_train_loss, hist_val_loss,
                            hist_train_acc, hist_val_acc,
                            best_epoch, live=True
                            )

    print(f'Best epoch = {best_epoch}, best val_acc = {best_val_accuracy:.4f}')

    return pd.DataFrame([best_val_accuracy, best_epoch])
                
        