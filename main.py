from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import wandb
import torchmetrics

from model import Model, Encoder
from ctc_dataset import CTCDataset


def weighted_mse(pred, target):
    diff = torch.square(pred - target)
    weight = 10.0 * torch.abs(target) + 0.1

    return torch.mean(torch.sum(weight * diff))


if __name__ == '__main__':
    enc = Encoder()
    print(enc)
    summary(enc, (1, 256, 256), device='cpu')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Model().to(dev)
    summary(net, (1, 256, 256), device=dev)

    ds = CTCDataset('E:/CTC/DIC-C2DH-HeLa', 'curv', train=True)
    train_ds, val_ds = data.random_split(ds, [0.8, 0.2])
    test_ds = CTCDataset('E:/CTC/DIC-C2DH-HeLa', 'curv', train=False)

    config = {
        'epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 4
    }

    wandb.init(project='predict_curvature', config=config)

    train_batch_size = config['batch_size']
    val_batch_size = 8

    train_dl = data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_dl = data.DataLoader(val_ds, batch_size=val_batch_size)

    n_train_batches = len(train_dl)
    n_val_batches = len(val_dl)

    test_dl = data.DataLoader(test_ds, batch_size=8)

    adam = optim.Adam(net.parameters(), lr=config['learning_rate'])
    loss_fn = weighted_mse

    epochs = config['epochs']

    writer = SummaryWriter()

    print(f'Train dataset size = {len(train_ds)}')
    print(f'Validation dataset size = {len(val_ds)}')
    print(f'Test dataset size = {len(test_ds)}')

    best_val_loss = 999999

    metric = torchmetrics.MeanSquaredError().to(dev)

    for epoch in tqdm(range(epochs), desc='Epochs'):
        net.train()
        train_loss_per_epoch = 0.0
        train_acc_per_epoch = 0.0
        metric.reset()
        for imgs, anns in tqdm(train_dl, desc='Training'):
            adam.zero_grad()
            imgs, anns = imgs.to(dev), anns.to(dev)

            preds = net(imgs)
            train_loss = loss_fn(preds, anns)

            train_loss.backward()
            adam.step()

            train_loss_per_epoch += train_loss.detach().item()

            metric(preds, anns)
        train_acc_per_epoch = metric.compute()
        metric.reset()

        val_loss_per_epoch = 0.0
        val_accuracy_per_epoch = 0.0
        net.eval()
        with torch.no_grad():
            for imgs, anns in tqdm(val_dl, desc='Validation'):
                imgs, anns = imgs.to(dev), anns.to(dev)

                preds = net(imgs)
                val_loss = loss_fn(preds, anns)
                val_loss_per_epoch += val_loss.detach().item()
                metric(preds, anns)
        val_accuracy_per_epoch = metric.compute()
        metric.reset()

        train_loss_per_epoch /= n_train_batches
        val_loss_per_epoch /= n_val_batches

        print(f'Epoch {epoch}: train loss = {train_loss_per_epoch}\tval loss = {val_loss_per_epoch}')
        writer.add_scalar('Loss/train', train_loss_per_epoch, epoch)
        writer.add_scalar('Loss/val', val_loss_per_epoch, epoch)
        writer.add_scalars('Loss', {'train': train_loss_per_epoch, 'val': val_loss_per_epoch}, global_step=epoch)

        wandb.log({
            'loss/train': train_loss_per_epoch,
            'loss/val': val_loss_per_epoch,
            'accuracy/train': train_acc_per_epoch,
            'accuracy/val': val_accuracy_per_epoch
        })

        if val_loss_per_epoch < best_val_loss:
            torch.save(net.state_dict(), f'best_curvature_model_3x3conv_epoch{epoch}.pth')
            best_val_loss = val_loss_per_epoch
