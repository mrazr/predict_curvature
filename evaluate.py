import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt

from ctc_dataset import CTCDataset
from model import Model


if __name__ == '__main__':
    ds = CTCDataset('E:/CTC/DIC-C2DH-HeLa/', 'curv', train=False)

    dl = data.DataLoader(ds, batch_size=4, shuffle=True)

    net = Model()
    net.load_state_dict(torch.load('best_curvature_model2_epoch23.pth'))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(dev)

    imgs, anns = next(iter(dl))
    imgs_ = imgs.to(dev)
    anns = anns

    net.eval()
    with torch.no_grad():
        preds = net(imgs_).to('cpu')

        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(np.squeeze(imgs[0].numpy()))
        ax[1].imshow(np.squeeze(anns[0].numpy()))
        ax[2].imshow(np.squeeze(preds[0].numpy()))
        # plt.colorbar()
        plt.show()





