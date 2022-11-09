import torch
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Union, Optional

from fastai.vision.all import Learner, DataLoader

from .wndataset import WNDataSet
from .wnbaseimage import WNBaseImage


class WNVisualizer:
    def __init__(self, dl: DataLoader, learner: Learner) -> None:
        self.dl = dl
        self.learner = learner

    def predict_item(self, idx) -> dict:
        item, target = self.dl.create_item(idx)
        prob = self.learner.model(torch.tensor(item[None, ...]))
        pred = prob.squeeze().argmax(0)

        return dict(
            item=item,
            target=target,
            probabilities=prob.detach().numpy(),
            prediction=pred.detach().numpy(),
        )

    def show_pred(self, idx: int, axs: Optional[np.ndarray] = None, size: int = 5):

        if axs is None:
            _, axs = plt.subplots(1, 3, figsize=(3 * size, size))  # type: ignore

        # Get the predictions
        predictions = self.predict_item(idx)

        # plot the three contents (item, target and prediction)
        WNBaseImage._plot(predictions["item"].squeeze(), ax=axs[0])  # type: ignore
        WNBaseImage._plot(predictions["target"].squeeze(), ax=axs[1])  # type: ignore
        WNBaseImage._plot(predictions["prediction"].squeeze(), ax=axs[2])  # type: ignore
