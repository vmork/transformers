from jaxtyping import Float 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from transformers.utils import Model
from transformers.vision.data import CIFAROutput, CIFARDataLoader, CIFARBatch

class VisionModel(Model):
    def __init__(self, n_classes: int=10):
        super().__init__()

    def get_output(self, batch: CIFARBatch) -> CIFAROutput: # type: ignore
        logits = self.forward(batch.x)
        loss = F.cross_entropy(logits, batch.y)
        return CIFAROutput(loss=loss, logits=logits)

    def predict(self, x: Float[Tensor, "B 3 32 32"]):
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def full_eval(self, test_loader: CIFARDataLoader, device: torch.device, idx_to_class: dict[int, str], plot=True):
        self.eval()
        self.to(device)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch: CIFARBatch = batch.to(device)  # type: ignore
                preds = self.predict(batch.x)
                labels = batch.y
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average='macro')

        if plot:
            cm = confusion_matrix(all_labels, all_preds)
            display_labels = [idx_to_class[idx] for idx in range(10)]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=75) # type: ignore
            plt.title("Confusion Matrix")
            plt.show()
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score (macro): {f1*100:.2f}")
        
        self.train()
        return accuracy, f1

class BadNet(VisionModel):
    """
    From a pytorch [tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    """
    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.head = nn.Linear(84, n_classes)
    
    def forward(self, x: Float[Tensor, "B 3 32 32"]) -> Float[Tensor, "B C"]:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x