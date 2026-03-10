from train_ehr_federated import MedicalModel,FocalLoss
from opacus import PrivacyEngine
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate_privacy(noise_values,X,y):

```
results = []

for noise in noise_values:

    model = MedicalModel(X.shape[1])

    optimizer = optim.AdamW(model.parameters(),lr=1e-3)

    privacy_engine = PrivacyEngine()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.Tensor(X),
            torch.LongTensor(y)
        ),
        batch_size=64
    )

    model,optimizer,loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise,
        max_grad_norm=1.0
    )

    criterion = FocalLoss()

    for _ in range(3):

        for data,target in loader:

            optimizer.zero_grad()

            loss = criterion(model(data),target)

            loss.backward()

            optimizer.step()

    eps = privacy_engine.get_epsilon(delta=1e-5)

    model.eval()

    with torch.no_grad():

        p = torch.softmax(model(torch.Tensor(X)),dim=1)[:,1].numpy()

        auc = roc_auc_score(y,p)

    results.append((noise,eps,auc))

return results
```
