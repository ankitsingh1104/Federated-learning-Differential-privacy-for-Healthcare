import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
import copy
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLIENTS = 3
ROUNDS = 30
DIRICHLET_ALPHA = 0.5
MU = 0.001
BASE_LR = 2e-3
TARGET_DELTA = 1e-5

CLASS_WEIGHTS = torch.tensor([1.0, 2.5]).to(DEVICE)

DATA_PATH = "data/xray_dataset_covid19"

def dirichlet_partition(dataset, n_clients, alpha):

```
n_classes = len(dataset.classes)

label_indices = [
    np.where(np.array(dataset.targets) == i)[0]
    for i in range(n_classes)
]

client_indices = [[] for _ in range(n_clients)]

for k in range(n_classes):

    np.random.shuffle(label_indices[k])

    proportions = np.random.dirichlet([alpha] * n_clients)

    proportions = (
        np.cumsum(proportions) * len(label_indices[k])
    ).astype(int)[:-1]

    split = np.split(label_indices[k], proportions)

    for i in range(n_clients):
        client_indices[i].extend(split[i])

return [
    DataLoader(
        Subset(dataset, idx),
        batch_size=16,
        shuffle=True
    )
    for idx in client_indices
]
```

def create_model():

```
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

for p in model.parameters():
    p.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2)
)

model = ModuleValidator.fix(model)

for m in model.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False

return model
```

def train_client(global_model, loader, norm, lr, mu):

```
model = copy.deepcopy(global_model).to(DEVICE)

global_params = [p.detach().clone() for p in model.parameters()]

optimizer = optim.Adam(model.fc.parameters(), lr=lr)

privacy_engine = PrivacyEngine()

model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=norm
)

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

model.train()

for imgs, labels in loader:

    optimizer.zero_grad()

    outputs = model(imgs.to(DEVICE))

    base_loss = criterion(outputs, labels.to(DEVICE))

    prox = sum(
        (p - g).norm(2)
        for p, g in zip(model.parameters(), global_params)
    )

    loss = base_loss + (mu / 2) * prox

    loss.backward()

    optimizer.step()

return model._module.state_dict(), privacy_engine.get_epsilon(delta=TARGET_DELTA)
```

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_PATH,"train"), transform=transform)

test_ds = datasets.ImageFolder(os.path.join(DATA_PATH,"test"), transform=transform)

client_loaders = dirichlet_partition(train_ds, N_CLIENTS, DIRICHLET_ALPHA)

test_loader = DataLoader(test_ds, batch_size=20)

global_model = create_model()

norm = 1.2

for r in range(ROUNDS):

```
lr = BASE_LR * (0.97 ** r)

results = [
    train_client(global_model, loader, norm, lr, MU)
    for loader in client_loaders
]

new_state = copy.deepcopy(results[0][0])

for k in new_state.keys():

    new_state[k] = torch.stack([
        res[0][k].float() for res in results
    ]).mean(0)

global_model.load_state_dict(new_state)

global_model.eval()

y_true, y_pred = [], []

with torch.no_grad():

    for imgs, labels in test_loader:

        out = global_model(imgs.to(DEVICE))

        y_true.extend(labels.numpy())

        y_pred.extend(out.argmax(1).cpu().numpy())

acc = np.mean(np.array(y_true) == np.array(y_pred))

print(f"Round {r+1} | Accuracy: {acc:.4f}")
```
