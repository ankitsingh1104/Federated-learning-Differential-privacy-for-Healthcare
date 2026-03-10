import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models, datasets
from sklearn.metrics import accuracy_score
import kagglehub

# ==============================

# CONFIGURATION

# ==============================

ALPHA = 0.5
NUM_CLIENTS = 5
NUM_ROUNDS = 10
TOTAL_SAMPLES = 1000
BASE_SIGMA = 0.00005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================

# DATASET LOADING

# ==============================

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

train_path = os.path.join(path, "Training")

transform = transforms.Compose([
transforms.Resize((64,64)),
transforms.ToTensor(),
transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(train_path, transform=transform)

num_classes = len(dataset.classes)

indices = np.random.choice(len(dataset), TOTAL_SAMPLES, replace=False)

labels = np.array(dataset.targets)[indices]

# ==============================

# NON-IID SPLIT

# ==============================

def dirichlet_split(indices, labels, n_clients, alpha):

```
n_classes = len(np.unique(labels))

client_idx = [[] for _ in range(n_clients)]

for k in range(n_classes):

    idx_k = np.where(labels == k)[0]

    np.random.shuffle(idx_k)

    proportions = np.random.dirichlet([alpha]*n_clients)

    proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]

    splits = np.split(idx_k, proportions)

    for i in range(n_clients):
        client_idx[i].extend(indices[splits[i]])

return client_idx
```

hospital_indices = dirichlet_split(indices, labels, NUM_CLIENTS, ALPHA)

# ==============================

# FEDERATED CLIENT

# ==============================

class MRIClient(fl.client.NumPyClient):

```
def __init__(self, indices):

    self.model = models.resnet18(num_classes=num_classes).to(DEVICE)

    self.loader = DataLoader(
        Subset(dataset, indices),
        batch_size=32,
        shuffle=True
    )

    self.criterion = nn.CrossEntropyLoss()


def get_parameters(self, config):

    return [v.cpu().numpy() for v in self.model.state_dict().values()]


def set_parameters(self, parameters):

    params_dict = zip(self.model.state_dict().keys(), parameters)

    state_dict = {k: torch.tensor(v) for k,v in params_dict}

    self.model.load_state_dict(state_dict, strict=True)


def fit(self, parameters, config):

    self.set_parameters(parameters)

    round_num = config.get("curr_round",1)

    lr = 1e-3 if round_num <= 5 else 1e-5

    optimizer = optim.Adam(self.model.parameters(), lr=lr)

    self.model.train()

    for x,y in self.loader:

        x,y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        loss = self.criterion(self.model(x),y)

        loss.backward()

        optimizer.step()


    params = self.get_parameters({})


    if round_num > 5:

        sens = BASE_SIGMA/(1+np.sqrt(round_num))

        params = [
            p + np.random.normal(
                0,
                np.mean(np.abs(p))*sens,
                p.shape
            )
            for p in params
        ]


    return params, len(self.loader.dataset), {}


def evaluate(self, parameters, config):

    self.set_parameters(parameters)

    self.model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for x,y in self.loader:

            out = self.model(x.to(DEVICE))

            preds.extend(torch.argmax(out,1).cpu().numpy())

            labels.extend(y.numpy())

    acc = accuracy_score(labels,preds)

    return 0.0, len(self.loader.dataset), {"accuracy":acc}
```

# ==============================

# FEDERATED TRAINING

# ==============================

strategy = fl.server.strategy.FedProx(

```
proximal_mu=0.1,

evaluate_metrics_aggregation_fn=lambda m:{
    "accuracy":sum([x["accuracy"] for _,x in m])/len(m)
},

on_fit_config_fn=lambda r:{"curr_round":r}
```

)

history = fl.simulation.start_simulation(

```
client_fn=lambda cid:
    MRIClient(hospital_indices[int(cid)]).to_client(),

num_clients=NUM_CLIENTS,

config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),

strategy=strategy
```

)

print("Federated MRI training complete.")
#                              ,/
#                            ,'/
#                          ,' / 
#                        ,'  /_____,
#                       .'____    ,'    
#                             /  ,'
#                             / ,'
#                             /,'
#                             /'
