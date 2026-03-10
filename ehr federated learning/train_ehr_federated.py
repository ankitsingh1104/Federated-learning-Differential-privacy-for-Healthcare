import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve
import copy

# ==============================

# DATA LOADING



def load_dataset(path="data/diabetic_data.csv"):
df = pd.read_csv(path).replace("?", np.nan)

```
features = [
    "race","gender","age","time_in_hospital","num_lab_procedures",
    "num_procedures","num_medications","number_outpatient",
    "number_emergency","number_inpatient","number_diagnoses","readmitted"
]

df = df[features].dropna()

df["visits_intensity"] = df["number_inpatient"] / (df["time_in_hospital"] + 1)
df["medication_load"] = df["num_medications"] / (df["number_diagnoses"] + 1)

y = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0).values
X = df.drop(columns=["readmitted"])

for col in ["race","gender","age"]:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

return X.values, y
```

# ==============================

# MODEL



class FocalLoss(nn.Module):

```
def __init__(self, alpha=0.8, gamma=2):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.ce = nn.CrossEntropyLoss(reduction="none")

def forward(self, logits, targets):

    ce = self.ce(logits, targets)
    pt = torch.exp(-ce)

    return (self.alpha * (1 - pt) ** self.gamma * ce).mean()
```

class MedicalModel(nn.Module):

```
def __init__(self, input_dim):
    super().__init__()

    self.base = nn.Sequential(
        nn.Linear(input_dim,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU()
    )

    self.head = nn.Linear(64,2)

def forward(self,x):
    return self.head(self.base(x))
```

# ==============================

# FEDERATED ROUND



def train_round(model, X, y, indices):

```
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = FocalLoss()

y_local = y[indices]

pos = indices[y_local == 1]
neg = indices[y_local == 0]

s = min(len(pos),len(neg))

if s < 2:
    return model.state_dict()

balanced = np.concatenate([
    np.random.choice(pos,s),
    np.random.choice(neg,s)
])

loader = DataLoader(
    TensorDataset(
        torch.Tensor(X[balanced]),
        torch.LongTensor(y[balanced])
    ),
    batch_size=32,
    shuffle=True
)

privacy_engine = PrivacyEngine()

model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=0.4,
    max_grad_norm=1.0
)

for data,target in loader:

    optimizer.zero_grad()

    loss = criterion(model(data)/2.0,target)

    loss.backward()

    optimizer.step()

return model._module.state_dict()
```

# ==============================

# FEDERATED TRAINING



def federated_training(X,y,num_clients=5,rounds=20):

```
input_dim = X.shape[1]

global_model = MedicalModel(input_dim)

client_heads = [copy.deepcopy(global_model.head.state_dict()) for _ in range(num_clients)]

client_indices = np.array_split(np.arange(len(X)), num_clients)

client_weights = [len(i) for i in client_indices]

for r in range(rounds):

    updates = []

    for i in range(num_clients):

        local_model = MedicalModel(input_dim)

        local_model.base.load_state_dict(global_model.base.state_dict())
        local_model.head.load_state_dict(client_heads[i])

        state = train_round(local_model,X,y,client_indices[i])

        client_heads[i] = {k.replace("head.",""):v for k,v in state.items() if k.startswith("head")}

        updates.append({k.replace("base.",""):v for k,v in state.items() if k.startswith("base")})

    new_base = copy.deepcopy(global_model.base.state_dict())

    for k in new_base:

        new_base[k] = sum(
            updates[i][k] * (client_weights[i]/sum(client_weights))
            for i in range(num_clients)
        )

    global_model.base.load_state_dict(new_base)

return global_model,client_heads
```

if **name** == "**main**":

```
X,y = load_dataset()

scaler = StandardScaler()
X = scaler.fit_transform(X)

global_model,heads = federated_training(X,y)

print("Training Complete")
```
#                              ,/
#                            ,'/
#                          ,' / 
#                        ,'  /_____,
#                       .'____    ,'    
#                             /  ,'
#                             / ,'
#                             /,'
#                             /'

