import os
import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLIENTS = 3
ROUNDS = 30
ALPHA = 1.0     
MU = 0.001       
LR = 1e-3
EPS_DELTA = 1e-5
NOISE = 0.8    
WEIGHTS = torch.tensor([1.0, 1.8]).to(DEVICE) 
PATH = "/kaggle/input/datasets/khoongweihao/covid19-xray-dataset-train-test-sets/xray_dataset_covid19"

def dirichlet_split(ds, n_clients, alpha):
    n_classes = len(ds.classes)
    indices = [np.where(np.array(ds.targets) == i)[0] for i in range(n_classes)]
    client_idx = [[] for _ in range(n_clients)]
    
    for k in range(n_classes):
        np.random.shuffle(indices[k])
        dist = np.random.dirichlet([alpha] * n_clients)
        splits = (np.cumsum(dist) * len(indices[k])).astype(int)[:-1]
        for i, subset in enumerate(np.split(indices[k], splits)):
            client_idx[i].extend(subset)
            
    return [DataLoader(Subset(ds, idx), batch_size=16, shuffle=True) for idx in client_idx]

def get_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters(): 
        p.requires_grad = False
    
    # Custom head for 2-class xray classification
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )
    
    m = ModuleValidator.fix(m)
    for module in m.modules():
        if isinstance(module, nn.ReLU): 
            module.inplace = False
    return m

def tta_inference(model, img):
    img = img.to(DEVICE)
    model.to(DEVICE).eval()
    
    # Simple TTA: Original, Flip, Rotate, Brightness
    augments = [
        img,
        torch.flip(img, dims=[-1]), 
        transforms.RandomRotation(10)(img),
        transforms.ColorJitter(brightness=0.1)(img)
    ]
    
    with torch.no_grad():
        preds = [torch.softmax(model(a.unsqueeze(0)), dim=1) for a in augments]
    return torch.mean(torch.stack(preds), dim=0)

def local_train(global_m, loader, norm, lr):
    global_m.train()
    local_m = copy.deepcopy(global_m).to(DEVICE)
    ref_params = [p.detach().clone() for p in local_m.parameters()]
    
    opt = optim.Adam(local_m.fc.parameters(), lr=lr)
    engine = PrivacyEngine()
    
    local_m, opt, loader = engine.make_private(
        module=local_m, optimizer=opt, data_loader=loader,
        noise_multiplier=NOISE, max_grad_norm=norm
    )
    
    loss_fn = nn.CrossEntropyLoss(weight=WEIGHTS)
    
    for x, y in loader:
        opt.zero_grad()
        out = local_m(x.to(DEVICE))
        # Weighted loss + FedProx penalty
        loss = loss_fn(out, y.to(DEVICE))
        prox = sum((p - rp).norm(2) for p, rp in zip(local_m.parameters(), ref_params))
        (loss + (MU/2)*prox).backward()
        opt.step()
        
    return {k: v.cpu() for k, v in local_m._module.state_dict().items()}, engine.get_epsilon(delta=EPS_DELTA)

if __name__ == "__main__":
    img_tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(PATH, "train"), transform=img_tf)
    test_set = datasets.ImageFolder(os.path.join(PATH, "test"), transform=img_tf)
    
    loaders = dirichlet_split(train_set, N_CLIENTS, ALPHA)
    test_loader = DataLoader(test_set, batch_size=1)

    net = get_model()
    current_norm, top_acc = 1.2, 0.0

    print("Starting Federated Training...")
    for r in range(ROUNDS):
        current_lr = LR * (0.99 ** r)
        client_updates = [local_train(net, l, current_norm, current_lr) for l in loaders]
        
        # Simple Averaging
        avg_weights = client_updates[0][0]
        for k in avg_weights.keys():
            avg_weights[k] = torch.stack([u[0][k].float() for u in client_updates]).mean(0).type(avg_weights[k].dtype)
        net.load_state_dict(avg_weights)

        # Eval
        hits = 0
        for x, y in test_loader:
            if tta_inference(net, x[0]).argmax(1).item() == y.item(): 
                hits += 1
        
        acc = 100 * hits / len(test_set)
        print(f"Round {r+1} | TTA Acc: {acc:.2f}% | Eps: {client_updates[0][1]:.2f}")
        
        if acc > top_acc:
            top_acc = acc
            torch.save(net.state_dict(), "best_model.pth")

        current_norm *= 0.98
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Best: {top_acc}%")
