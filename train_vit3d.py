
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vit3d import ViT3D, MaskedViT3D
from dataset_ct import CTScanDataset
from tqdm import tqdm

# Config
data_path = "./MosMedData"
input_shape = (64, 128, 128)
batch_size = 1
epochs = 10
lr = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_id = 0  # or 0, 2, etc.
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(device)

# Dataset & Dataloader
dataset = CTScanDataset(data_path, target_shape=input_shape)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
# model = ViT3D(input_shape=input_shape).to(device)
model = MaskedViT3D(input_shape=input_shape).to(device)


# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# # Training Loop
# model.train()
# for epoch in range(epochs):
#     total_loss = 0
#     correct = 0
#     total = 0
#     for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#         x, y = x.to(device), y.to(device)

#         print(x.shape, y.shape, y)

#         logits = model(x)
#         loss = criterion(logits, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = torch.argmax(logits, dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     acc = correct / total
#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")


loss_list = []
acc_list = []


for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)
        logits = model(x, mask=None)  # 如果有mask就加上
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total

    # 记录
    loss_list.append(epoch_loss)
    acc_list.append(epoch_acc)

    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")
