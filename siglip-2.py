import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import json
from io import BytesIO
from tqdm import tqdm
import wandb
import os
import open_clip
from sklearn.metrics import average_precision_score
import torch.backends.cudnn as cudnn
import cv2

# ✅ Initialize Weights & Biases for experiment tracking
wandb.login(key="")
wandb.init(project="siglip-verb-classification", name="siglip2-finetune-unfreeze-all")

# ✅ Set device and enable fast training
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚀 Step 1: Load Pretrained SigLIP2 Model
print("Loading SigLIP2 ViT-SO400M-16-SigLIP2-512...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-SO400M-16-SigLIP2-512", pretrained="webli"
)
print("✅ SigLIP loaded.")

# ✅ Wrapper around SigLIP visual encoder for multi-label classification
def get_siglip_model(base_model, num_classes=10, dropout_rate=0.3):
    class SigLIPVerbClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base_model.visual
            self.fc = nn.Sequential(
                nn.Linear(1152, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.fc(features)

    return SigLIPVerbClassifier()

# ✅ Instantiate and load Stage 1 weights
siglip_model = get_siglip_model(model).to(device)
weights_path = "weights/siglip_epoch_9.pth"
siglip_model.load_state_dict(torch.load(weights_path, map_location=device))
print(f"✅ Loaded weights from {weights_path}")

# 🚀 Step 2: Unfreeze all layers for fine-tuning
for param in siglip_model.backbone.parameters():
    param.requires_grad = True
print("✅ Unfroze all SigLIP layers.")

# ✅ Custom Dataset Loader for base64 image + CLAHE
class CholecT45Dataset(Dataset):
    def __init__(self, json_file, preprocess, augment=False):
        with open(json_file, 'r') as f:
            self.data = [item for item in json.load(f) if 'image' in item and 'verb_labels' in item]
        self.preprocess = preprocess
        self.augment = augment
        self.aug_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(0.4),
            transforms.ColorJitter(0.1, 0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            preprocess
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(BytesIO(base64.b64decode(item['image']))).convert("RGB")
        image_np = np.array(image)

        # CLAHE
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        image_pil = Image.fromarray(clahe_img)

        # Final preprocessing
        if self.augment:
            image_final = self.aug_pipeline(image_pil)
        else:
            image_final = self.preprocess(image_pil)

        labels = torch.tensor(item['verb_labels'], dtype=torch.float32)
        return image_final, labels

# 🚀 Step 3: Create DataLoaders
train_dataset = CholecT45Dataset("../../../../instrument_verb_train.json", preprocess_train, augment=True)
val_dataset = CholecT45Dataset("../../../../instrument_verb_val.json", preprocess_val)

data_config = {
    "batch_size": 8,
    "num_workers": 12,
    "pin_memory": True,
    "persistent_workers": True
}

train_loader = DataLoader(train_dataset, shuffle=True, **data_config)
val_loader = DataLoader(val_dataset, shuffle=False, **data_config)

# 🚀 Step 4: Loss, Optimizer, Scheduler
loss_weights = torch.tensor([0.38, 0.06, 0.07, 0.84, 1.21, 2.52, 1.30, 6.88, 17.07, 0.45], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
optimizer = optim.AdamW(siglip_model.parameters(), lr=2e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-5,
    total_steps=len(train_loader) * 15,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10,
    final_div_factor=100
)

# ✅ Create directory to save weights
os.makedirs("weights_finetune", exist_ok=True)

# 🚀 Step 5: Training Loop

def train_finetune(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    print("🚀 Starting fine-tuning...")
    best_val_mAP = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs.float(), labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            torch.cuda.empty_cache()

        scheduler.step()
        val_loss, val_mAP = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}, Val mAP={val_mAP:.4f}")

        # Save model
        torch.save(model.state_dict(), f"weights_finetune/finetune_epoch_{epoch+1}.pth")
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save(model.state_dict(), "weights_finetune/best_finetune_model.pth")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_mAP": val_mAP
        })

# 🚀 Step 6: Evaluation

def evaluate(model, loader, criterion):
    model.eval()
    all_targets, all_preds = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.float(), labels.float())
            running_loss += loss.item()
            all_targets.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    val_mAP = average_precision_score(
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        average="macro"
    )
    return running_loss / len(loader), val_mAP

# 🚀 Step 7: Inference on Test Set

def test_model(model, json_file):
    model.eval()
    test_dataset = CholecT45Dataset(json_file, preprocess_val)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_targets, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(labels.numpy())

    mAP = average_precision_score(
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        average="macro"
    )
    print(f"✅ Test mAP: {mAP:.4f}")

# ✅ Start training
train_finetune(siglip_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15)

# ✅ Evaluate on test set
test_model(siglip_model, "../../../../instrument_verb_test.json")
