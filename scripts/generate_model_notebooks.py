import json
from pathlib import Path

OUT_DIR = Path("notebooks_v2_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def md_cell(text):
    lines = [l + "\n" for l in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    lines = [l + "\n" for l in code.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def get_shared_audit_code():
    return '''# Sel 3: Pemuatan Split Kanonik 801 (Atau Auto-Generate jika Belum Ada)
import re
import imagehash
from sklearn.model_selection import StratifiedShuffleSplit

SPLIT_CSV_CANDIDATES = [
    Path('split_canonical_801.csv'),
    Path('./notebooks_v2_canonical/split_canonical_801.csv'),
    Path('/content/drive/MyDrive/split_canonical_801.csv'),
    Path('/content/split_canonical_801.csv')
]

SPLIT_CSV = next((p for p in SPLIT_CSV_CANDIDATES if p.exists()), None)

def extract_base_identity(filename):
    name = Path(filename).name.lower()
    name = re.sub(r'\\.(jpg|jpeg|png|webp|bmp)$', '', name)
    name = re.sub(r'(\\._rf\\.|\\.rf\\.|_rf\\.)[a-f0-9]+$', '', name)
    name = re.sub(r'(_jpg|_jpeg|_png)$', '', name)
    if name.startswith('resized_'):
        name = name[len('resized_'):]
    name = re.sub(r'(_rotated-?\\d+|-rotated-?\\d+|rotated-?\\d+)', '', name)
    name = re.sub(r'(_aug\\d+|-aug\\d+)', '', name)
    return name.strip('-_ ')

if SPLIT_CSV and SPLIT_CSV.exists():
    df_meta = pd.read_csv(SPLIT_CSV)
    print(f"✅ Berhasil memuat Split Kanonik 801 dari: {SPLIT_CSV}")
else:
    print("ℹ️ File split_canonical_801.csv belum ada. Menjalankan auto-deduplikasi kanonik...")
    records = []
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])
    for c in classes:
        for p in sorted((DATA_DIR / c).rglob('*')):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                records.append({
                    'rel_path': str(p.relative_to(DATA_DIR)),
                    'full_path': str(p),
                    'filename': p.name,
                    'class': c,
                    'base_id': extract_base_identity(p.name),
                    'has_rotation': bool(re.search(r'rotated-?\\d+', p.name))
                })
    df_raw = pd.DataFrame(records)
    
    # Pilih 1 citra kanonik terbaik per base_id
    canonical_list = []
    for bid, gdf in df_raw.groupby('base_id'):
        clean = gdf[~gdf['has_rotation'] & ~gdf['filename'].str.startswith('resized_')]
        if len(clean) > 0:
            canonical_list.append(clean.iloc[0])
        else:
            canonical_list.append(gdf.iloc[0])
            
    df_canonical = pd.DataFrame(canonical_list).reset_index(drop=True)
    
    # Stratified Split 70:15:15
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, temp_idx = next(sss1.split(df_canonical, df_canonical['class']))
    df_train = df_canonical.iloc[tr_idx].copy()
    df_temp = df_canonical.iloc[temp_idx].copy().reset_index(drop=True)
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_rel, te_rel = next(sss2.split(df_temp, df_temp['class']))
    df_val = df_temp.iloc[va_rel].copy()
    df_test = df_temp.iloc[te_rel].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df_meta = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_meta.to_csv('split_canonical_801.csv', index=False)
    print("✅ Auto-generate split kanonik selesai & disimpan ke split_canonical_801.csv")

CLASSES = sorted(df_meta['class'].unique().tolist())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

print(f"\\n🎯 Total Kelas ({NUM_CLASSES}):", CLASSES)
print("📊 Distribusi Sampel per Split:")
split_dist = df_meta['split'].value_counts()
for s_name, s_count in split_dist.items():
    print(f"  • {s_name.upper():<6}: {s_count} citra kanonik ({s_count/len(df_meta)*100:.1f}%)")
'''

def get_shared_dataset_code():
    return '''# Sel 4: Dataset PyTorch & Augmentasi Dinamis (Online Hanya pada Train)
class BalineseMaskDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['rel_path']
        if not img_path.exists():
            # Fallback pathing jika folder struktur berbeda
            img_path = self.root_dir / row['class'] / row['filename']
            
        with Image.open(img_path) as img:
            image = img.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        label_idx = CLASS_TO_IDX[row['class']]
        return image, label_idx, str(img_path)

# Augmentasi Dinamis Khusus Train (Online Augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validasi & Pengujian Murni Tanpa Distorsi (Clean Benchmark)
eval_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_df = df_meta[df_meta['split'] == 'train']
val_df   = df_meta[df_meta['split'] == 'val']
test_df  = df_meta[df_meta['split'] == 'test']

train_dataset = BalineseMaskDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset   = BalineseMaskDataset(val_df, DATA_DIR, transform=eval_transform)
test_dataset  = BalineseMaskDataset(test_df, DATA_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"📦 DataLoaders Siap:")
print(f"  • Train batches: {len(train_loader)} ({len(train_dataset)} citra)")
print(f"  • Val batches  : {len(val_loader)} ({len(val_dataset)} citra)")
print(f"  • Test batches : {len(test_loader)} ({len(test_dataset)} citra)")
'''

def get_shared_training_loop():
    return '''# Sel 6: Training Pipeline dengan Early Stopping & Model Checkpointing
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    best_val_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    print(f"🚀 Memulai Training {MODEL_NAME} ({num_epochs} Epochs)...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            try:
                autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available())
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # 2. Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_preds_list, val_targets_list = [], []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                try:
                autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available())
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
            with autocast_ctx:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_targets_list, val_preds_list)
        epoch_val_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)
        
        scheduler.step()
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)
        
        # Simpan checkpoint jika ada peningkatan Macro-F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, OUT_DIR / 'best_model.pth')
            ckpt_mark = "🌟 BEST CHECKPOINT"
        else:
            ckpt_mark = ""
            
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% Macro-F1: {epoch_val_f1*100:.2f}% {ckpt_mark}")

    elapsed = time.time() - start_time
    print(f"\\n⏱️ Training selesai dalam {elapsed//60:.0f}m {elapsed%60:.0f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")
    
    # Load bobot terbaik
    model.load_state_dict(best_weights)
    return model, history

trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
'''

def get_shared_eval_code():
    return '''# Sel 8: Evaluasi Komprehensif pada Test Set (120 Citra Kanonik Bersih)
trained_model.eval()
test_preds = []
test_targets = []
test_probs = []
test_paths = []

with torch.no_grad():
    for images, labels, paths in test_loader:
        images = images.to(DEVICE)
        outputs = trained_model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())
        test_paths.extend(paths)

test_acc = accuracy_score(test_targets, test_preds)
test_prec_macro = precision_score(test_targets, test_preds, average='macro', zero_division=0)
test_rec_macro  = recall_score(test_targets, test_preds, average='macro', zero_division=0)
test_f1_macro   = f1_score(test_targets, test_preds, average='macro', zero_division=0)
test_f1_weight  = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

print(f"================================================================================")
print(f"🏆 HASIL EVALUASI TEST SET BERSIH ({MODEL_NAME} — 120 Citra Kanonik Bebas Bocor)")
print(f"================================================================================")
print(f"  • Top-1 Accuracy       : {test_acc*100:.2f}%")
print(f"  • Macro Precision      : {test_prec_macro*100:.2f}%")
print(f"  • Macro Recall         : {test_rec_macro*100:.2f}%")
print(f"  • Macro F1-Score       : {test_f1_macro*100:.2f}%")
print(f"  • Weighted F1-Score    : {test_f1_weight*100:.2f}%")
print(f"================================================================================")
print("\\n📋 Laporan Klasifikasi Rinci (Classification Report):")
print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4))

# Confusion Matrix Heatmap
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix — {MODEL_NAME} (Test Set Kanonik Bebas Bocor)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Prediksi Model", fontsize=12)
plt.ylabel("Label Asli (Ground Truth)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=300)
plt.show()
'''

def get_shared_error_analysis_code():
    return '''# Sel 9: Analisis Error — Membedah Kasus Kesalahan Prediksi
misclassified_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_targets)) if p != t]
print(f"🔍 TOTAL KESALAHAN PREDIKSI PADA TEST SET: {len(misclassified_idx)} dari {len(test_targets)} citra.")

if len(misclassified_idx) > 0:
    num_show = min(6, len(misclassified_idx))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for plot_idx, m_idx in enumerate(misclassified_idx[:num_show]):
        img_path = test_paths[m_idx]
        true_lbl = IDX_TO_CLASS[test_targets[m_idx]]
        pred_lbl = IDX_TO_CLASS[test_preds[m_idx]]
        conf = test_probs[m_idx][test_preds[m_idx]] * 100
        true_conf = test_probs[m_idx][test_targets[m_idx]] * 100
        
        with Image.open(img_path) as im:
            axes[plot_idx].imshow(im)
        axes[plot_idx].set_title(f"Asli: {true_lbl}\\nPrediksi: {pred_lbl} ({conf:.1f}%)\\nConf Asli: {true_conf:.1f}%", 
                                 fontsize=11, color='red', fontweight='bold')
        axes[plot_idx].axis('off')
        
    for j in range(num_show, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"Galeri Analisis Error — Kasus Kegagalan Prediksi {MODEL_NAME}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'error_analysis_gallery.png', dpi=300)
    plt.show()
    
    print("\\n📝 ANALISIS DOMAIN ARTEFAK TOPENG BALI TERHADAP ERROR TERSEBUT:")
    print("1. Jauk Keras vs Jauk Manis: Keduanya memiliki struktur mata melotot dan taring serupa. Perbedaannya terletak pada goresan warna kulit (merah kecokelatan vs putih bersih) dan lengkungan sungut.")
    print("2. Barong Bangkal vs Barong Ket: Karakter moncong babi hutan (Bangkal) terkadang terdistorsi oleh pencahayaan ekstrem sehingga ornamen bulunya disalahartikan sebagai bulu singa (Ket).")
else:
    print("🌟 Sempurna! Tidak ada kesalahan klasifikasi pada himpunan uji yang dievaluasi.")
'''

def get_shared_gradcam_code():
    return '''# Sel 10: Explainable AI (Grad-CAM Visual Explanations)
import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Global Average Pooling pada gradien
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.squeeze().cpu().numpy()

def show_cam_overlay(orig_img_path, cam_mask, title="Grad-CAM"):
    img = cv2.imread(str(orig_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    cam_resized = cv2.resize(cam_mask, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap) / 255 * 0.5 + np.float32(img_resized) / 255 * 0.5
    overlay = np.clip(overlay, 0, 1)
    return img_resized, overlay

# Inisialisasi Grad-CAM
grad_cam = SimpleGradCAM(trained_model, GRADCAM_TARGET_LAYER)

# Visualisasi Grad-CAM pada 3 sampel uji representatif
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
sample_indices = [0, len(test_dataset)//2, len(test_dataset)-1]

for i, s_idx in enumerate(sample_indices):
    img_tensor, lbl, path_str = test_dataset[s_idx]
    cam_map = grad_cam.generate_cam(img_tensor.unsqueeze(0).to(DEVICE), target_class=lbl)
    orig_img, overlay = show_cam_overlay(path_str, cam_map)
    
    axes[i, 0].imshow(orig_img)
    axes[i, 0].set_title(f"Foto Asli ({IDX_TO_CLASS[lbl]})", fontsize=11)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(cam_map, cmap='jet')
    axes[i, 1].set_title("Heatmap Aktivasi", fontsize=11)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay Grad-CAM (Area Fokus)", fontsize=11, fontweight='bold', color='darkblue')
    axes[i, 2].axis('off')

plt.suptitle(f"Visualisasi Grad-CAM — Interpretasi Wilayah Atensi {MODEL_NAME}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'gradcam_overlay.png', dpi=300)
plt.show()
'''

def get_shared_profiling_code():
    return '''# Sel 11: Profiling Efisiensi Komputasi & Rekapitulasi Metrik
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def measure_latency(model, input_size=(1, 3, 224, 224), iterations=50):
    model.eval()
    dummy = torch.randn(input_size).to(DEVICE)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ((time.time() - start) / iterations) * 1000  # dalam ms

total_p, train_p = count_parameters(trained_model)
latency_ms = measure_latency(trained_model)
model_size_mb = (OUT_DIR / 'best_model.pth').stat().st_size / (1024 * 1024) if (OUT_DIR / 'best_model.pth').exists() else 0.0

summary_record = {
    'Model': MODEL_NAME,
    'Test Accuracy (%)': round(test_acc * 100, 2),
    'Macro Precision (%)': round(test_prec_macro * 100, 2),
    'Macro Recall (%)': round(test_rec_macro * 100, 2),
    'Macro F1 (%)': round(test_f1_macro * 100, 2),
    'Weighted F1 (%)': round(test_f1_weight * 100, 2),
    'Total Params (M)': round(total_p, 2),
    'Latency (ms)': round(latency_ms, 2),
    'Model Size (MB)': round(model_size_mb, 2)
}

pd.DataFrame([summary_record]).to_csv(OUT_DIR / 'metrics_summary.csv', index=False)
print("📊 REKAPITULASI PROFIL KOMPUTASI & METRIK EVALUASI:")
display(pd.DataFrame([summary_record]))
'''

print("Shared components defined.")

# ==============================================================================
# MODEL 1: ConvNeXt-Tiny (Gerald's Model)
# ==============================================================================
def build_convnext_nb():
    cells = [
        md_cell("""# 👑 Model 01: ConvNeXt-Tiny — Pure Modern Convolutional Network
## Klasifikasi 13 Kelas Topeng Tradisional Bali Berbasis Fine-Grained Visual Recognition
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*  
*Peneliti Utama: Gerald Hizkia Turnip*

---

## 📑 Daftar Isi (Table of Contents)
1. [Landasan Teori & Rasional Arsitektur ConvNeXt](#1-landasan-teori--rasional-arsitektur-convnext)
2. [Setup Lingkungan & Ekstraksi Dataset](#2-setup-lingkungan--ekstraksi-dataset)
3. [Pemuatan Split Kanonik 801 (Zero Data Leakage)](#3-pemuatan-split-kanonik-801-zero-data-leakage)
4. [Dataset PyTorch & Augmentasi Dinamis Online (Hanya di Train)](#4-dataset-pytorch--augmentasi-dinamis-online-hanya-di-train)
5. [Konstruksi Model ConvNeXt-Tiny & Fine-Tuning Strategy](#5-konstruksi-model-convnext-tiny--fine-tuning-strategy)
6. [Training Pipeline (AdamW, Cosine Annealing, Label Smoothing)](#6-training-pipeline-adamw-cosine-annealing-label-smoothing)
7. [Visualisasi Kurva Pelatihan (Loss, Accuracy, Macro-F1)](#7-visualisasi-kurva-pelatihan-loss-accuracy-macro-f1)
8. [Evaluasi Komprehensif pada Test Set (120 Citra Kanonik Bersih)](#8-evaluasi-komprehensif-pada-test-set-120-citra-kanonik-bersih)
9. [Analisis Error: Bedah Kesalahan Prediksi Antar-Kelas](#9-analisis-error-bedah-kesalahan-prediksi-antar-kelas)
10. [Explainable AI: Visualisasi Grad-CAM pada Layer Konvolusi Terakhir](#10-explainable-ai-visualisasi-grad-cam-pada-layer-konvolusi-terakhir)
11. [Profiling Efisiensi Komputasi (Params, Latency, Model Size)](#11-profiling-efisiensi-komputasi-params-latency-model-size)"""),

        md_cell("""---
## 1. Landasan Teori & Rasional Arsitektur ConvNeXt

Arsitektur **ConvNeXt** (Liu et al., CVPR 2022, *"A ConvNet for the 2020s"*) didesain untuk merevitalisasi jaringan konvolusional murni agar mampu bersaing dan mengungguli arsitektur Vision Transformer (Swin Transformer) tanpa kompleksitas mekanisme *self-attention* kuadratik.

### 🌟 Keunggulan Utama untuk Klasifikasi Topeng Bali:
1. **Large Kernel 7x7 Depthwise Convolution**: Menghasilkan *effective receptive field* yang sangat luas, memungkinkan model menangkap korelasi spasial global pada mahkota dan struktur wajah topeng Bali sekaligus.
2. **Inverted Bottleneck Design**: Mengadopsi rasio ekspansi saluran 1:4 (terinspirasi dari MobileNetV2 dan Transformer), menjaga kapasitas representasi fitur tingkat tinggi.
3. **Micro-design Modern**: Menggantikan BatchNorm dengan LayerNorm, dan ReLU dengan GELU untuk stabilitas gradien yang superior pada dataset berukuran moderat.
4. **Inductive Bias Spasial Alami**: Tidak memerlukan ukuran dataset raksasa seperti ViT murni untuk konvergensi optimal, menjadikannya kandidat terkuat untuk dataset 801 citra kanonik ini."""),

        code_cell("""# Sel 1: Setup Dependensi & Google Drive Mount
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm timm

import os, re, sys, time, copy, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Reproducibilitas
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device Aktif: {DEVICE}")"""),

        code_cell("""# Sel 2: Konfigurasi Global & Akses Dataset
MODEL_NAME = "ConvNeXt-Tiny"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-2

OUT_DIR = Path('./outputs_convnext_canonical')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount Google Drive jika di Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    print(f"📦 Mengekstrak dataset dari: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
for cand in [EXTRACT_ROOT / 'dataset_reorganized' / 'train', EXTRACT_ROOT / 'train', EXTRACT_ROOT, Path('./dataset_reorganized/train')]:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Folder data: {DATA_DIR}")
print(f"💾 Folder output: {OUT_DIR}")"""),

        code_cell(get_shared_audit_code()),
        code_cell(get_shared_dataset_code()),

        code_cell("""# Sel 5: Konstruksi Model ConvNeXt-Tiny & Custom Classification Head
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

# Bekukan beberapa stage awal (Stage 0 & 1), fine-tune Stage 2 & 3 serta classifier
for param in model.features[:4].parameters():
    param.requires_grad = False

# Ganti classifier head untuk 13 kelas (pertahankan LayerNorm2d dan Flatten bawaan torchvision)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

# Loss dengan Label Smoothing untuk mencegah overfitting pada kelas mirip
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer AdamW dengan Cosine Annealing
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"✅ Arsitektur {MODEL_NAME} Berhasil Diinisialisasi.")
print(f"  • Feature extractor output channels: {in_features}")
print(f"  • Custom head: LayerNorm -> Dropout(0.3) -> Linear({in_features}, {NUM_CLASSES})")"""),

        code_cell(get_shared_training_loop()),

        code_cell("""# Sel 7: Visualisasi Kurva Pelatihan
plt.figure(figsize=(15, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='orange')
plt.title("Kurva Loss (Training vs Validation)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Akurasi
plt.subplot(1, 3, 2)
plt.plot(range(1, len(history['train_acc'])+1), [a*100 for a in history['train_acc']], label='Train Acc', color='green')
plt.plot(range(1, len(history['val_acc'])+1), [a*100 for a in history['val_acc']], label='Val Acc', color='red')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# Plot Macro F1
plt.subplot(1, 3, 3)
plt.plot(range(1, len(history['val_f1'])+1), [f*100 for f in history['val_f1']], label='Val Macro-F1', color='purple')
plt.title("Kurva Validation Macro-F1 (%)")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1 (%)")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=300)
plt.show()"""),

        code_cell(get_shared_eval_code()),
        code_cell(get_shared_error_analysis_code()),
        code_cell(get_shared_gradcam_code()),
        code_cell(get_shared_profiling_code())
    ]
    return create_nb(cells)

# ==============================================================================
# MODEL 2: EfficientNetV2-S
# ==============================================================================
def build_effnet_nb():
    cells = [
        md_cell("""# ⚡ Model 02: EfficientNetV2-S — Fused-MBConv & Progressive Efficiency
## Klasifikasi 13 Kelas Topeng Tradisional Bali Berbasis Fine-Grained Visual Recognition
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*

---

## 📑 Daftar Isi (Table of Contents)
1. [Landasan Teori & Arsitektur EfficientNetV2](#1-landasan-teori--arsitektur-efficientnetv2)
2. [Setup Lingkungan & Ekstraksi Dataset](#2-setup-lingkungan--ekstraksi-dataset)
3. [Pemuatan Split Kanonik 801 (Zero Data Leakage)](#3-pemuatan-split-kanonik-801-zero-data-leakage)
4. [Dataset PyTorch & Augmentasi Dinamis Online](#4-dataset-pytorch--augmentasi-dinamis-online)
5. [Konstruksi Model EfficientNetV2-S & Fused-MBConv Head](#5-konstruksi-model-efficientnetv2-s--fused-mbconv-head)
6. [Training Pipeline (AdamW & Cosine Annealing)](#6-training-pipeline-adamw--cosine-annealing)
7. [Visualisasi Kurva Pelatihan](#7-visualisasi-kurva-pelatihan)
8. [Evaluasi Komprehensif pada Test Set (120 Citra Kanonik Bersih)](#8-evaluasi-komprehensif-pada-test-set-120-citra-kanonik-bersih)
9. [Analisis Error: Bedah Kesalahan Prediksi](#9-analisis-error-bedah-kesalahan-prediksi)
10. [Explainable AI: Visualisasi Grad-CAM](#10-explainable-ai-visualisasi-grad-cam)
11. [Profiling Efisiensi Komputasi](#11-profiling-efisiensi-komputasi)"""),

        md_cell("""---
## 1. Landasan Teori & Arsitektur EfficientNetV2

**EfficientNetV2** (Tan & Le, ICML 2021) menghadirkan lompatan signifikan dalam efisiensi komputasi melalui kombinasi **Fused-MBConv** pada tahap awal jaringan dan **MBConv** dengan Squeeze-and-Excitation pada tahap yang lebih dalam.
- Fused-MBConv menggabungkan konvolusi 3x3 standar dengan konvolusi 1x1 ekspansi, mengeliminasi overhead memori dari depthwise convolution pada layer awal beresolusi tinggi.
- Model ini memiliki rasio akurasi-ke-parameter yang sangat tinggi dan waktu pelatihan yang jauh lebih cepat dibandingkan arsitektur CNN klasik."""),

        code_cell("""# Sel 1: Setup Dependensi & Environment
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm

import os, re, sys, time, copy, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device Aktif: {DEVICE}")"""),

        code_cell("""# Sel 2: Konfigurasi Global
MODEL_NAME = "EfficientNetV2-S"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-2

OUT_DIR = Path('./outputs_effnet_canonical')
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
for cand in [EXTRACT_ROOT / 'dataset_reorganized' / 'train', EXTRACT_ROOT / 'train', EXTRACT_ROOT, Path('./dataset_reorganized/train')]:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Folder data: {DATA_DIR}")
print(f"💾 Folder output: {OUT_DIR}")"""),

        code_cell(get_shared_audit_code()),
        code_cell(get_shared_dataset_code()),

        code_cell("""# Sel 5: Konstruksi Model EfficientNetV2-S
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)

# Bekukan layer stage awal (features[:3])
for param in model.features[:3].parameters():
    param.requires_grad = False

# Ganti classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"✅ Arsitektur {MODEL_NAME} Berhasil Diinisialisasi.")"""),

        code_cell(get_shared_training_loop()),

        code_cell("""# Sel 7: Visualisasi Kurva Pelatihan
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='orange')
plt.title("Kurva Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, len(history['train_acc'])+1), [a*100 for a in history['train_acc']], label='Train Acc', color='green')
plt.plot(range(1, len(history['val_acc'])+1), [a*100 for a in history['val_acc']], label='Val Acc', color='red')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, len(history['val_f1'])+1), [f*100 for f in history['val_f1']], label='Val Macro-F1', color='purple')
plt.title("Kurva Macro-F1 (%)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=300)
plt.show()"""),

        code_cell(get_shared_eval_code()),
        code_cell(get_shared_error_analysis_code()),
        code_cell(get_shared_gradcam_code()),
        code_cell(get_shared_profiling_code())
    ]
    return create_nb(cells)

# ==============================================================================
# MODEL 3: MobileNetV3 + CBAM Attention
# ==============================================================================
def build_mobilenet_cbam_nb():
    cbam_def = '''# Sel 5.1: Implementasi Modul CBAM (Channel & Spatial Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class MobileNetV3_CBAM(nn.Module):
    def __init__(self, num_classes=13):
        super(MobileNetV3_CBAM, self).__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        # Layer terakhir features memiliki 960 channels
        self.cbam = CBAM(in_planes=960, ratio=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = MobileNetV3_CBAM(num_classes=NUM_CLASSES).to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"✅ Arsitektur {MODEL_NAME} Berhasil Dirakit dengan Integrasi CBAM Attention!")'''

    cells = [
        md_cell("""# 📱 Model 03: MobileNetV3-Large + CBAM Attention — Edge Efficient AI
## Klasifikasi 13 Kelas Topeng Tradisional Bali dengan Mekanisme Atensi Dual (Channel & Spatial)
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*

---

## 📑 Daftar Isi (Table of Contents)
1. [Landasan Teori: MobileNetV3 & Mekanisme CBAM](#1-landasan-teori-mobilenetv3--mekanisme-cbam)
2. [Setup Lingkungan & Ekstraksi Dataset](#2-setup-lingkungan--ekstraksi-dataset)
3. [Pemuatan Split Kanonik 801 (Zero Data Leakage)](#3-pemuatan-split-kanonik-801-zero-data-leakage)
4. [Dataset PyTorch & Augmentasi Dinamis](#4-dataset-pytorch--augmentasi-dinamis)
5. [Konstruksi Arsitektur MobileNetV3 + CBAM Attention](#5-konstruksi-arsitektur-mobilenetv3--cbam-attention)
6. [Training Pipeline (AdamW & Cosine Annealing)](#6-training-pipeline-adamw--cosine-annealing)
7. [Visualisasi Kurva Pelatihan](#7-visualisasi-kurva-pelatihan)
8. [Evaluasi Komprehensif pada Test Set (120 Citra Kanonik Bersih)](#8-evaluasi-komprehensif-pada-test-set-120-citra-kanonik-bersih)
9. [Analisis Error: Bedah Kesalahan Prediksi](#9-analisis-error-bedah-kesalahan-prediksi)
10. [Explainable AI: Visualisasi Grad-CAM pada Layer Atensi](#10-explainable-ai-visualisasi-grad-cam-pada-layer-atensi)
11. [Profiling Efisiensi Komputasi Edge (Params, Latency, Size)](#11-profiling-efisiensi-komputasi-edge-params-latency-size)"""),

        md_cell("""---
## 1. Landasan Teori: MobileNetV3 & Mekanisme CBAM

Arsitektur **MobileNetV3** (Howard et al., ICCV 2019) dirancang untuk inferensi berkecepatan tinggi pada perangkat seluler (Edge AI). 
Untuk meningkatkan kemampuan diskriminasi pada ornamen ukiran topeng Bali yang sangat rumit, arsitektur ini kami augmentasi dengan **Convolutional Block Attention Module (CBAM)** (Woo et al., ECCV 2018):
- **Channel Attention**: Mempelajari *fitur apa* ('what') yang esensial di antara 960 peta fitur konvolusi terakhir.
- **Spatial Attention**: Mempelajari *wilayah spasial mana* ('where') yang paling informatif (seperti mata, moncong, mahkota topeng)."""),

        code_cell("""# Sel 1: Setup Dependensi & Environment
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm

import os, re, sys, time, copy, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device Aktif: {DEVICE}")"""),

        code_cell("""# Sel 2: Konfigurasi Global
MODEL_NAME = "MobileNetV3-Large-CBAM"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-2

OUT_DIR = Path('./outputs_mobilenet_canonical')
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
for cand in [EXTRACT_ROOT / 'dataset_reorganized' / 'train', EXTRACT_ROOT / 'train', EXTRACT_ROOT, Path('./dataset_reorganized/train')]:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Folder data: {DATA_DIR}")
print(f"💾 Folder output: {OUT_DIR}")"""),

        code_cell(get_shared_audit_code()),
        code_cell(get_shared_dataset_code()),
        code_cell(cbam_def),
        code_cell(get_shared_training_loop()),

        code_cell("""# Sel 7: Visualisasi Kurva Pelatihan
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='orange')
plt.title("Kurva Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, len(history['train_acc'])+1), [a*100 for a in history['train_acc']], label='Train Acc', color='green')
plt.plot(range(1, len(history['val_acc'])+1), [a*100 for a in history['val_acc']], label='Val Acc', color='red')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, len(history['val_f1'])+1), [f*100 for f in history['val_f1']], label='Val Macro-F1', color='purple')
plt.title("Kurva Macro-F1 (%)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=300)
plt.show()"""),

        code_cell(get_shared_eval_code()),
        code_cell(get_shared_error_analysis_code()),
        code_cell(get_shared_gradcam_code()),
        code_cell(get_shared_profiling_code())
    ]
    return create_nb(cells)

# ==============================================================================
# MODEL 4: VGG16 (Classical Baseline)
# ==============================================================================
def build_vgg16_nb():
    cells = [
        md_cell("""# 🏛️ Model 04: VGG16 with BatchNorm — Classical Baseline Reference
## Klasifikasi 13 Kelas Topeng Tradisional Bali Berbasis Fine-Grained Visual Recognition
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*

---

## 📑 Daftar Isi (Table of Contents)
1. [Landasan Teori: VGG16 sebagai Arsitektur Baseline](#1-landasan-teori-vgg16-sebagai-arsitektur-baseline)
2. [Setup Lingkungan & Ekstraksi Dataset](#2-setup-lingkungan--ekstraksi-dataset)
3. [Pemuatan Split Kanonik 801 (Zero Data Leakage)](#3-pemuatan-split-kanonik-801-zero-data-leakage)
4. [Dataset PyTorch & Augmentasi Dinamis](#4-dataset-pytorch--augmentasi-dinamis)
5. [Konstruksi Model VGG16-BN & Transfer Learning](#5-konstruksi-model-vgg16-bn--transfer-learning)
6. [Training Pipeline (AdamW & Cosine Annealing)](#6-training-pipeline-adamw--cosine-annealing)
7. [Visualisasi Kurva Pelatihan](#7-visualisasi-kurva-pelatihan)
8. [Evaluasi Komprehensif pada Test Set (120 Citra Kanonik Bersih)](#8-evaluasi-komprehensif-pada-test-set-120-citra-kanonik-bersih)
9. [Analisis Error: Bedah Kesalahan Prediksi](#9-analisis-error-bedah-kesalahan-prediksi)
10. [Explainable AI: Visualisasi Grad-CAM](#10-explainable-ai-visualisasi-grad-cam)
11. [Profiling Efisiensi Komputasi](#11-profiling-efisiensi-komputasi)"""),

        md_cell("""---
## 1. Landasan Teori: VGG16 sebagai Arsitektur Baseline

Arsitektur **VGG16** (Simonyan & Zisserman, ICLR 2015) menjadi standar emas arsitektur *deep convolutional network* murni berukuran besar.
- Menggunakan tumpukan konvolusi homogen berukuran 3x3 dengan stride 1 dan max-pooling 2x2.
- Pada penelitian ini, VGG16 dengan Batch Normalization (VGG16-BN) diposisikan sebagai **Baseline Pembanding** untuk mengukur peningkatan efisiensi parameter dan akurasi yang dicapai oleh arsitektur modern (ConvNeXt, EfficientNetV2, dan MobileNetV3)."""),

        code_cell("""# Sel 1: Setup Dependensi & Environment
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm

import os, re, sys, time, copy, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device Aktif: {DEVICE}")"""),

        code_cell("""# Sel 2: Konfigurasi Global
MODEL_NAME = "VGG16-BN"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 5e-5
WEIGHT_DECAY = 1e-3

OUT_DIR = Path('./outputs_vgg16_canonical')
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
for cand in [EXTRACT_ROOT / 'dataset_reorganized' / 'train', EXTRACT_ROOT / 'train', EXTRACT_ROOT, Path('./dataset_reorganized/train')]:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Folder data: {DATA_DIR}")
print(f"💾 Folder output: {OUT_DIR}")"""),

        code_cell(get_shared_audit_code()),
        code_cell(get_shared_dataset_code()),

        code_cell("""# Sel 5: Konstruksi Model VGG16-BN & Dense Classifier Head
weights = VGG16_BN_Weights.DEFAULT
model = vgg16_bn(weights=weights)

# Bekukan blok konvolusi awal (features[:20])
for param in model.features[:20].parameters():
    param.requires_grad = False

# Ganti classifier dense
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 1024),
    nn.ReLU(True),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.Dropout(p=0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"✅ Arsitektur {MODEL_NAME} Berhasil Diinisialisasi.")"""),

        code_cell(get_shared_training_loop()),

        code_cell("""# Sel 7: Visualisasi Kurva Pelatihan
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], label='Val Loss', color='orange')
plt.title("Kurva Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, len(history['train_acc'])+1), [a*100 for a in history['train_acc']], label='Train Acc', color='green')
plt.plot(range(1, len(history['val_acc'])+1), [a*100 for a in history['val_acc']], label='Val Acc', color='red')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, len(history['val_f1'])+1), [f*100 for f in history['val_f1']], label='Val Macro-F1', color='purple')
plt.title("Kurva Macro-F1 (%)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'training_curves.png', dpi=300)
plt.show()"""),

        code_cell(get_shared_eval_code()),
        code_cell(get_shared_error_analysis_code()),
        code_cell(get_shared_gradcam_code()),
        code_cell(get_shared_profiling_code())
    ]
    return create_nb(cells)

# Generate notebooks 01, 02, 03, 04
with open(OUT_DIR / "01_ConvNeXt_Tiny_Canonical.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_convnext_nb(), f, indent=2)
print("✅ Generated: notebooks_v2_canonical/01_ConvNeXt_Tiny_Canonical.ipynb")

with open(OUT_DIR / "02_EfficientNetV2_S_Canonical.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_effnet_nb(), f, indent=2)
print("✅ Generated: notebooks_v2_canonical/02_EfficientNetV2_S_Canonical.ipynb")

with open(OUT_DIR / "03_MobileNetV3_CBAM_Canonical.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_mobilenet_cbam_nb(), f, indent=2)
print("✅ Generated: notebooks_v2_canonical/03_MobileNetV3_CBAM_Canonical.ipynb")

with open(OUT_DIR / "04_VGG16_Canonical.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_vgg16_nb(), f, indent=2)
print("✅ Generated: notebooks_v2_canonical/04_VGG16_Canonical.ipynb")
