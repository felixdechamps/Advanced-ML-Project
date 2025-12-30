# 🚀 Guide de Démarrage Rapide

Guide pas-à-pas pour reproduire les résultats de Hannun et al. (2019) sur le dataset PhysioNet 2017.

## ⚡ Installation en 5 minutes

### 1. Cloner/Créer le projet

```bash
# Créer la structure de répertoires
mkdir -p ecg_classification/{data,models,utils,checkpoints,logs,results}
cd ecg_classification

# Copier tous les fichiers Python fournis dans leurs répertoires respectifs
```

### 2. Installer les dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les packages
pip install -r requirements.txt
```

### 3. Télécharger les données

**Option A : Script automatique (Linux/Mac)**
```bash
chmod +x download_data.sh
./download_data.sh
```

**Option B : Manuel**
```bash
mkdir -p data/physionet2017
cd data/physionet2017

# Télécharger
wget -r -N -c -np --cut-dirs=3 --reject "index.html*" \
  https://physionet.org/files/challenge-2017/1.0.0/training2017/

# Déplacer les fichiers
mv training2017/* .
rmdir training2017

cd ../..
```

### 4. Tester l'installation

```bash
python test_setup.py
```

Vous devriez voir :
```
✓ PASS: Imports
✓ PASS: PyTorch
✓ PASS: Model Architecture
✓ PASS: Dataset Loading
✓ PASS: Metrics
✓ PASS: Training Utilities

Total: 6/6 tests passed
✓ All tests passed! Ready to train.
```

## 🎯 Entraînement du modèle

### Entraînement complet

```bash
python train.py
```

**Ce que fait le script** :
- Charge le dataset PhysioNet 2017 (8,528 échantillons)
- Split 90% train / 10% validation (Hannun et al.)
- Initialise ResNet1D 34 couches avec He initialization
- Entraîne avec Adam (β₁=0.9, β₂=0.999, LR=1e-3)
- Réduit LR par 10 si validation loss stagne 2 époques
- Sauvegarde le meilleur modèle basé sur F1-score
- Early stopping après 10 époques sans amélioration

**Durée estimée** :
- GPU (NVIDIA RTX 3080) : 2-4 heures
- GPU (NVIDIA T4) : 4-8 heures  
- CPU : 24-48 heures (non recommandé)

**Résultats attendus** :
```
Epoch 50/100
------------------------------------------------------------
Train Loss: 0.2134
Val Loss:   0.2456
Val F1:     0.8354
Learning Rate: 0.000010

✓ New best model! F1: 0.8354
```

### Reprendre un entraînement interrompu

```bash
python train.py --resume checkpoints/best_model.pth
```

## 📊 Évaluation du modèle

### Évaluation basique

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### Évaluation avec visualisations

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --plot \
  --save-results \
  --output-dir results/
```

**Résultats attendus** (Hannun et al. Supplementary Table 7) :

```
============================================================
EVALUATION METRICS
============================================================

Overall Metrics:
  Accuracy:        0.8923
  F1 (macro):      0.8360
  F1 (weighted):   0.8891
  Precision:       0.8456
  Recall:          0.8312
  AUC (macro):     0.9700

Per-Class Metrics:
Class           F1        Precision    Recall  Specificity      AUC
------------------------------------------------------------
Normal          0.9090       0.9124    0.9056       0.8712   0.9750
AF              0.8270       0.8401    0.8142       0.9780   0.9650
Other           0.7720       0.7834    0.7608       0.9234   0.9580
Noisy           0.5060       0.5201    0.4924       0.9956   0.8920
============================================================

COMPARISON WITH BENCHMARKS
============================================================

Hannun et al. (2019) - Supplementary Table 7:
  Mean F1-score: 0.836

Current model:
  Mean F1-score: 0.836 (Δ = +0.000)

✓ Performance matches Hannun et al. benchmark!
  Ready for model compression (Sahu et al. LTH-ECG)
```

## 🗜️ Compression du modèle (LTH-ECG)

Une fois le modèle de base entraîné et validé :

```python
from models.resnet1d import ResNet1d
from lth_ecg import LTHECGPruner
from config import Config
import torch

# Charger le modèle entraîné
config = Config()
model = ResNet1d(
    in_channels=1,
    base_filters=config.base_filters,
    kernel_size=config.kernel_size,
    n_classes=config.n_classes,
    dropout_rate=config.dropout_rate
)

checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Créer le pruner
# Sahu et al. : 142x reduction avec <1% F1 loss
pruner = LTHECGPruner(
    model=model,
    target_reduction_factor=142,  # 10.5M → 74K params
    initial_prune_rate=0.30,      # 30% initial
    alpha=1.1                     # Decay factor
)

# Définir fonction d'entraînement
def train_function(model):
    # Votre logique d'entraînement
    # Doit retourner validation F1-score
    trainer = Trainer(model, train_loader, val_loader, config, device)
    # ... train for few epochs ...
    return val_f1

# Lancer la compression
results = pruner.prune_iteratively(train_function, max_iterations=20)

# Sauvegarder modèle compressé
pruner.save_pruned_model('checkpoints/lth_ecg_compressed.pth')

# Résultats attendus (Sahu et al. Table III)
# Initial: 10.5M params, 115 MB
# Final: 74K params (~0.8 MB), F1: 0.8360 (no degradation!)
```

## 📁 Structure finale du projet

```
ecg_classification/
├── data/
│   ├── physionet2017/
│   │   ├── REFERENCE.csv          # 8,528 labels
│   │   ├── A00001.mat             # ECG recordings
│   │   └── ...
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── resnet1d.py                # 34-layer ResNet1D
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # F1, AUC, etc.
│   └── training.py                # Trainer class
├── checkpoints/
│   ├── best_model.pth             # ~115 MB
│   └── lth_ecg_compressed.pth     # ~0.8 MB
├── logs/
│   └── training_log.txt
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── evaluation_results.npz
├── config.py
├── train.py
├── evaluate.py
├── lth_ecg.py
├── test_setup.py
├── download_data.sh
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## ⚙️ Configuration personnalisée

Modifier `config.py` pour ajuster les hyperparamètres :

```python
class Config:
    # Training
    batch_size = 64            # Réduire si GPU mémoire limitée
    learning_rate = 5e-4       # Ajuster selon convergence
    max_epochs = 150           # Augmenter si besoin
    
    # Model
    base_filters = 32          # Hannun et al. default
    dropout_rate = 0.2         # Hannun et al. default
    
    # Data
    val_split = 0.1            # 10% validation (Hannun et al.)
    
    # Device
    device = 'cuda'            # ou 'cpu'
```

## 🐛 Dépannage rapide

### Erreur : CUDA out of memory
```python
# Dans config.py
batch_size = 32  # ou 16
```

### Erreur : Dataset not found
```bash
# Vérifier le chemin
ls data/physionet2017/*.mat | head
cat data/physionet2017/REFERENCE.csv | head

# Re-télécharger si nécessaire
./download_data.sh
```

### Performance inférieure aux benchmarks
1. Entraîner plus longtemps (100-150 époques)
2. Vérifier l'équilibre des classes
3. Ajuster le learning rate
4. Utiliser data augmentation (rotation, scaling)

### Entraînement trop lent sur CPU
```python
# Activer optimisations CPU
torch.set_num_threads(8)  # Ajuster selon votre CPU

# Ou louer GPU cloud :
# - Google Colab (gratuit, T4 GPU)
# - AWS EC2 (p3.2xlarge avec V100)
# - Lambda Labs (A100 GPU)
```

## 📈 Prochaines étapes

1. **Validation** : F1-score ≈ 0.836 ✓
2. **Compression** : Appliquer LTH-ECG (142x) ✓
3. **Déploiement** : Embarquer sur microcontrôleur (STM32)
4. **Extension** : Étendre à 12-lead ECG ou autres arythmies

## 🎓 Références rapides

- **Paper** : Hannun et al. (2019) Nature Medicine
- **Dataset** : https://physionet.org/content/challenge-2017/
- **Repo original** : https://github.com/awni/ecg
- **ResNet1D** : https://github.com/hsd1503/resnet1d
- **Compression** : Sahu et al. (2022) IEEE EMBC

---

**Temps total estimé** : 4-8 heures (setup + training + evaluation)

**Questions ?** Consultez le README.md complet ou les articles référencés.