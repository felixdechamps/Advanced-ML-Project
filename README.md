# ECG Classification - Reproduction de Hannun et al. (2019)

Ce projet reproduit les résultats de l'article **"Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network"** de Hannun et al. (2019) publié dans Nature Medicine, en utilisant le dataset PhysioNet/CinC Challenge 2017.

## 📚 Références

### Articles principaux
1. **Hannun et al. (2019)** - *Nature Medicine* - "Cardiologist-Level Arrhythmia Detection and Classification"
   - Architecture ResNet 34 couches pour classification ECG
   - Performance cardiologiste sur 12 classes de rythmes
   
2. **Sahu et al. (2022)** - *IEEE EMBC* - "LTH-ECG: Lottery Ticket Hypothesis-based Deep Learning Model Compression"
   - Compression du modèle de Hannun et al. par 142x
   - Performance maintenue avec <1% de perte

### Repositories GitHub
- **awni/ecg**: https://github.com/awni/ecg/tree/master (Hannun et al. - TensorFlow)
- **hsd1503/resnet1d**: https://github.com/hsd1503/resnet1d (Implémentation ResNet1D PyTorch)

### Dataset
- **PhysioNet/CinC Challenge 2017**: https://physionet.org/content/challenge-2017/1.0.0/
- 8,528 enregistrements ECG mono-dérivation
- 4 classes: Normal, AF (Fibrillation Auriculaire), Other, Noisy

## 🏗️ Architecture du projet

```
ecg_classification/
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Chargement dataset PhysioNet 2017
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   └── resnet1d.py             # Architecture ResNet1D (34 couches)
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # Métriques d'évaluation (F1, AUC, etc.)
│   └── training.py             # Utilitaires d'entraînement
├── config.py                   # Configuration (hyperparamètres)
├── train.py                    # Script d'entraînement principal
├── evaluate.py                 # Script d'évaluation
├── requirements.txt
└── README.md
```

## 📋 Prérequis

### Installation des dépendances

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm
```

Ou avec le fichier requirements.txt :

```bash
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## 📥 Téléchargement des données

### 1. Télécharger le dataset PhysioNet 2017

```bash
# Créer le répertoire des données
mkdir -p data/physionet2017
cd data/physionet2017

# Télécharger les fichiers
wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/training2017/

# Les données seront dans training2017/
```

### 2. Structure attendue des données

```
data/physionet2017/
├── REFERENCE.csv              # Labels (format: A00001,N)
├── A00001.mat                 # Fichier ECG 1
├── A00002.mat                 # Fichier ECG 2
└── ...
```

Le fichier `REFERENCE.csv` contient les labels au format :
```
A00001,N
A00002,A
A00003,O
A00004,~
```

Où :
- N = Normal sinus rhythm
- A = Atrial Fibrillation (AF)
- O = Other rhythms  
- ~ = Noisy

## 🚀 Utilisation

### 1. Entraînement du modèle

```bash
python train.py
```

**Paramètres de training (config.py)** basés sur Hannun et al. :
- Optimiseur : Adam (β₁=0.9, β₂=0.999)
- Learning rate initial : 1e-3
- Batch size : 128
- Dropout : 0.2
- Kernel size : 16
- Base filters : 32

L'entraînement suit la procédure de Hannun et al. :
1. Initialisation He des poids (He et al., 2015)
2. Réduction du learning rate par 10 si la loss de validation stagne pendant 2 époques
3. Sauvegarde du meilleur modèle basé sur le F1-score de validation
4. Early stopping si pas d'amélioration pendant 10 époques

### 2. Reprendre un entraînement

```bash
python train.py --resume checkpoints/best_model.pth
```

### 3. Évaluation du modèle

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --plot --save-results
```

Options :
- `--checkpoint` : Chemin vers le checkpoint du modèle (requis)
- `--plot` : Générer les visualisations (confusion matrix, ROC curves)
- `--save-results` : Sauvegarder les résultats dans un fichier .npz
- `--output-dir` : Répertoire de sortie (défaut: ./results)

## 📊 Métriques d'évaluation

### Métriques principales (Hannun et al.)

1. **F1-Score** : Moyenne harmonique de la précision et du recall
   - Calculé par classe
   - Moyenne macro (non pondérée)
   - Moyenne pondérée (par fréquence de classe)

2. **AUC** : Area Under ROC Curve
   - Stratégie one-vs-rest
   - Par classe et moyenne

3. **Sensitivity (Recall)** et **Specificity**
   - Par classe
   - Comparaison avec cardiologues

### Résultats attendus (Hannun et al. - Supplementary Table 7)

Sur le dataset PhysioNet 2017 :

| Classe | F1-Score |
|--------|----------|
| Normal | 0.909    |
| AF     | 0.827    |
| Other  | 0.772    |
| Noisy  | 0.506    |
| **Moyenne** | **0.836** |

## 🔍 Détails de l'architecture

### ResNet1D (34 couches)

Basée sur Hannun et al. Extended Data Figure 1 :

```
Input: (batch, 1, 9000) - ECG mono-dérivation 30s à 300Hz

Conv1D (kernel=16, stride=2) → BatchNorm → ReLU

16 Residual Blocks organisés en 4 groupes :
├── Blocks 1-4:   32 filters  (k=0, 32×2⁰)
├── Blocks 5-8:   64 filters  (k=1, 32×2¹)
├── Blocks 9-12:  128 filters (k=2, 32×2²)
└── Blocks 13-16: 256 filters (k=3, 32×2³)

Chaque Residual Block :
├── BatchNorm → ReLU → Dropout(0.2) → Conv1D(kernel=16)
├── BatchNorm → ReLU → Dropout(0.2) → Conv1D(kernel=16)
└── Shortcut connection (identity ou projection)

Downsampling : stride=2 tous les 2 blocks (blocks alternés)

Global Average Pooling → Linear(256, 4) → Softmax

Output: (batch, 4) - Probabilités des classes
```

**Nombre de paramètres** : ~10.5M (comme Hannun et al.)

### Justifications architecturales

1. **Pre-activation design** (He et al., 2016)
   ```python
   # Hannun et al. : "Before each convolutional layer we applied 
   # Batch Normalization and a rectified linear activation"
   out = self.bn1(x)
   out = self.relu1(out)
   out = self.conv1(out)
   ```

2. **Shortcut connections** (He et al., 2016)
   ```python
   # Hannun et al. : "employed shortcut connections in manner similar to 
   # Residual Network architecture"
   out = conv_block(x) + shortcut(x)
   ```

3. **Dropout** (Srivastava et al., 2014)
   ```python
   # Hannun et al. : "applied Dropout... with probability of 0.2"
   self.dropout = nn.Dropout(p=0.2)
   ```

4. **Filter progression**
   ```python
   # Hannun et al. : "32*2^k filters, where k starts at zero and 
   # incremented by one every fourth residual block"
   filters = base_filters * (2 ** k)
   ```

## 🔬 Preprocessing des données

### Signal ECG (dataset.py)

```python
# PhysioNet 2017 : Signaux de longueur variable (9-60s, moyenne ~30s)
# Hannun et al. : Segments de 30s

target_length = 9000  # 30s × 300Hz

# Si signal trop long : truncation
if len(signal) >= target_length:
    signal = signal[:target_length]
    
# Si signal trop court : zero-padding
else:
    signal = np.pad(signal, (0, target_length - len(signal)))

# Normalisation Z-score
signal = (signal - np.mean(signal)) / np.std(signal)
```

### Justification

- **Truncation** : Hannun et al. utilisent segments de 30s fixes
- **Zero-padding** : Standard pour longueurs variables (hsd1503/resnet1d)
- **Normalisation** : Améliore la stabilité d'entraînement (bien que BatchNorm soit utilisé)

## 📈 Résultats et comparaisons

### Comparaison avec Hannun et al.

Lors de l'évaluation, le script compare automatiquement :

```
COMPARISON WITH BENCHMARKS
============================================================

Hannun et al. (2019) - Supplementary Table 7:
  Mean F1-score: 0.836
  Normal: 0.909
  AF: 0.827
  Other: 0.772
  Noisy: 0.506

Current model:
  Mean F1-score: 0.XXX (Δ = ±0.XXX)
  ...
```

### Préparation pour Sahu et al. (LTH-ECG)

Une fois que le modèle de base atteint les performances de Hannun et al. (~0.836 F1-score), il peut être compressé avec la méthode LTH-ECG de Sahu et al. :

**Objectif de compression** :
- Réduction de paramètres : 142× (de 10.5M à ~74K paramètres)
- Perte de performance : <1% F1-score
- Taille mémoire : de 115 MB à ~0.8 MB

## 🐛 Debugging et troubleshooting

### Problème : CUDA out of memory

```python
# Réduire batch size dans config.py
batch_size = 64  # au lieu de 128
```

### Problème : Performance inférieure aux benchmarks

1. **Vérifier le preprocessing** :
   - Normalisation correcte des signaux
   - Longueur des segments (9000 samples)

2. **Augmenter le nombre d'époques** :
   ```python
   max_epochs = 150  # au lieu de 100
   ```

3. **Vérifier l'équilibre des classes** :
   - Le dataset est déséquilibré (60% Normal, 9% AF)
   - Considérer weighted sampling ou class weights

### Problème : Fichiers .mat non trouvés

```bash
# Vérifier la structure des données
ls data/physionet2017/*.mat | head
cat data/physionet2017/REFERENCE.csv | head
```

## 📚 Citations

Si vous utilisez ce code, veuillez citer :

```bibtex
@article{hannun2019cardiologist,
  title={Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network},
  author={Hannun, Awni Y and Rajpurkar, Pranav and Haghpanahi, Masoumeh and Tison, Geoffrey H and Bourn, Codie and Turakhia, Mintu P and Ng, Andrew Y},
  journal={Nature medicine},
  volume={25},
  number={1},
  pages={65--69},
  year={2019}
}

@inproceedings{sahu2022lth,
  title={LTH-ECG: Lottery Ticket Hypothesis-based Deep Learning Model Compression for Atrial Fibrillation Detection from Single Lead ECG On Wearable and Implantable Devices},
  author={Sahu, Ishan and Ukil, Arijit and Khandelwal, Sundeep and Pal, Arpan},
  booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={1655--1658},
  year={2022}
}

@inproceedings{clifford2017af,
  title={AF classification from a short single lead ECG recording: the PhysioNet/computing in cardiology challenge 2017},
  author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and Lehman, Li-wei H and Silva, Ikaro and Li, Qiao and Johnson, AE and Mark, Roger G},
  booktitle={2017 Computing in Cardiology (CinC)},
  pages={1--4},
  year={2017}
}
```

## 📞 Support

Pour toute question concernant :
- **Dataset** : https://physionet.org/content/challenge-2017/
- **Architecture** : Voir Extended Data Figure 1 dans Hannun et al. (2019)
- **Implementation** : https://github.com/awni/ecg et https://github.com/hsd1503/resnet1d

## 📝 License

Ce code est fourni à des fins de recherche et d'éducation. Veuillez respecter les licences des articles et datasets originaux.

## ✅ Checklist de reproduction

- [ ] Dataset PhysioNet 2017 téléchargé
- [ ] Environnement Python configuré (PyTorch, scikit-learn, etc.)
- [ ] Structure des fichiers correcte
- [ ] Entraînement lancé avec succès
- [ ] F1-score ≈ 0.836 (±0.01) atteint
- [ ] Visualisations générées (confusion matrix, ROC curves)
- [ ] Modèle sauvegardé pour compression future (Sahu et al.)

---

**Note** : Ce projet reproduit la méthodologie de Hannun et al. (2019) avec PyTorch. Pour la version TensorFlow originale, voir https://github.com/awni/ecg