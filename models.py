import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob, is_first_block=False):
        super(ResidualBlock, self).__init__()

        # Calcul du padding manuel pour le cas stride > 1
        self.manual_padding = (kernel_size - 1) // 2
        
        layers = []
        
        # Partie 1 : Avant la première convolution (inchangée)
        if not is_first_block:
            layers.append(nn.BatchNorm1d(in_channels))
            layers.append(nn.ReLU())
        
        # --- Partie 2 : Première Convolution ---
        # CORRECTION ICI : Choix dynamique du padding
        # Si stride=1, 'same' garantit la conservation de taille.
        # Si stride=2, 'same' est interdit, donc on utilise le manuel.
        padding_val = 'same' if stride == 1 else self.manual_padding
        
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding_val))
        
        # Partie 3 : Entre les deux convolutions (inchangée)
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # --- Partie 4 : Deuxième Convolution ---
        # Ici le stride est toujours 1, donc 'same' est parfait
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, 
                                stride=1, padding='same'))
        
        self.block = nn.Sequential(*layers)

        # --- Shortcut (inchangé) ---
        self.downsample = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.block(x) + self.downsample(x)


class SOTAModel(nn.Module):

    def __init__(self, filters, kernel_size, num_classes=12):
        super(SOTAModel, self).__init__()
        # Paramètres de base du papier
        dropout_prob = 0.2

        # --- Bloc Initial (Stem) ---
        self.initial = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size, padding='same'),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        # --- La Boucle des 16 Blocs ---
        self.blocks = nn.ModuleList()
        current_filters = filters
        
        for i in range(16):
            is_first_block = False
            if i == 0 :
                is_first_block = True
            # 1. Gestion du Subsampling (Un bloc sur deux)
            # Le papier dit "Every alternate... subsamples".
            # Disons que les blocs pairs (0, 2, 4...) réduisent la taille.
            stride = 2 if (i % 2 == 0) else 1
            
            # 2. Gestion des Filtres (Doubler tous les 4 blocs)
            # Astuce mathématique : i // 4 augmente de 1 tous les 4 pas (0,0,0,0, 1,1,1,1...)
            # Formule du papier : 32 * (2^k)
            k = i // 4
            out_filters = 32 * (2 ** k)
            
            # Création du bloc
            block = ResidualBlock(
                in_channels=current_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=stride,
                dropout_prob=dropout_prob, 
                is_first_block=is_first_block
            )
            
            self.blocks.append(block)
            
            # Mise à jour pour le prochain tour : la sortie devient l'entrée du suivant
            current_filters = out_filters
        
        # --- Couche Finale ---
        self.final_bn = nn.BatchNorm1d(current_filters)
        self.final_relu = nn.ReLU()
        # Le papier mentionne "Dense" (Linear) à la fin.
        # Pour connecter Conv1d -> Linear, on fait souvent une moyenne globale (Global Average Pooling)
        self.dense = nn.Linear(current_filters, num_classes)
        # Note : Softmax est inclus dans CrossEntropyLoss de PyTorch, pas besoin ici.

    def forward(self, x):
        x = self.initial(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Global Average Pooling : on moyenne sur toute la longueur temporelle
        # x shape: (Batch, Channels, Length) -> (Batch, Channels)
        x = x.mean(dim=2) 
        
        x = self.dense(x)
        return x


if __name__ == "__main__":
    # --- SANITY CHECK ---
    print("Running Sanity Check...")
    
    # 1. Define dummy input parameters
    # Batch size = 2 (just to check if batch processing works)
    # Channels = 1 (Single-lead ECG)
    # Length = 2000 (Random length, just needs to be long enough)
    N, C, L = 2, 1, 2000
    
    # 2. Create random dummy data
    # torch.randn generates data from a normal distribution
    x = torch.randn(N, C, L)
    print(f"Input shape: {x.shape}")
    
    # 3. Instantiate the model
    # We use the defaults: filters=32, kernel_size=16
    model = SOTAModel(filters=32, kernel_size=16, num_classes=12)
    
    # 4. Forward pass (The moment of truth!)
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        
        # 5. Verification
        # We expect Output to be (Batch_Size, Num_Classes) -> (2, 12)
        expected_shape = (N, 12)
        if y.shape == expected_shape:
            print("✅ Success! The output shape is correct.")
        else:
            print(f"❌ Error! Expected {expected_shape}, but got {y.shape}")
            
    except Exception as e:
        print("❌ Crash! Something went wrong inside the model.")
        print(e)