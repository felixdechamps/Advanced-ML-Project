import torch
import torch.nn as nn


class ZeroPad1d(nn.Module):
    def __init__(self):
        super(ZeroPad1d, self).__init__()

    def forward(self, x):
        # Colle des zéros le long de la dimension des channels (dim 1)
        # Cela double la profondeur : [N, C, L] -> [N, 2C, L]
        return torch.cat([x, torch.zeros_like(x)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob, is_first_block=False):
        super(ResidualBlock, self).__init__()

        # Calcul du padding manuel pour le cas stride > 1
        self.manual_padding = (kernel_size - 1) // 2
        
        # --- MAIN BLOCK (Chemin principal) ---
        layers = []
        
        if not is_first_block:
            layers.append(nn.BatchNorm1d(in_channels))
            layers.append(nn.ReLU())
        
        # 1ère Conv : Choix dynamique du padding
        padding_val = 'same' if stride == 1 else self.manual_padding
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding_val))
        
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # 2ème Conv : Toujours stride=1 donc padding='same'
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, 
                                stride=1, padding='same'))
        
        self.block = nn.Sequential(*layers)

        # --- SHORTCUT (Raccourci) ---
        # On construit le raccourci étape par étape
        shortcut_layers = []
        
        # 1. Si on réduit la longueur (L), on ajoute MaxPool
        if stride > 1:
            shortcut_layers.append(nn.MaxPool1d(kernel_size=stride, stride=stride))
            
        # 2. Si (et seulement si) on change la profondeur (C), on ajoute ZeroPad
        if in_channels != out_channels:
            shortcut_layers.append(ZeroPad1d())

        # Si la liste contient quelque chose, on en fait un Sequential
        # Sinon, c'est une liste vide qui agit comme une identité
        self.downsample = nn.Sequential(*shortcut_layers)

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
        #x = x.mean(dim=2) 
        x = x.transpose(1, 2)

        x = self.dense(x)
        return x


if __name__ == "__main__":
    import torch
    
    # --- SANITY CHECK ---
    print("🚀 Running Sanity Check...")
    
    # 1. Define dummy input parameters
    N, C, L = 2, 1, 2000
    x = torch.randn(N, C, L)
    print(f"Input shape: {x.shape}")
    
    # 2. Instantiate the model
    # Note: We use the defaults from the paper (filters=32, kernel=16)
    model = SOTAModel(kernel_size=16, filters=32, num_classes=12)
    
    # 3. Try to use torchinfo for a beautiful summary
    try:
        from torchinfo import summary
        print("\n--- Model Summary ---")
        summary(model, input_size=(N, C, L), 
                col_names=["input_size", "output_size", "num_params", "kernel_size"],
                depth=3) # depth=3 allows to see inside the Sequential blocks
        print("---------------------\n")
    except ImportError:
        print("⚠️ torchinfo not found. Run 'pip install torchinfo' to get a detailed summary.")

    # 4. Forward pass (The moment of truth!)
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        
        # 5. Verification
        expected_shape = (N, 7, 12)
        if y.shape == expected_shape:
            print("✅ Success! The output shape is correct.")
        else:
            print(f"❌ Error! Expected {expected_shape}, but got {y.shape}")
            
    except Exception as e:
        print("❌ Crash! Something went wrong inside the model.")
        print(e)