import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_lth_progress(history_theta, history_f1, save_dir="plots/"):
    """Génère le graphique de performance et le sauvegarde en PNG."""
    # Création du dossier si inexistant
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lth_performance_plot.png")
    
    clear_output(wait=True)
    plt.figure(figsize=(10, 6))
    
    # Courbe de votre modèle
    plt.plot(history_theta, history_f1, 'r-.', label='LTH-ECG (Trained & pruned model)')
    
    # Benchmark du papier : F1 = 0.836 [cite: 36, 137]
    plt.axhline(y=0.836, color='black', linestyle='--', label='Benchmark Papier (0.836)')
    
    # Seuil de tolérance de 1% (0.836 * 0.99 ≈ 0.827) [cite: 15, 108]
    plt.axhline(y=0.827, color='gray', linestyle=':', label='Tolérance 1% (0.827)')
    
    plt.xlabel('Factor de réduction des paramètres ($\\theta$)') 
    plt.ylabel('Test mean F1-score')
    plt.title('Reproduction LTH-ECG : Performance vs Compression')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # --- MODIFICATION ICI ---

    plt.ylim(0.50, 0.95) # Zoom sur la zone d'intérêt [cite: 143-153]
    plt.xlim(0, max(max(history_theta), 175)) # Va jusqu'à 175 comme le papier [cite: 159]

    # ------------------------

    plt.legend(loc='lower left') # Legend en bas à gauche pour ne pas cacher la courbe
    
    # Sauvegarde physique du fichier
    plt.savefig(save_path)
    plt.show()
    print(f"Graphique mis à jour et sauvegardé dans : {save_path}")


def plot_layerwise_remaining_params(model, mask, save_dir="plots/"): 
    # Création du dossier si inexistant
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "layerwise_remaining_params_plot.png")

    # 1. Récupération des données brutes (désordonnées ou ordre incertain)
    raw_counts = mask.layerwise_remaining_params()

    # 2. Réordonnancement strict basé sur la structure du modèle
    # On utilise model.prunable_layer_names pour garantir l'ordre séquentiel (Input -> Output)
    ordered_layer_names = model.prunable_layer_names
    
    # On extrait les valeurs dans l'ordre
    y_values = []
    x_indices = []
    
    for i, name in enumerate(ordered_layer_names):
        if name in raw_counts:
            y_values.append(raw_counts[name])
            x_indices.append(i + 1) # Figure 2 commence à Layer 1, pas 0 [cite: 171]

    # 3. Création du graphique style Figure 2
    plt.figure(figsize=(10, 5))
    
    # Trace la ligne rouge comme dans l'article ("LTH-ECG (Our)")
    plt.plot(x_indices, y_values, color='red', marker='.', label='LTH-ECG (Our)')
    
    # Configuration des axes pour matcher l'article
    plt.xlabel('Layer Number') 
    plt.ylabel("Remaining parameters, eta'") 
    plt.title("Reproduction Figure 2 : Remaining parameters per layer")
    
    # L'article va jusqu'à la couche 34
    plt.xlim(0, 35)
    plt.xticks(np.arange(1, 35, step=3)) # Ticks tous les 3 (1, 4, 7...) comme sur l'image source
    
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Sauvegarde physique du fichier
    plt.savefig(save_path)
    plt.show()
    print(f"Graphique mis à jour et sauvegardé dans : {save_path}")