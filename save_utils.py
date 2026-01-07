import subprocess
import os
import torch

def push_checkpoint_to_git(filepaths, branch_name="felix-v3", commit_msg="Auto-save: checkpoint"):
    """
    Ajoute, commit et push un fichier spécifique vers le repo distant.
    """
    try:
        for filepath in filepaths: 

            if not os.path.exists(filepath):
                print(f"Erreur Git : Le fichier {filepath} n'existe pas.")
                return

            print(f"Git : Sauvegarde de {filepath} vers la branche '{branch_name}'...")
            
            # 1. Git Add
            subprocess.run(["git", "add", filepath], check=True)
            
        # 2. Git Commit (ignore l'erreur si rien n'a changé)
        subprocess.run(["git", "commit", "-m", commit_msg], check=False, stdout=subprocess.DEVNULL)
            
        # 3. Git Push
        # Note : Cela suppose que tes crédentiels (SSH ou Token) sont déjà configurés
        subprocess.run(["git", "push", "origin", branch_name], check=True)
        
        print("Git : Push réussi !")
        
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du push Git : {e}")

def save_checkpoint(state, filename="lth_checkpoint.pth"):
    """Sauvegarde l'état complet de l'expérience."""
    print(f"--> Sauvegarde du checkpoint : {filename}")
    torch.save(state, filename)

def load_checkpoint(filename="lth_checkpoint.pth"):
    """Charge l'état pour reprendre l'expérience."""
    print(f"--> Chargement du checkpoint : {filename}")
    return torch.load(filename, weights_only=False)
