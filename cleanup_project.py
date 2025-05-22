import os
import shutil
import glob

def cleanup_project():
    """Nettoie le projet en gardant seulement l'essentiel"""
    
    print("🧹 NETTOYAGE DU PROJET DE DÉTECTION")
    print("="*50)
    
    # === DOSSIERS À GARDER (ESSENTIELS) ===
    keep_folders = {
        'output_extended_30',     # Vos meilleurs modèles (28 classes)
        'output_fast',            # Modèles originaux (10 classes)
        'coco_evaluation',        # Données d'évaluation COCO
        'test_images',            # Images de test
        'utils',                  # Utilitaires du modèle
        'models'                  # Architecture des modèles
    }
    
    # === DOSSIERS À SUPPRIMER (REDONDANTS/INUTILES) ===
    remove_folders = {
        'output',                 # Probablement vide ou ancien
        'output_pretrained',      # Doublon avec output_fast
        'detection_results',      # Résultats de test (peuvent être regénérés)
        'detection_results_extended',
        'model_comparison',       # Résultats de comparaison (peuvent être regénérés)
        '__pycache__',           # Cache Python
        '.vscode'              # Configuration IDE
    }
    
    # === FICHIERS À GARDER (ESSENTIELS) ===
    keep_files = {
        # Configuration principale
        'config_extended.py',     # Config du meilleur modèle
        
        # Scripts d'entraînement
        'train_extended.py',      # Script d'entraînement principal
        
        # Scripts de test/évaluation
        'evaluate_all_models_complete.py',  # Script d'évaluation complet
        'final_coco_evaluation.py',         # Évaluation COCO
        
        # Script de déploiement/inférence
        'inference.py',           # Pour utiliser les modèles
        
        # Fichiers système
        '.gitignore',
        'README.md',             # Si existe
        'requirements.txt'       # Si existe
    }
    
    # === FICHIERS À SUPPRIMER (DOUBLONS/TESTS) ===
    remove_files = {
        # Configurations redondantes
        'config_fast.py',
        
        # Scripts d'entraînement redondants
        'train_fast.py',
        
        # Scripts de test multiples/redondants
        'test_model.py',
        'test_extended_model.py',
        'test_all_model_cmp.py',
        'test_all_models_auto.py',
        'simple_test.py',
        'smart_coco_test.py',
        
        # Scripts de téléchargement (peuvent être regénérés)
        'download_test_images.py',
        'capture_test_images.py',
        
        # Autres fichiers de développement
        'size_distribution.png',
        'class_distribution.png'
    }
    
    return keep_folders, remove_folders, keep_files, remove_files

def show_cleanup_plan():
    """Affiche le plan de nettoyage"""
    
    keep_folders, remove_folders, keep_files, remove_files = cleanup_project()
    
    print("📁 DOSSIERS À GARDER:")
    for folder in sorted(keep_folders):
        status = "✅" if os.path.exists(folder) else "❌"
        print(f"  {status} {folder}")
    
    print("\n🗑️ DOSSIERS À SUPPRIMER:")
    folders_to_delete = []
    for folder in sorted(remove_folders):
        if os.path.exists(folder):
            size = get_folder_size(folder)
            folders_to_delete.append((folder, size))
            print(f"  🗑️ {folder} ({size})")
        else:
            print(f"  ⚪ {folder} (n'existe pas)")
    
    print("\n📄 FICHIERS À GARDER:")
    for file in sorted(keep_files):
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")
    
    print("\n🗑️ FICHIERS À SUPPRIMER:")
    files_to_delete = []
    for file in sorted(remove_files):
        if os.path.exists(file):
            size = get_file_size(file)
            files_to_delete.append((file, size))
            print(f"  🗑️ {file} ({size})")
        else:
            print(f"  ⚪ {file} (n'existe pas)")
    
    # Calcul de l'espace libéré
    total_size = 0
    for _, size_str in folders_to_delete + files_to_delete:
        try:
            size_mb = float(size_str.replace('MB', '').replace('GB', '').replace('KB', ''))
            if 'GB' in size_str:
                size_mb *= 1024
            elif 'KB' in size_str:
                size_mb /= 1024
            total_size += size_mb
        except:
            pass
    
    print(f"\n💾 ESPACE TOTAL À LIBÉRER: ~{total_size:.1f} MB")
    
    return folders_to_delete, files_to_delete

def get_folder_size(folder_path):
    """Calcule la taille d'un dossier"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        # Convertir en unités lisibles
        if total_size > 1024 * 1024 * 1024:  # GB
            return f"{total_size / (1024 * 1024 * 1024):.1f}GB"
        elif total_size > 1024 * 1024:  # MB
            return f"{total_size / (1024 * 1024):.1f}MB"
        else:  # KB
            return f"{total_size / 1024:.1f}KB"
    except:
        return "?MB"

def get_file_size(file_path):
    """Calcule la taille d'un fichier"""
    try:
        size = os.path.getsize(file_path)
        if size > 1024 * 1024:  # MB
            return f"{size / (1024 * 1024):.1f}MB"
        else:  # KB
            return f"{size / 1024:.1f}KB"
    except:
        return "?KB"

def execute_cleanup():
    """Exécute le nettoyage"""
    
    keep_folders, remove_folders, keep_files, remove_files = cleanup_project()
    
    deleted_count = 0
    
    print("\n🗑️ SUPPRESSION DES DOSSIERS...")
    for folder in remove_folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"  ✅ Supprimé: {folder}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Erreur: {folder} - {e}")
    
    print("\n🗑️ SUPPRESSION DES FICHIERS...")
    for file in remove_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  ✅ Supprimé: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Erreur: {file} - {e}")
    
    print(f"\n✅ NETTOYAGE TERMINÉ: {deleted_count} éléments supprimés")

def create_project_structure_doc():
    """Crée la documentation de la structure finale"""
    
    structure = """
# 📁 STRUCTURE FINALE DU PROJET

## 🎯 STRUCTURE PROPRE ET ORGANISÉE

```
detection_model/
├── 🤖 MODÈLES
│   ├── output_extended_30/          # Modèles 28 classes (CHAMPION: epoch_9)
│   ├── output_fast/                 # Modèles 10 classes (comparaison)
│   ├── models/                      # Architecture des modèles
│   └── utils/                       # Utilitaires
│
├── 🧪 ÉVALUATION
│   ├── coco_evaluation/             # Dataset COCO + résultats
│   └── test_images/                 # Images de test personnalisées
│
├── 📜 SCRIPTS ESSENTIELS
│   ├── config_extended.py           # Configuration principale
│   ├── train_extended.py            # Entraînement
│   ├── evaluate_all_models_complete.py  # Évaluation complète
│   ├── final_coco_evaluation.py     # Évaluation COCO
│   └── inference.py                 # Déploiement/utilisation
│
└── 📋 DOCUMENTATION
    └── PROJECT_STRUCTURE.md         # Ce fichier
```

## 🏆 MODÈLE RECOMMANDÉ

**Meilleur modèle:** `output_extended_30/extended_model_epoch_9.pth`
- ✅ F1-Score: 0.500 (50%)
- ✅ 28 classes d'objets perdus
- ✅ Équilibre optimal précision/rappel

## 🚀 UTILISATION

### Évaluer tous les modèles:
```bash
python evaluate_all_models_complete.py
```

### Évaluation COCO quantitative:
```bash
python final_coco_evaluation.py
```

### Utiliser le modèle pour la détection:
```bash
python inference.py
```

## 📊 CLASSES DÉTECTABLES (28)

**Objets personnels:** backpack, suitcase, handbag, tie, hair drier, toothbrush
**Électronique:** cell phone, laptop, keyboard, mouse, remote, tv
**Cuisine:** bottle, cup, bowl, knife, spoon, fork, wine glass
**Bureau/Maison:** scissors, book, clock, umbrella, vase, potted plant
**Sport/Transport:** bicycle, skateboard, sports ball
"""
    
    with open('PROJECT_STRUCTURE.md', 'w', encoding='utf-8') as f:
        f.write(structure)
    
    print("📝 Documentation créée: PROJECT_STRUCTURE.md")

def main():
    print("🧹 NETTOYAGE DU PROJET DE DÉTECTION D'OBJETS")
    print("="*60)
    
    print("Ce script va nettoyer votre projet en gardant seulement l'essentiel:")
    print("✅ Meilleurs modèles (epoch_9 = champion)")
    print("✅ Scripts principaux")
    print("✅ Données d'évaluation")
    print("🗑️ Supprime les doublons et fichiers de test")
    
    # Afficher le plan
    folders_to_delete, files_to_delete = show_cleanup_plan()
    
    if not folders_to_delete and not files_to_delete:
        print("\n✨ Le projet est déjà propre!")
        return
    
    print("\n⚠️ ATTENTION: Cette action est irréversible!")
    choice = input("\nConfirmer le nettoyage? (y/n): ").lower().strip()
    
    if choice == 'y':
        execute_cleanup()
        create_project_structure_doc()
        
        print("\n🎉 PROJET NETTOYÉ AVEC SUCCÈS!")
        print("📁 Structure finale documentée dans PROJECT_STRUCTURE.md")
        print("🏆 Votre champion: output_extended_30/extended_model_epoch_9.pth")
    else:
        print("❌ Nettoyage annulé")

if __name__ == "__main__":
    main()