# find_all_models.py - Trouve tous vos modèles entraînés
import os
import glob
from datetime import datetime

# Fix pour l'erreur OMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_file_info(file_path):
    """Obtient les informations d'un fichier"""
    try:
        stat = os.stat(file_path)
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        return size_mb, modified
    except:
        return 0, None

def find_all_model_files():
    """Trouve tous les fichiers de modèles .pth"""
    
    print("🔍 RECHERCHE DE TOUS VOS MODÈLES")
    print("="*80)
    
    # Dossiers à chercher (incluant les sous-dossiers)
    search_paths = [
        ".",  # Dossier actuel
        "**/",  # Tous les sous-dossiers
    ]
    
    all_models = []
    
    # Recherche récursive de tous les fichiers .pth
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pth'):
                file_path = os.path.join(root, file)
                size_mb, modified = get_file_info(file_path)
                
                all_models.append({
                    'path': file_path,
                    'name': file,
                    'directory': root,
                    'size_mb': size_mb,
                    'modified': modified
                })
    
    return all_models

def categorize_models(models):
    """Catégorise les modèles par type"""
    categories = {
        'stable': [],
        'improved': [],
        'extended': [],
        'fast': [], 
        'pretrained': [],
        'other': []
    }
    
    for model in models:
        path_lower = model['path'].lower()
        name_lower = model['name'].lower()
        
        if 'stable' in path_lower or 'stable' in name_lower:
            categories['stable'].append(model)
        elif 'improved' in path_lower or 'improved' in name_lower:
            categories['improved'].append(model)
        elif 'extended' in path_lower or 'extended' in name_lower:
            categories['extended'].append(model)
        elif 'fast' in path_lower or 'fast' in name_lower:
            categories['fast'].append(model)
        elif 'pretrained' in path_lower or 'pretrained' in name_lower:
            categories['pretrained'].append(model)
        else:
            categories['other'].append(model)
    
    return categories

def display_models_by_category(categories):
    """Affiche les modèles par catégorie"""
    
    total_models = sum(len(models) for models in categories.values())
    total_size = sum(model['size_mb'] for models in categories.values() for model in models)
    
    print(f"📊 RÉSUMÉ GLOBAL:")
    print(f"  🔢 Total de modèles: {total_models}")
    print(f"  💾 Espace total: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print()
    
    category_names = {
        'stable': '🛡️ MODÈLES STABILISÉS (Derniers entraînés)',
        'improved': '🚀 MODÈLES AMÉLIORÉS', 
        'extended': '📊 MODÈLES ÉTENDUS',
        'fast': '⚡ MODÈLES RAPIDES',
        'pretrained': '🏗️ MODÈLES PRÉ-ENTRAÎNÉS',
        'other': '📦 AUTRES MODÈLES'
    }
    
    for category, models in categories.items():
        if not models:
            continue
            
        print(f"{category_names[category]}")
        print("="*60)
        print(f"📂 Nombre: {len(models)} modèles")
        print(f"💾 Taille: {sum(m['size_mb'] for m in models):.1f} MB")
        print()
        
        # Trier par date de modification (plus récent en premier)
        models_sorted = sorted(models, key=lambda x: x['modified'] or datetime.min, reverse=True)
        
        for i, model in enumerate(models_sorted, 1):
            # Type de modèle
            if 'best' in model['name'].lower():
                model_type = "🏆"
            elif 'epoch' in model['name'].lower():
                # Extraire le numéro d'époque
                import re
                match = re.search(r'epoch_(\d+)', model['name'])
                epoch_num = match.group(1) if match else "?"
                model_type = f"📅 Ep.{epoch_num}"
            else:
                model_type = "📦"
            
            # Date formatée
            date_str = model['modified'].strftime("%d/%m %H:%M") if model['modified'] else "?"
            
            print(f"  {i:2d}. {model_type} {model['name']:<35} "
                  f"({model['size_mb']:6.1f} MB) "
                  f"📅 {date_str} "
                  f"📁 {model['directory']}")
        
        print()

def find_training_logs():
    """Trouve les logs et graphiques d'entraînement"""
    print("📈 LOGS ET GRAPHIQUES D'ENTRAÎNEMENT:")
    print("="*60)
    
    # Chercher les images de progression
    image_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(keyword in file.lower() for keyword in ['progress', 'loss', 'training']):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
    
    if image_files:
        print("📊 Graphiques de progression trouvés:")
        for img_file in image_files:
            print(f"  📈 {img_file}")
    else:
        print("❌ Aucun graphique de progression trouvé")
    
    print()

def suggest_cleanup():
    """Suggère un nettoyage des modèles"""
    print("🧹 SUGGESTIONS DE NETTOYAGE:")
    print("="*60)
    print("Pour économiser l'espace disque, vous pourriez supprimer:")
    print("  • Les modèles d'époques intermédiaires (gardez seulement epoch_30)")
    print("  • Les anciens modèles moins performants") 
    print("  • Les modèles de test/debug")
    print()
    print("⚠️ GARDEZ TOUJOURS:")
    print("  🏆 best_stable_model.pth (votre champion)")
    print("  📅 Le modèle de la dernière époque")
    print("  📊 Les modèles que vous voulez comparer")
    print()

def analyze_training_progression():
    """Analyse la progression de l'entraînement"""
    print("📈 ANALYSE DE LA PROGRESSION:")
    print("="*60)
    
    # Chercher les modèles avec numéros d'époque
    epoch_models = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pth') and 'epoch' in file.lower():
                import re
                match = re.search(r'epoch_(\d+)', file)
                if match:
                    epoch_num = int(match.group(1))
                    file_path = os.path.join(root, file)
                    size_mb, modified = get_file_info(file_path)
                    
                    epoch_models.append({
                        'epoch': epoch_num,
                        'file': file,
                        'path': file_path,
                        'size_mb': size_mb,
                        'modified': modified
                    })
    
    if epoch_models:
        # Trier par époque
        epoch_models.sort(key=lambda x: x['epoch'])
        
        print(f"📊 Progression d'entraînement détectée:")
        print(f"  🎯 Époque de départ: {epoch_models[0]['epoch']}")
        print(f"  🏁 Époque finale: {epoch_models[-1]['epoch']}")
        print(f"  📂 Nombre d'époques sauvegardées: {len(epoch_models)}")
        
        # Afficher quelques époques clés
        key_epochs = [epoch_models[0], epoch_models[len(epoch_models)//2], epoch_models[-1]]
        print(f"\n📋 Époques clés:")
        for model in key_epochs:
            date_str = model['modified'].strftime("%d/%m %H:%M") if model['modified'] else "?"
            print(f"  📅 Époque {model['epoch']:2d}: {model['file']:<30} ({date_str})")
    else:
        print("❌ Aucune progression d'époque détectée")
    
    print()

def main():
    print("🔍 ANALYSE COMPLÈTE DE VOS MODÈLES")
    print("="*80)
    print("Ce script va analyser tous vos modèles PyTorch (.pth)")
    print()
    
    # Trouver tous les modèles
    all_models = find_all_model_files()
    
    if not all_models:
        print("❌ Aucun modèle (.pth) trouvé dans ce dossier et ses sous-dossiers!")
        print("Vérifiez que vous êtes dans le bon répertoire.")
        return
    
    # Catégoriser et afficher
    categories = categorize_models(all_models)
    display_models_by_category(categories)
    
    # Analyser la progression
    analyze_training_progression()
    
    # Trouver les logs
    find_training_logs()
    
    # Suggestions
    suggest_cleanup()
    
    # Informations pour le debug
    print("🔧 INFORMATIONS DE DEBUG:")
    print("="*60)
    print(f"📂 Dossier actuel: {os.getcwd()}")
    print(f"📁 Sous-dossiers détectés:")
    for root, dirs, files in os.walk("."):
        if root != "." and any(f.endswith('.pth') for f in files):
            pth_count = sum(1 for f in files if f.endswith('.pth'))
            print(f"  📁 {root}: {pth_count} modèle(s)")

if __name__ == "__main__":
    main()