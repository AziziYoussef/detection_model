# compare_all_models.py - Test comparatif de tous vos modèles
import os
import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time

from config_improved import config

class ModelComparator:
    """Comparateur pour tester tous les modèles sur une image"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"🖥️ Device: {self.device}")
        print(f"📊 Classes: {len(config['classes'])}")
    
    def find_all_models(self):
        """Trouve tous les modèles disponibles"""
        models = []
        
        # Dossiers à chercher
        search_dirs = [
            'output_stable_training',
            'output_improved_with_persons', 
            'output_extended_30',
            'output_fast',
            'output',
            'checkpoints'
        ]
        
        print("🔍 Recherche de modèles...")
        
        for directory in search_dirs:
            if os.path.exists(directory):
                # Chercher tous les fichiers .pth
                model_files = glob.glob(os.path.join(directory, '*.pth'))
                
                for model_file in model_files:
                    # Extraire le nom et créer une description
                    filename = os.path.basename(model_file)
                    
                    # Déterminer le type de modèle
                    if 'best' in filename:
                        model_type = "🏆 Meilleur"
                    elif 'epoch' in filename:
                        epoch_num = self.extract_epoch_number(filename)
                        model_type = f"📅 Époque {epoch_num}"
                    else:
                        model_type = "📦 Standard"
                    
                    # Taille du fichier
                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    
                    models.append({
                        'name': filename,
                        'path': model_file,
                        'directory': directory,
                        'type': model_type,
                        'size_mb': size_mb,
                        'display_name': f"{model_type} - {directory}"
                    })
        
        # Trier par dossier puis par type
        models.sort(key=lambda x: (x['directory'], 'best' not in x['name'], x['name']))
        
        print(f"✅ {len(models)} modèles trouvés:")
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model['display_name']:<40} ({model['size_mb']:.1f} MB)")
        
        return models
    
    def extract_epoch_number(self, filename):
        """Extrait le numéro d'époque du nom de fichier"""
        import re
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    def load_model(self, model_path):
        """Charge un modèle"""
        try:
            model = detection_models.fasterrcnn_resnet50_fpn(pretrained=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.config['classes']) + 1)
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Erreur chargement {os.path.basename(model_path)}: {e}")
            return None
    
    def preprocess_image(self, image_path):
        """Prétraite une image"""
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"📷 Image: {os.path.basename(image_path)} - {image.size}")
        except Exception as e:
            print(f"❌ Erreur image: {e}")
            raise
        
        original_image = np.array(image)
        original_size = image.size
        
        # Redimensionner vers la taille d'entraînement
        target_size = self.config['image_size']
        image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Facteurs d'échelle
        scale_x = original_size[0] / target_size[0]
        scale_y = original_size[1] / target_size[1]
        
        image_tensor = self.transform(image_resized)
        return image_tensor, original_image, (scale_x, scale_y)
    
    def test_model_on_image(self, model, image_tensor, scale_factors, confidence_threshold=0.3):
        """Teste un modèle sur une image"""
        if model is None:
            return None
        
        start_time = time.time()
        
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            predictions = model(image_batch)
        
        inference_time = (time.time() - start_time) * 1000  # en ms
        
        # Extraire prédictions
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        
        # Filtrer par confiance
        mask = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        # Remettre à l'échelle
        if len(pred_boxes) > 0:
            scale_x, scale_y = scale_factors
            pred_boxes[:, [0, 2]] *= scale_x
            pred_boxes[:, [1, 3]] *= scale_y
        
        # Analyser les détections
        person_count = 0
        object_count = 0
        class_counts = {}
        
        for label in pred_labels:
            if 1 <= label <= len(self.config['classes']):
                class_name = self.config['classes'][label - 1]
                if class_name == 'person':
                    person_count += 1
                else:
                    object_count += 1
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'boxes': pred_boxes,
            'labels': pred_labels,
            'scores': pred_scores,
            'inference_time': inference_time,
            'person_count': person_count,
            'object_count': object_count,
            'total_detections': len(pred_labels),
            'class_counts': class_counts,
            'avg_confidence': np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
        }
    
    def visualize_comparison(self, original_image, results_dict, image_name, confidence_threshold):
        """Visualise la comparaison de tous les modèles"""
        num_models = len(results_dict)
        
        if num_models == 0:
            print("❌ Aucun résultat à afficher")
            return
        
        # Calculer la grille optimale
        cols = min(3, num_models)  # Maximum 3 colonnes
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        # S'assurer que axes est toujours une liste 2D
        if rows == 1:
            axes = [axes] if cols == 1 else [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        # Aplatir la liste pour faciliter l'itération
        axes_flat = [ax for row in axes for ax in row]
        
        model_names = list(results_dict.keys())
        
        for i, (model_name, result) in enumerate(results_dict.items()):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            # Afficher l'image
            ax.imshow(original_image)
            
            if result is not None:
                # Dessiner les détections
                self.draw_detections_on_axis(ax, result, model_name)
                
                # Titre avec statistiques
                title = f"{model_name}\n"
                title += f"👥 {result['person_count']} • 📦 {result['object_count']} • "
                title += f"⚡ {result['inference_time']:.0f}ms"
                ax.set_title(title, fontsize=10, weight='bold', pad=10)
            else:
                ax.set_title(f"{model_name}\n❌ Erreur", fontsize=10, color='red')
            
            ax.axis('off')
        
        # Masquer les axes non utilisés
        for i in range(len(results_dict), len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Titre général
        fig.suptitle(f'Comparaison de Modèles - {image_name}\nSeuil: {confidence_threshold}', 
                    fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Sauvegarder
        output_name = f"comparison_{os.path.splitext(image_name)[0]}.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"💾 Comparaison sauvegardée: {output_name}")
        
        plt.show()
    
    def draw_detections_on_axis(self, ax, result, model_name):
        """Dessine les détections sur un axe"""
        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            if len(box) == 0:
                continue
            
            x1, y1, x2, y2 = box
            
            if 1 <= label <= len(self.config['classes']):
                class_name = self.config['classes'][label - 1]
                
                # Couleur selon le type
                if class_name == 'person':
                    color = 'red'
                    linewidth = 2
                else:
                    color = 'lime'
                    linewidth = 1.5
                
                # Rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=linewidth, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Texte (seulement pour les détections de confiance élevée)
                if score > 0.5:
                    class_name_fr = self.config['class_names_fr'].get(class_name, class_name)
                    ax.text(x1, y1-2, f"{class_name_fr}: {score:.2f}", 
                           color='white', fontsize=8, weight='bold',
                           bbox=dict(facecolor=color, alpha=0.8, pad=1))
    
    def create_summary_table(self, results_dict):
        """Crée un tableau de résumé des performances"""
        print(f"\n{'='*80}")
        print("📊 TABLEAU COMPARATIF DE PERFORMANCE")
        print(f"{'='*80}")
        
        # En-tête
        header = f"{'Modèle':<25} {'Personnes':<9} {'Objets':<7} {'Total':<6} {'Temps':<8} {'Conf Moy':<8}"
        print(header)
        print("-" * 80)
        
        # Trier par nombre total de détections (décroissant)
        sorted_results = sorted(results_dict.items(), 
                              key=lambda x: x[1]['total_detections'] if x[1] else 0, 
                              reverse=True)
        
        for model_name, result in sorted_results:
            if result is not None:
                row = (f"{model_name:<25} "
                      f"{result['person_count']:<9} "
                      f"{result['object_count']:<7} "
                      f"{result['total_detections']:<6} "
                      f"{result['inference_time']:<8.0f} "
                      f"{result['avg_confidence']:<8.3f}")
                print(row)
            else:
                print(f"{model_name:<25} {'❌ ERREUR':<40}")
        
        # Trouver le champion
        if sorted_results and sorted_results[0][1] is not None:
            champion = sorted_results[0]
            print(f"\n🏆 CHAMPION: {champion[0]}")
            print(f"   📊 {champion[1]['total_detections']} détections")
            print(f"   ⚡ {champion[1]['inference_time']:.0f}ms")
            print(f"   🎯 Confiance moyenne: {champion[1]['avg_confidence']:.3f}")
    
    def select_image_from_directory(self):
        """Sélectionne une image depuis un répertoire"""
        # Dossiers possibles
        image_dirs = ['test_images', 'images', 'samples', '.']
        
        selected_dir = None
        for dir_name in image_dirs:
            if os.path.exists(dir_name):
                images = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    images.extend(glob.glob(os.path.join(dir_name, f'*{ext}')))
                    images.extend(glob.glob(os.path.join(dir_name, f'*{ext.upper()}')))
                
                if images:
                    selected_dir = dir_name
                    break
        
        if not selected_dir:
            print("❌ Aucun dossier d'images trouvé")
            return None
        
        # Lister les images disponibles
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            images.extend(glob.glob(os.path.join(selected_dir, f'*{ext}')))
            images.extend(glob.glob(os.path.join(selected_dir, f'*{ext.upper()}')))
        
        print(f"\n📁 Images disponibles dans '{selected_dir}':")
        for i, img_path in enumerate(images, 1):
            img_name = os.path.basename(img_path)
            try:
                with Image.open(img_path) as img:
                    size_info = f"{img.size[0]}x{img.size[1]}"
            except:
                size_info = "?"
            print(f"  {i:2d}. {img_name:<30} ({size_info})")
        
        # Sélection
        while True:
            try:
                choice = input(f"\nChoisissez une image (1-{len(images)}) ou 'q' pour quitter: ").strip()
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(images):
                    return images[idx]
                else:
                    print(f"❌ Choisissez un nombre entre 1 et {len(images)}")
            except ValueError:
                print("❌ Entrez un nombre valide")
    
    def run_comparison(self):
        """Lance la comparaison complète"""
        print("="*80)
        print("🔥 COMPARAISON DE TOUS VOS MODÈLES")
        print("="*80)
        
        # Trouver tous les modèles
        all_models = self.find_all_models()
        
        if not all_models:
            print("❌ Aucun modèle trouvé!")
            return
        
        # Sélectionner les modèles à tester
        print(f"\n🎯 OPTIONS:")
        print(f"1. Tester TOUS les modèles ({len(all_models)})")
        print(f"2. Tester seulement les meilleurs modèles")
        print(f"3. Sélection personnalisée")
        
        choice = input("\nVotre choix (1/2/3): ").strip()
        
        if choice == "2":
            # Seulement les meilleurs
            selected_models = [m for m in all_models if 'best' in m['name']]
        elif choice == "3":
            # Sélection personnalisée
            print("\nSélectionnez les modèles (numéros séparés par des virgules):")
            indices = input("Ex: 1,3,5: ").strip().split(',')
            selected_models = []
            for idx in indices:
                try:
                    selected_models.append(all_models[int(idx) - 1])
                except (ValueError, IndexError):
                    pass
        else:
            # Tous les modèles
            selected_models = all_models
        
        if not selected_models:
            print("❌ Aucun modèle sélectionné")
            return
        
        print(f"\n✅ {len(selected_models)} modèles sélectionnés")
        
        # Sélectionner l'image
        image_path = self.select_image_from_directory()
        if not image_path:
            print("❌ Aucune image sélectionnée")
            return
        
        # Seuil de confiance
        conf_thresh = input("\nSeuil de confiance (défaut 0.3): ").strip()
        try:
            confidence_threshold = float(conf_thresh) if conf_thresh else 0.3
        except ValueError:
            confidence_threshold = 0.3
        
        print(f"\n🚀 Démarrage des tests...")
        print(f"📷 Image: {os.path.basename(image_path)}")
        print(f"🎯 Seuil: {confidence_threshold}")
        print(f"🤖 Modèles: {len(selected_models)}")
        
        # Prétraiter l'image une seule fois
        image_tensor, original_image, scale_factors = self.preprocess_image(image_path)
        
        # Tester tous les modèles
        results = {}
        
        for i, model_info in enumerate(selected_models, 1):
            model_name = model_info['display_name']
            print(f"\n🧪 Test {i}/{len(selected_models)}: {model_name}")
            
            # Charger et tester le modèle
            model = self.load_model(model_info['path'])
            result = self.test_model_on_image(model, image_tensor, scale_factors, confidence_threshold)
            
            results[model_name] = result
            
            if result:
                print(f"   ✅ {result['total_detections']} détections en {result['inference_time']:.0f}ms")
            else:
                print(f"   ❌ Échec")
            
            # Libérer la mémoire
            if model:
                del model
                torch.cuda.empty_cache()
        
        # Créer la visualisation comparative
        print(f"\n🎨 Création de la visualisation...")
        self.visualize_comparison(original_image, results, os.path.basename(image_path), confidence_threshold)
        
        # Afficher le tableau de résumé
        self.create_summary_table(results)
        
        print(f"\n✅ Comparaison terminée!")

def main():
    """Fonction principale"""
    comparator = ModelComparator(config)
    comparator.run_comparison()

if __name__ == "__main__":
    main()