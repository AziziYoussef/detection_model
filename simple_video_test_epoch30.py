#!/usr/bin/env python3
"""
Script SIMPLE pour tester l'Époque 30 sur vidéos
Support: MP4, AVI, MOV, MKV, WMV, MPG, MPEG
Usage: python simple_video_test_epoch30.py
"""

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob

# Configuration simple
EPOCH_30_MODEL = "output_stable_training/stable_model_epoch_30.pth"
VIDEOS_DIR = "videos"
OUTPUT_DIR = "videos_output"
CONFIDENCE = 0.5  # Seuil optimal démontré

# Classes du modèle (28 classes)
CLASSES_FR = [
    'Personne', 'Sac à dos', 'Valise', 'Sac à main', 'Cravate',
    'Parapluie', 'Sèche-cheveux', 'Brosse à dents', 'Téléphone',
    'Ordinateur portable', 'Clavier', 'Souris', 'Télécommande', 'Télévision',
    'Horloge', 'Micro-ondes', 'Bouteille', 'Tasse', 'Bol',
    'Couteau', 'Cuillère', 'Fourchette', 'Verre', 'Réfrigérateur',
    'Ciseaux', 'Livre', 'Vase', 'Chaise'
]

def load_epoch30_model():
    """Charge le modèle epoch 30"""
    print("🤖 Chargement du modèle Époque 30...")
    
    if not os.path.exists(EPOCH_30_MODEL):
        print(f"❌ Modèle non trouvé: {EPOCH_30_MODEL}")
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)  # 28 + fond
        
        model.load_state_dict(torch.load(EPOCH_30_MODEL, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle epoch 30 chargé sur {device}")
        return model, device
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None, None

def find_videos():
    """Trouve toutes les vidéos dans le dossier videos"""
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Créez le dossier '{VIDEOS_DIR}' et placez-y vos vidéos")
        return []
    
    # Extensions vidéo supportées (AVEC MPG/MPEG AJOUTÉS)
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.mpg', '*.mpeg', '*.m4v']
    videos = []
    
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext.upper())))
    
    if not videos:
        print(f"❌ Aucune vidéo trouvée dans '{VIDEOS_DIR}'")
        print("🎬 Formats supportés: .mp4, .avi, .mov, .mkv, .wmv, .mpg, .mpeg, .m4v")
        print("💡 Placez vos vidéos dans le dossier 'videos/'")
        return []
    
    print(f"📹 {len(videos)} vidéo(s) trouvée(s):")
    for i, video in enumerate(videos, 1):
        # Obtenir info sur la vidéo
        try:
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Taille du fichier
                size_mb = os.path.getsize(video) / (1024 * 1024)
                
                print(f"  {i:2d}. {os.path.basename(video):<30} "
                      f"({width}x{height}, {fps}fps, {duration:.1f}s, {size_mb:.1f}MB)")
                cap.release()
            else:
                print(f"  {i:2d}. {os.path.basename(video):<30} (❌ Erreur lecture)")
        except:
            print(f"  {i:2d}. {os.path.basename(video):<30} (❓ Info indisponible)")
    
    return videos

def select_video(videos):
    """Sélectionne une vidéo à traiter"""
    if len(videos) == 1:
        print(f"📹 Vidéo sélectionnée automatiquement: {os.path.basename(videos[0])}")
        return videos[0]
    
    print(f"\n🎯 SÉLECTION DE LA VIDÉO:")
    while True:
        try:
            choice = input(f"\nChoisissez une vidéo (1-{len(videos)}) ou 'q' pour quitter: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                selected = videos[idx]
                print(f"✅ Sélectionné: {os.path.basename(selected)}")
                return selected
            else:
                print(f"❌ Choisissez un nombre entre 1 et {len(videos)}")
        except ValueError:
            print("❌ Entrez un nombre valide ou 'q' pour quitter")

def test_video_compatibility(video_path):
    """Test la compatibilité de la vidéo"""
    print(f"\n🔍 Test de compatibilité: {os.path.basename(video_path)}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la vidéo")
            print("💡 Essayez de convertir avec: ffmpeg -i input.mpg output.mp4")
            return False
        
        # Lire quelques frames de test
        ret, frame = cap.read()
        if not ret:
            print("❌ Impossible de lire les frames")
            cap.release()
            return False
        
        # Vérifier les propriétés
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Vidéo compatible:")
        print(f"   📏 Résolution: {width}x{height}")
        print(f"   🎬 FPS: {fps}")
        print(f"   🖼️ Frame test: OK")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

def process_video(model, device, video_path):
    """Traite une vidéo avec détection"""
    print(f"\n🎬 TRAITEMENT DE: {os.path.basename(video_path)}")
    print("="*60)
    
    # Test de compatibilité d'abord
    if not test_video_compatibility(video_path):
        return False
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Propriétés vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Propriétés: {width}x{height} à {fps} FPS")
    print(f"⏱️ Durée: {total_frames/fps:.1f} secondes ({total_frames} frames)")
    
    # Fichier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Garder l'extension originale dans le nom
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    original_ext = os.path.splitext(video_path)[1]
    output_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}{original_ext}")
    
    # Si c'est un MPG, convertir en MP4 pour la sortie
    if original_ext.lower() in ['.mpg', '.mpeg']:
        output_path = os.path.join(OUTPUT_DIR, f"detected_{base_name}.mp4")
        print(f"💡 Conversion {original_ext} → .mp4 pour la sortie")
    
    # Writer pour sauvegarder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transformation pour le modèle
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Couleurs aléatoires pour les classes
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
              for _ in range(28)]
    
    frame_count = 0
    total_detections = 0
    processing_times = []
    
    print("🚀 Traitement en cours...")
    print("💡 Traitement optimisé pour performance maximale")
    
    try:
        import time
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start_time = time.time()
            
            # Prétraitement pour le modèle
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # Détection
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            # Extraire résultats
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            # Filtrer par confiance
            mask = scores > CONFIDENCE
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            frame_detections = len(boxes)
            total_detections += frame_detections
            
            # Dessiner les détections
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = label - 1  # -1 car 0 est le fond
                
                if 0 <= class_idx < 28:
                    class_name = CLASSES_FR[class_idx]
                    color = colors[class_idx]
                    
                    # Rectangle de détection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texte avec nom et confiance
                    text = f"{class_name}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Fond du texte
                    cv2.rectangle(frame, (x1, y1-text_height-10), 
                                (x1+text_width, y1), color, -1)
                    
                    # Texte
                    cv2.putText(frame, text, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Informations sur la frame
            info_text = f"Frame: {frame_count}/{total_frames} | Objets: {frame_detections} | Total: {total_detections}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informations du modèle
            model_info = f"Modele: Epoque 30 | Format: {original_ext.upper()} | Confiance: {CONFIDENCE}"
            cv2.putText(frame, model_info, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Sauvegarder la frame
            writer.write(frame)
            
            # Mesurer performance
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            
            # Affichage progression
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(processing_times[-50:]) if processing_times else 0
                fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                
                print(f"📈 {progress:5.1f}% | {frame_count:4d}/{total_frames} frames | "
                      f"{total_detections:3d} détections | {fps_actual:.1f} fps")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par Ctrl+C")
    
    finally:
        # Nettoyage
        cap.release()
        writer.release()
        
        # Statistiques finales
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n📊 RÉSULTATS FINAUX:")
        print("="*50)
        print(f"   🎬 Frames traitées: {frame_count}/{total_frames}")
        print(f"   🔍 Total détections: {total_detections}")
        print(f"   📈 Moyenne par frame: {total_detections/frame_count:.1f}")
        print(f"   ⏱️ Temps total: {total_time:.1f}s")
        print(f"   🚀 Vitesse moyenne: {frame_count/total_time:.1f} fps")
        print(f"   💾 Vidéo sauvegardée: {output_path}")
        
        # Vérifier si le fichier de sortie existe
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   📁 Taille fichier: {size_mb:.1f} MB")
            print(f"   ✅ Format converti: {original_ext} → {os.path.splitext(output_path)[1]}")
        
        return True

def main():
    """Fonction principale"""
    print("="*70)
    print("🎬 TEST VIDÉO SIMPLE - MODÈLE ÉPOQUE 30 (AVEC SUPPORT MPG)")
    print("="*70)
    print("🏆 Performance: F1=49.86% | Précision=60.73%")
    print("🎯 Seuil optimal: 0.5")
    print("📹 Formats supportés: MP4, AVI, MOV, MKV, WMV, MPG, MPEG, M4V")
    
    # Charger le modèle
    model, device = load_epoch30_model()
    if model is None:
        return
    
    # Trouver les vidéos
    videos = find_videos()
    if not videos:
        return
    
    # Sélectionner une vidéo
    selected_video = select_video(videos)
    if not selected_video:
        print("❌ Aucune vidéo sélectionnée")
        return
    
    # Traiter la vidéo
    success = process_video(model, device, selected_video)
    
    if success:
        print(f"\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Résultat dans: {OUTPUT_DIR}/")
        print(f"💡 Conseil: Les fichiers MPG sont convertis en MP4 pour compatibilité")
    else:
        print(f"\n❌ Erreur lors du traitement")
        print(f"💡 Si problème avec MPG, essayez: ffmpeg -i video.mpg video.mp4")

if __name__ == "__main__":
    main()