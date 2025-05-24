#!/usr/bin/env python3
"""
Script SIMPLE pour tester l'Époque 30 sur vidéos
Usage: python test_video_epoch30.py
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
    
    # Extensions vidéo supportées
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    videos = []
    
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext.upper())))
    
    if not videos:
        print(f"❌ Aucune vidéo trouvée dans '{VIDEOS_DIR}'")
        print("Formats supportés: .mp4, .avi, .mov, .mkv, .wmv")
        return []
    
    print(f"📹 {len(videos)} vidéo(s) trouvée(s):")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    return videos

def select_video(videos):
    """Sélectionne une vidéo à traiter"""
    if len(videos) == 1:
        print(f"📹 Vidéo sélectionnée: {os.path.basename(videos[0])}")
        return videos[0]
    
    while True:
        try:
            choice = input(f"\nChoisissez une vidéo (1-{len(videos)}) ou 'q' pour quitter: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
            else:
                print(f"❌ Choisissez un nombre entre 1 et {len(videos)}")
        except ValueError:
            print("❌ Entrez un nombre valide")

def process_video(model, device, video_path):
    """Traite une vidéo avec détection"""
    print(f"\n🎬 Traitement de: {os.path.basename(video_path)}")
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir: {video_path}")
        return False
    
    # Propriétés vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Vidéo: {width}x{height} à {fps} FPS, {total_frames} frames")
    print(f"⏱️ Durée: {total_frames/fps:.1f} secondes")
    
    # Fichier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(video_path)}")
    
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
    
    print("🚀 Traitement en cours (vitesse maximale)...")
    print("💡 Traitement sans affichage pour performance optimale")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
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
            model_info = f"Modele: Epoque 30 | Confiance: {CONFIDENCE} | F1: 49.86%"
            cv2.putText(frame, model_info, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Sauvegarder la frame directement (pas d'affichage)
            writer.write(frame)
            
            # Affichage progression moins fréquent pour plus de vitesse
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progression: {progress:.1f}% - {total_detections} détections totales")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par Ctrl+C")
    
    finally:
        # Nettoyage
        cap.release()
        writer.release()
        
        # Statistiques finales
        print(f"\n📊 RÉSULTATS FINAUX:")
        print(f"   🎬 Frames traitées: {frame_count}/{total_frames}")
        print(f"   🔍 Total détections: {total_detections}")
        print(f"   📈 Moyenne par frame: {total_detections/frame_count:.1f}")
        print(f"   💾 Vidéo sauvegardée: {output_path}")
        
        return True

def main():
    """Fonction principale"""
    print("="*60)
    print("🎬 TEST VIDÉO SIMPLE - MODÈLE ÉPOQUE 30")
    print("="*60)
    print("🏆 Performance: F1=49.86% | Précision=60.73%")
    print("🎯 Seuil optimal: 0.5")
    
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
    else:
        print(f"\n❌ Erreur lors du traitement")

if __name__ == "__main__":
    main()