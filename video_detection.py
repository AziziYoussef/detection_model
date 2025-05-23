#!/usr/bin/env python3
"""
Script rapide pour détecter les objets perdus dans une vidéo
Utilise le modèle epoch 9 (champion)
"""

import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import os

# Classes détectables (28 classes réelles du modèle epoch 9)
CLASSES_FR = {
    'backpack': 'Sac à dos',
    'suitcase': 'Valise', 
    'handbag': 'Sac à main',
    'tie': 'Cravate',
    'hair drier': 'Sèche-cheveux',
    'toothbrush': 'Brosse à dents',
    'cell phone': 'Téléphone',
    'laptop': 'Ordinateur portable',
    'keyboard': 'Clavier',
    'mouse': 'Souris',
    'remote': 'Télécommande',
    'tv': 'Télévision',
    'bottle': 'Bouteille',
    'cup': 'Tasse',
    'bowl': 'Bol',
    'knife': 'Couteau',
    'spoon': 'Cuillère',
    'fork': 'Fourchette',
    'wine glass': 'Verre',
    'scissors': 'Ciseaux',
    'book': 'Livre',
    'clock': 'Horloge',
    'umbrella': 'Parapluie',
    'vase': 'Vase',
    'potted plant': 'Plante',
    'bicycle': 'Vélo',
    'skateboard': 'Skateboard',
    'sports ball': 'Ballon'
}

CLASSES = list(CLASSES_FR.keys())

def load_model():
    """Charge le modèle epoch 9"""
    model_path = "output_extended_30/extended_model_epoch_9.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    # Créer l'architecture
    model = detection_models.fasterrcnn_resnet50_fpn(weights=None)  # Nouveau paramètre
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # CORRECTION: Utiliser 28 classes (nombre réel dans le modèle sauvegardé)
    actual_num_classes = 28
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, actual_num_classes + 1)
    
    # Charger les poids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Modèle epoch 9 chargé sur {device} avec {actual_num_classes} classes")
    return model, device

def detect_objects_in_video(video_path, output_path=None, confidence=0.5):
    """
    Détecte les objets dans une vidéo
    
    Args:
        video_path (str): Chemin vers la vidéo
        output_path (str): Chemin de sortie (optionnel)
        confidence (float): Seuil de confiance
    """
    print(f"🎬 Ouverture de la vidéo: {video_path}")
    
    # Charger le modèle
    model, device = load_model()
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir: {video_path}")
    
    # Propriétés vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 {width}x{height} à {fps} FPS, {total_frames} frames")
    
    # Writer pour sauvegarde
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Sauvegarde vers: {output_path}")
    
    # Couleurs pour les classes
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
              for _ in range(len(CLASSES))]
    
    frame_count = 0
    total_detections = 0
    
    print("🚀 Début du traitement...")
    print("Appuyez sur 'q' pour quitter")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Prétraitement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # Détection
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            # Traitement des résultats
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            # Filtrer par confiance
            mask = scores > confidence
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            frame_detections = len(boxes)
            total_detections += frame_detections
            
            # Annoter la frame
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = label - 1
                
                if 0 <= class_idx < len(CLASSES):
                    class_name = CLASSES[class_idx]
                    class_name_fr = CLASSES_FR[class_name]
                    color = colors[class_idx]
                    
                    # Rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texte
                    text = f"{class_name_fr}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Fond du texte
                    cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Informations sur la frame
            info = f"Frame: {frame_count}/{total_frames} | Objets: {frame_detections}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Affichage
            cv2.imshow('Détection Objets Perdus - Epoch 9', frame)
            
            # Sauvegarde
            if writer:
                writer.write(frame)
            
            # Contrôle (quitter avec 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Affichage progression (toutes les 30 frames)
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progression: {progress:.1f}% - {frame_detections} objets détectés")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé par l'utilisateur")
    
    finally:
        # Nettoyage
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Statistiques finales
        print("\n📊 RÉSULTATS FINAUX:")
        print(f"   🎬 Frames traitées: {frame_count}")
        print(f"   🔍 Total détections: {total_detections}")
        print(f"   📈 Moyenne par frame: {total_detections/frame_count:.1f}")

def detect_webcam():
    """Détection en temps réel sur webcam"""
    print("📹 Démarrage webcam...")
    
    model, device = load_model()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la webcam")
    
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
              for _ in range(len(CLASSES))]
    
    print("🚀 Webcam active - Appuyez sur 'q' pour quitter")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Détection (même logique que pour vidéo)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy() 
            scores = predictions[0]['scores'].cpu().numpy()
            
            mask = scores > 0.5
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            # Annotation
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = label - 1
                
                if 0 <= class_idx < len(CLASSES):
                    class_name = CLASSES[class_idx]
                    class_name_fr = CLASSES_FR[class_name]
                    color = colors[class_idx]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{class_name_fr}: {score:.2f}"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Informations
            cv2.putText(frame, f"Webcam - Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Webcam - Objets Perdus', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Frames webcam traitées: {frame_count}")

if __name__ == "__main__":
    print("="*60)
    print("🏆 DÉTECTION RAPIDE - MODÈLE EPOCH 9 (CHAMPION)")
    print("="*60)
    
    mode = input("Mode (1=Vidéo, 2=Webcam): ").strip()
    
    if mode == "1":
        video_file = input("Nom du fichier vidéo: ").strip()
        if not video_file:
            video_file = "test.mp4"
        
        output_file = f"detected_{video_file}"
        
        try:
            detect_objects_in_video(video_file, output_file)
            print(f"✅ Vidéo traitée et sauvegardée: {output_file}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    elif mode == "2":
        try:
            detect_webcam()
        except Exception as e:
            print(f"❌ Erreur webcam: {e}")
    
    else:
        print("❌ Mode invalide")