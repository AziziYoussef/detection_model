#!/usr/bin/env python3
"""
Script ultra-simple : Prend la première vidéo du dossier 'videos' et la traite
Usage: python simple_videos_detection.py
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
CONFIDENCE = 0.4
VIDEO_DIR = "videos"
OUTPUT_DIR = "videos_output"

# Classes (28 objets perdus)
CLASSES_FR = [
    'Sac à dos', 'Valise', 'Sac à main', 'Cravate', 'Sèche-cheveux', 'Brosse à dents',
    'Téléphone', 'Ordinateur portable', 'Clavier', 'Souris', 'Télécommande', 'Télévision',
    'Bouteille', 'Tasse', 'Bol', 'Couteau', 'Cuillère', 'Fourchette', 'Verre',
    'Ciseaux', 'Livre', 'Horloge', 'Parapluie', 'Vase', 'Plante',
    'Vélo', 'Skateboard', 'Ballon'
]

def main():
    print("🎥 DÉTECTION SIMPLE - MODÈLE EPOCH 9")
    print("="*50)
    
    # 1. Trouver la première vidéo
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Créez le dossier '{VIDEO_DIR}' et placez-y une vidéo")
        return
    
    videos = glob.glob(f"{VIDEO_DIR}/*.mp4") + glob.glob(f"{VIDEO_DIR}/*.avi") + glob.glob(f"{VIDEO_DIR}/*.mov")
    
    if not videos:
        print(f"❌ Aucune vidéo trouvée dans '{VIDEO_DIR}'")
        print("Formats supportés: .mp4, .avi, .mov")
        return
    
    video_path = videos[0]  # Prendre la première
    print(f"📹 Vidéo sélectionnée: {os.path.basename(video_path)}")
    
    # 2. Charger le modèle epoch 9
    model_path = "output_extended_30/extended_model_epoch_9.pth"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    print("🤖 Chargement du modèle...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)  # 28 + fond
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Modèle prêt sur {device}")
    
    # 3. Traitement
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fichier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(video_path)}")
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"🚀 Traitement en cours...")
    
    # Couleurs
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
              for _ in range(28)]
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Détection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            # Filtrer
            mask = scores > CONFIDENCE
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            total_detections += len(boxes)
            
            # Annoter
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = label - 1
                
                if 0 <= class_idx < 28:
                    color = colors[class_idx]
                    class_name = CLASSES_FR[class_idx]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Info
            cv2.putText(frame, f"Frame: {frame_count} | Objets: {len(boxes)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Afficher et sauvegarder
            cv2.imshow('Détection Epoch 9', frame)
            writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress toutes les 100 frames
            if frame_count % 100 == 0:
                print(f"📈 Frame {frame_count} - {len(boxes)} objets détectés")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par Ctrl+C")
    
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎉 TERMINÉ!")
        print(f"📊 {frame_count} frames traitées")
        print(f"🔍 {total_detections} détections au total")
        print(f"💾 Résultat: {output_path}")

if __name__ == "__main__":
    main()