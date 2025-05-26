#!/usr/bin/env python3
"""
Détection d'objets perdus en TEMPS RÉEL avec la webcam
Utilise le modèle Époque 30 (F1=49.86%, Précision=60.73%)
Usage: python webcam_detection_epoch30.py
"""

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time

# Configuration
EPOCH_30_MODEL = "output_stable_training/stable_model_epoch_30.pth"
CONFIDENCE = 0.55  # Seuil optimal démontré (60.73% précision)

# Classes du modèle (28 classes) - Noms français
CLASSES_FR = [
    'Personne', 'Sac à dos', 'Valise', 'Sac à main', 'Cravate',
    'Parapluie', 'Sèche-cheveux', 'Brosse à dents', 'Téléphone',
    'Ordinateur portable', 'Clavier', 'Souris', 'Télécommande', 'Télévision',
    'Horloge', 'Micro-ondes', 'Bouteille', 'Tasse', 'Bol',
    'Couteau', 'Cuillère', 'Fourchette', 'Verre', 'Réfrigérateur',
    'Ciseaux', 'Livre', 'Vase', 'Chaise'
]

def load_epoch30_model():
    """Charge le modèle champion Époque 30"""
    print("🤖 Chargement du modèle Époque 30...")
    print("🏆 Performance: F1=49.86% | Précision=60.73%")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)  # 28 + fond
        
        model.load_state_dict(torch.load(EPOCH_30_MODEL, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle chargé sur {device}")
        return model, device
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None, None

def setup_webcam():
    """Configure la webcam"""
    print("📹 Initialisation de la webcam...")
    
    # Essayer d'ouvrir la webcam (0 = webcam par défaut)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'accéder à la webcam")
        print("💡 Vérifiez que votre webcam n'est pas utilisée par une autre application")
        return None
    
    # Optimiser les paramètres webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur
    cap.set(cv2.CAP_PROP_FPS, 30)           # FPS
    
    # Vérifier les paramètres appliqués
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📊 Webcam configurée: {width}x{height} à {fps} FPS")
    return cap

def run_webcam_detection(model, device, cap):
    """Lance la détection en temps réel"""
    print("\n🚀 DÉTECTION EN TEMPS RÉEL ACTIVÉE")
    print("="*50)
    print("💡 Contrôles:")
    print("   • Appuyez 'q' pour quitter")
    print("   • Appuyez 's' pour prendre une capture")
    print("   • Appuyez 'r' pour réinitialiser les statistiques")
    print("="*50)
    
    # Transformation pour le modèle
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Couleurs pour les classes (consistantes)
    np.random.seed(42)
    colors = {}
    for i, class_name in enumerate(CLASSES_FR):
        colors[class_name] = (
            np.random.randint(50, 255),
            np.random.randint(50, 255), 
            np.random.randint(50, 255)
        )
    
    # Variables pour statistiques
    frame_count = 0
    total_detections = 0
    fps_counter = 0
    fps_start_time = time.time()
    avg_inference_time = 0
    screenshot_count = 0
    
    print("🎬 Webcam active! Montrez des objets à détecter...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur lecture webcam")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Mesurer le temps d'inférence
            inference_start = time.time()
            
            # Prétraitement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # Détection
            with torch.no_grad():
                predictions = model(frame_tensor)
            
            inference_time = (time.time() - inference_start) * 1000
            avg_inference_time = (avg_inference_time * (frame_count - 1) + inference_time) / frame_count
            
            # Extraire résultats
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            
            # Filtrer par confiance
            mask = scores > CONFIDENCE
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            
            current_detections = len(boxes)
            total_detections += current_detections
            
            # Dessiner les détections
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                class_idx = label - 1
                
                if 0 <= class_idx < 28:
                    class_name = CLASSES_FR[class_idx]
                    color = colors[class_name]
                    
                    # Rectangle avec couleur spéciale pour personnes
                    if class_name == 'Personne':
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Rouge épais
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texte avec fond
                    text = f"{class_name}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Fond du texte
                    cv2.rectangle(frame, (x1, y1-text_height-10), 
                                (x1+text_width+10, y1), color, -1)
                    
                    # Texte
                    cv2.putText(frame, text, (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculer FPS
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:  # Chaque seconde
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            else:
                current_fps = fps_counter / (current_time - fps_start_time) if current_time > fps_start_time else 0
            
            # Informations sur l'écran
            info_y = 30
            
            # Titre
            cv2.putText(frame, "DETECTION OBJETS PERDUS - EPOCH 30", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            
            # Détections actuelles
            cv2.putText(frame, f"Objets detectes: {current_detections}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            
            # Performance
            cv2.putText(frame, f"FPS: {current_fps:.1f} | Inference: {inference_time:.1f}ms", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += 20
            
            # Statistiques
            cv2.putText(frame, f"Total detections: {total_detections} | Frames: {frame_count}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Modèle info (en bas)
            frame_height = frame.shape[0]
            cv2.putText(frame, f"Modele: Epoch 30 | F1: 49.86% | Precision: 60.73% | Seuil: {CONFIDENCE}", 
                       (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Afficher la frame
            cv2.imshow('Detection Webcam - Epoch 30', frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Arrêt demandé par l'utilisateur")
                break
            elif key == ord('s'):
                # Capture d'écran
                screenshot_count += 1
                screenshot_name = f"capture_webcam_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Capture sauvée: {screenshot_name}")
            elif key == ord('r'):
                # Réinitialiser statistiques
                frame_count = 0
                total_detections = 0
                avg_inference_time = 0
                print("🔄 Statistiques réinitialisées")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par Ctrl+C")
    
    finally:
        # Statistiques finales
        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"   🎬 Frames traitées: {frame_count}")
        print(f"   🔍 Total détections: {total_detections}")
        print(f"   📈 Moyenne détections/frame: {total_detections/frame_count:.2f}")
        print(f"   ⚡ Temps inférence moyen: {avg_inference_time:.1f}ms")
        print(f"   📸 Captures prises: {screenshot_count}")
        
        # Nettoyage
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Fonction principale"""
    print("="*60)
    print("📹 DÉTECTION WEBCAM - MODÈLE ÉPOQUE 30")
    print("="*60)
    print("🏆 Le meilleur modèle testé:")
    print("   • F1-Score: 49.86%")
    print("   • Précision: 60.73%")
    print("   • Vitesse: 16.3 img/sec")
    print("   • 28 classes d'objets")
    print("="*60)
    
    # Vérifier si le modèle existe
    import os
    if not os.path.exists(EPOCH_30_MODEL):
        print(f"❌ Modèle non trouvé: {EPOCH_30_MODEL}")
        print("💡 Assurez-vous d'avoir entraîné le modèle epoch 30")
        return
    
    # Charger le modèle
    model, device = load_epoch30_model()
    if model is None:
        return
    
    # Configurer la webcam
    cap = setup_webcam()
    if cap is None:
        return
    
    # Lancer la détection
    run_webcam_detection(model, device, cap)
    
    print("\n✅ Session de détection terminée!")

if __name__ == "__main__":
    main()
