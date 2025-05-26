#!/usr/bin/env python3
"""
Script ULTRA-RAPIDE pour traiter des vidéos longues - VERSION DEBUG FIXÉE
Le problème était dans la logique de batch processing
Usage: python fast_video_detection_epoch30_fixed.py
"""

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
import time

# Configuration optimisée
EPOCH_30_MODEL = "output_stable_training/stable_model_epoch_30.pth"
VIDEOS_DIR = "videos"
OUTPUT_DIR = "videos_output"
CONFIDENCE = 0.5

# MODES DE VITESSE
SPEED_MODES = {
    'ULTRA_FAST': {
        'skip_frames': 10,
        'max_resolution': 416,
        'batch_size': 4,
        'description': '10x plus rapide - Idéal pour vidéos très longues'
    },
    'FAST': {
        'skip_frames': 5,
        'max_resolution': 640,
        'batch_size': 2,
        'description': '5x plus rapide - Bon équilibre vitesse/qualité'
    },
    'BALANCED': {
        'skip_frames': 3,
        'max_resolution': 720,
        'batch_size': 1,
        'description': '3x plus rapide - Qualité préservée'
    },
    'QUALITY': {
        'skip_frames': 1,
        'max_resolution': 1080,
        'batch_size': 1,
        'description': 'Vitesse normale - Qualité maximale'
    }
}

CLASSES_FR = [
    'Personne', 'Sac à dos', 'Valise', 'Sac à main', 'Cravate',
    'Parapluie', 'Sèche-cheveux', 'Brosse à dents', 'Téléphone',
    'Ordinateur portable', 'Clavier', 'Souris', 'Télécommande', 'Télévision',
    'Horloge', 'Micro-ondes', 'Bouteille', 'Tasse', 'Bol',
    'Couteau', 'Cuillère', 'Fourchette', 'Verre', 'Réfrigérateur',
    'Ciseaux', 'Livre', 'Vase', 'Chaise'
]

class FastVideoProcessor:
    def __init__(self, model, device, speed_mode='FAST'):
        self.model = model
        self.device = device
        self.speed_config = SPEED_MODES[speed_mode]
        self.use_fp16 = device.type == 'cuda'
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.colors = self._generate_colors()
        
        print(f"⚡ Mode sélectionné: {speed_mode}")
        print(f"📋 Configuration: {self.speed_config['description']}")
        print(f"🔧 Debug: skip={self.speed_config['skip_frames']}, batch={self.speed_config['batch_size']}")
    
    def _generate_colors(self):
        np.random.seed(42)
        return [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
                for _ in range(28)]
    
    def _resize_frame_smart(self, frame):
        h, w = frame.shape[:2]
        max_res = self.speed_config['max_resolution']
        
        if max(h, w) <= max_res:
            return frame, 1.0, 1.0
        
        if w > h:
            new_w = max_res
            new_h = int(h * max_res / w)
        else:
            new_h = max_res
            new_w = int(w * max_res / h)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = w / new_w
        scale_y = h / new_h
        
        return resized, scale_x, scale_y
    
    def _process_frame_single(self, frame, scale_x, scale_y):
        """Traite une seule frame (plus simple et plus sûr)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb)
        
        if self.use_fp16:
            frame_tensor = frame_tensor.half()
        
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                predictions = self.model(frame_tensor)
        except Exception as e:
            print(f"⚠️ Erreur inférence: {e}")
            if self.use_fp16:
                frame_tensor = frame_tensor.float()
                with torch.no_grad():
                    predictions = self.model(frame_tensor)
            else:
                raise e
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filtrer par confiance
        mask = scores > CONFIDENCE
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Remettre à l'échelle
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        return boxes, labels, scores
    
    def _draw_detections_fast(self, frame, boxes, labels, scores):
        detections_count = 0
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            class_idx = label - 1
            
            if 0 <= class_idx < 28:
                class_name = CLASSES_FR[class_idx]
                color = self.colors[class_idx]
                
                # RECTANGLE PLUS ÉPAIS (5 pixels au lieu de 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                
                # TEXTE PLUS GROS ET VISIBLE (seuil abaissé à 0.3)
                if score > 0.3:
                    text = f"{class_name}: {score:.2f}"  # Nom complet
                    
                    # POLICE PLUS GRANDE (1.2 au lieu de 0.5)
                    font_scale = 1.2
                    thickness = 3
                    
                    # Calculer la taille du texte pour le fond
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    
                    # FOND COLORÉ PLUS GRAND pour le texte
                    cv2.rectangle(frame, 
                                (x1, y1 - text_height - 15), 
                                (x1 + text_width + 10, y1), 
                                color, -1)
                    
                    # TEXTE BLANC AVEC CONTOUR NOIR pour meilleure visibilité
                    # Contour noir
                    cv2.putText(frame, text, (x1+5, y1-8), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
                    # Texte blanc par-dessus
                    cv2.putText(frame, text, (x1+5, y1-8), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                detections_count += 1
        
        return detections_count

def load_epoch30_model():
    print("🤖 Chargement du modèle Époque 30...")
    
    if not os.path.exists(EPOCH_30_MODEL):
        print(f"❌ Modèle non trouvé: {EPOCH_30_MODEL}")
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
        
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 29)
        
        model.load_state_dict(torch.load(EPOCH_30_MODEL, map_location=device))
        model.to(device)
        model.eval()
        
        optimizations = []
        if device.type == 'cuda':
            try:
                model.half()
                optimizations.append("Float16")
            except:
                model.float()
            optimizations.append("CUDA optimisé")
        
        opt_str = " + ".join(optimizations) if optimizations else "Standard"
        print(f"✅ Modèle chargé sur {device} ({opt_str})")
        return model, device
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None, None

def process_video_fixed(processor, video_path, speed_mode):
    """VERSION CORRIGÉE du traitement vidéo"""
    print(f"\n🚀 TRAITEMENT RAPIDE CORRIGÉ: {os.path.basename(video_path)}")
    print("="*70)
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo")
        return False
    
    # Propriétés
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    config = SPEED_MODES[speed_mode]
    
    print(f"📊 Vidéo: {width}x{height}, {fps}fps, {total_frames} frames")
    print(f"⚡ Mode: {speed_mode} - Skip: {config['skip_frames']}")
    
    # Fichier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"fast_detected_{base_name}_{speed_mode.lower()}_fixed.mp4")
    
    # Writer
    output_fps = max(1, fps // config['skip_frames'])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Variables de suivi
    frame_count = 0
    processed_frames = 0
    total_detections = 0
    start_time = time.time()
    
    print("🚀 Démarrage du traitement...")
    print("🔧 DEBUG: Mode frame par frame pour garantir le fonctionnement")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"🔚 Fin de vidéo atteinte à la frame {frame_count}")
                break
            
            frame_count += 1
            
            # Debug: afficher les premières frames
            if frame_count <= 5:
                print(f"🔧 DEBUG: Frame {frame_count} lue, taille: {frame.shape}")
            
            # Skip frames selon le mode
            if frame_count % config['skip_frames'] != 0:
                continue
            
            # Cette frame sera traitée
            processed_frames += 1
            
            # Debug pour les premières frames traitées
            if processed_frames <= 3:
                print(f"✅ DEBUG: Traitement frame {frame_count} (#{processed_frames})")
            
            # Redimensionner
            resized_frame, scale_x, scale_y = processor._resize_frame_smart(frame)
            
            if processed_frames <= 3:
                print(f"📏 DEBUG: Redimensionnement {frame.shape[:2]} → {resized_frame.shape[:2]}, scales: {scale_x:.2f}, {scale_y:.2f}")
            
            # Traitement individuel (plus sûr)
            try:
                boxes, labels, scores = processor._process_frame_single(resized_frame, scale_x, scale_y)
                
                if processed_frames <= 3:
                    print(f"🔍 DEBUG: {len(boxes)} détections trouvées")
                
            except Exception as e:
                print(f"❌ Erreur traitement frame {frame_count}: {e}")
                continue
            
            # Dessiner les détections
            detections = processor._draw_detections_fast(frame, boxes, labels, scores)
            total_detections += detections
            
            # Ajouter infos sur la frame
            info = f"Frame: {processed_frames} | Mode: {speed_mode} | Objets: {detections}"
            cv2.putText(frame, info, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Sauvegarder
            writer.write(frame)
            
            # Affichage progression
            if processed_frames % 50 == 0 or processed_frames <= 10:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_actual = processed_frames / elapsed if elapsed > 0 else 0
                
                print(f"📈 {progress:5.1f}% | Frame {frame_count}/{total_frames} | "
                      f"Traitées: {processed_frames} | Détections: {total_detections} | "
                      f"{fps_actual:.1f} fps")
    
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par Ctrl+C")
    
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        writer.release()
        
        # Statistiques finales
        end_time = time.time()
        total_time = end_time - start_time
        original_duration = total_frames / fps if fps > 0 else 0
        
        print(f"\n🏁 TRAITEMENT TERMINÉ (VERSION CORRIGÉE):")
        print("="*60)
        print(f"   🎬 Frames vidéo totales: {total_frames}")
        print(f"   ✅ Frames traitées: {processed_frames}")  
        print(f"   🎯 Ratio traitement: {processed_frames/total_frames*100:.1f}%")
        print(f"   🔍 Total détections: {total_detections}")
        if processed_frames > 0:
            print(f"   📊 Détections par frame: {total_detections/processed_frames:.2f}")
        print(f"   ⏱️ Temps traitement: {total_time:.1f}s")
        print(f"   📈 Durée originale: {original_duration:.1f}s")
        if total_time > 0:
            speed_boost = original_duration / total_time
            print(f"   🚀 Accélération: {speed_boost:.1f}x plus rapide")
            print(f"   📊 Vitesse traitement: {processed_frames/total_time:.1f} fps")
        print(f"   💾 Fichier: {output_path}")
        
        # Vérifier le fichier de sortie
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   📁 Taille fichier: {size_mb:.1f} MB")
            
            if size_mb < 1:
                print(f"   ⚠️ ATTENTION: Fichier très petit - vérifiez le contenu")
            else:
                print(f"   ✅ Fichier généré avec succès")
        else:
            print(f"   ❌ ERREUR: Fichier de sortie non créé")
            return False
        
        return processed_frames > 0

def find_videos():
    """Trouve les vidéos (version simplifiée sans doublons)"""
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Créez le dossier '{VIDEOS_DIR}' et placez-y vos vidéos")
        return []
    
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.mpg', '*.mpeg', '*.m4v']
    videos = []
    
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
        videos.extend(glob.glob(os.path.join(VIDEOS_DIR, ext.upper())))
    
    # Supprimer les doublons
    videos = list(set(videos))
    videos.sort()
    
    if not videos:
        print(f"❌ Aucune vidéo trouvée dans '{VIDEOS_DIR}'")
        return []
    
    print(f"📹 {len(videos)} vidéo(s) trouvée(s):")
    
    for i, video in enumerate(videos, 1):
        try:
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                size_mb = os.path.getsize(video) / (1024 * 1024)
                
                print(f"  {i:2d}. {os.path.basename(video):<30} "
                      f"({duration:6.1f}s, {size_mb:6.1f}MB)")
                cap.release()
            else:
                print(f"  {i:2d}. {os.path.basename(video):<30} (❌ Erreur)")
        except:
            print(f"  {i:2d}. {os.path.basename(video):<30} (❓ Erreur)")
    
    return videos

def select_speed_mode():
    print(f"\n⚡ MODES DE VITESSE:")
    print("="*50)
    
    for i, (mode, config) in enumerate(SPEED_MODES.items(), 1):
        print(f"{i}. {mode:<12} - {config['description']}")
        print(f"   Skip: {config['skip_frames']} | Max res: {config['max_resolution']}p")
    
    while True:
        try:
            choice = input(f"\nMode (1-{len(SPEED_MODES)}) ou ENTER pour FAST: ").strip()
            
            if not choice:
                return 'FAST'
            
            idx = int(choice) - 1
            if 0 <= idx < len(SPEED_MODES):
                return list(SPEED_MODES.keys())[idx]
            else:
                print(f"❌ Choisissez entre 1 et {len(SPEED_MODES)}")
        except ValueError:
            print("❌ Entrez un nombre valide")

def main():
    print("="*70)
    print("🚀 TRAITEMENT VIDÉO ULTRA-RAPIDE - VERSION CORRIGÉE")
    print("="*70)
    print("🔧 Cette version corrige le bug de 0 frames traitées")
    
    # Charger le modèle
    model, device = load_epoch30_model()
    if model is None:
        return
    
    # Trouver les vidéos
    videos = find_videos()
    if not videos:
        return
    
    # Sélectionner le mode
    speed_mode = select_speed_mode()
    
    # Sélectionner la vidéo
    print(f"\n📹 Sélection de la vidéo:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    while True:
        try:
            choice = input(f"\nChoisissez une vidéo (1-{len(videos)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                selected_video = videos[idx]
                break
            else:
                print(f"❌ Choisissez entre 1 et {len(videos)}")
        except ValueError:
            print("❌ Entrez un nombre valide")
    
    # Créer le processeur
    processor = FastVideoProcessor(model, device, speed_mode)
    
    # Traiter la vidéo avec la version corrigée
    success = process_video_fixed(processor, selected_video, speed_mode)
    
    if success:
        print(f"\n✅ SUCCÈS! Frames effectivement traitées cette fois!")
        print(f"📁 Résultat dans: {OUTPUT_DIR}/")
        print(f"🎬 Vérifiez votre vidéo de sortie maintenant!")
    else:
        print(f"\n❌ Erreur - aucune frame traitée")

if __name__ == "__main__":
    main()