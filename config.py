# Configuration optimisée pour RTX 4060
config = {
    # Paramètres du modèle
    'num_classes': 10,  # Nombre de classes sans compter le fond
    'pretrained_backbone': False,  # Utiliser un backbone pré-entraîné (False pour créer from scratch)
    
    # Paramètres des anchors
    'anchor_sizes': [32, 64, 128, 256, 512],  # Tailles de base pour chaque niveau
    'anchor_aspect_ratios': [0.5, 1.0, 2.0],  # Ratios hauteur/largeur
    'anchor_strides': [4, 8, 16, 32, 32],  # Pas pour chaque niveau
    
    # Paramètres d'entraînement optimisés pour RTX 4060
    'batch_size': 8,  # Taille du batch optimisée pour RTX 4060
    'learning_rate': 0.0001,  # Taux d'apprentissage réduit pour stabilité
    'num_epochs': 30,  # Nombre d'époques total
    'image_size': (384, 384),  # Taille d'image réduite pour vitesse
    'use_mixed_precision': True,  # Activer la précision mixte pour accélération
    
    # Paramètres d'optimisation
    'optimizer': 'adamw',  # Options: 'adam', 'sgd', 'adamw'
    'weight_decay': 1e-4,  # Régularisation L2
    'momentum': 0.9,  # Pour SGD
    'gradient_clipping': 1.0,  # Limite pour le clipping des gradients
    
    # Paramètres de la perte
    'cls_weight': 1.0,  # Poids de la perte de classification
    'reg_weight': 1.0,  # Poids de la perte de régression
    'pos_iou_thresh': 0.5,  # Seuil IoU pour les anchors positifs
    'neg_iou_thresh': 0.4,  # Seuil IoU pour les anchors négatifs
    
    # Paramètres du data loader
    'num_workers': 6,  # Nombre de workers pour le chargement des données
    'pin_memory': True,  # Améliore les transferts CPU->GPU
    'persistent_workers': True,  # Garde les workers actifs entre les époques
    
    # Paramètres d'augmentation des données
    'data_augmentation': {
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 15,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
        'translate': (0.1, 0.1),
        'scale': (0.9, 1.1)
    },
    
    # Paramètres de sauvegarde et reprise
    'save_every': 200,  # Sauvegarder tous les N lots
    'checkpoint_dir': 'checkpoints',  # Dossier pour les points de contrôle intermédiaires
    
    # Chemins des données
    'coco_dir': 'c:/Users/ay855/Documents/detction_model/coco',  # Chemin du dataset COCO
    'output_dir': 'output',  # Dossier pour sauvegarder les résultats
    
    # Classes COCO utilisées
    'classes': [
        'backpack',   # sac à dos
        'suitcase',   # valise
        'handbag',    # sac à main
        'cell phone', # téléphone portable
        'laptop',     # ordinateur portable
        'book',       # livre
        'umbrella',   # parapluie
        'bottle',     # bouteille
        'keyboard',   # clavier
        'remote'      # télécommande
    ],
    
    # Paramètres de test et d'inférence
    'confidence_threshold': 0.3,  # Seuil de confiance minimum pour les détections
    'nms_threshold': 0.5,  # Seuil pour la suppression non-maximale
    'test_batch_size': 4,  # Taille du batch pour l'inférence
    'test_image_dir': 'test_images',  # Dossier pour les images de test
    
    # Paramètres de debug
    'debug_mode': False,  # Activer/désactiver le mode debug
    'verbose': True,  # Afficher des informations détaillées
}