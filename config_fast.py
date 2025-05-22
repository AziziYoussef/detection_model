# Configuration optimisée pour la vitesse d'entraînement
config = {
    # Paramètres du modèle
    'num_classes': 10,
    
    # Paramètres d'entraînement optimisés pour vitesse
    'batch_size': 16,  # Augmenter significativement
    'learning_rate': 0.01,  # Plus élevé pour SSD
    'num_epochs': 10,  # Moins d'époques car converge plus vite
    'image_size': (320, 320),  # Taille plus petite pour SSD
    'use_mixed_precision': True,  # Activer pour accélération
    
    # Réduction du dataset pour les tests
    'max_train_images': 5000,  # Limiter à 5000 images pour tester
    'max_val_images': 500,     # Limiter à 500 images pour validation
    
    # Paramètres d'optimisation
    'optimizer': 'sgd',
    'weight_decay': 5e-4,
    'momentum': 0.9,
    
    # Paramètres du data loader optimisés
    'num_workers': 8,  # Plus de workers
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,  # Préchargement
    
    # Chemins des données
    'coco_dir': 'c:/Users/ay855/Documents/detction_model/coco',
    'output_dir': 'output_fast',
    
    # Classes COCO utilisées
    'classes': [
        'backpack', 'suitcase', 'handbag', 'cell phone', 'laptop',
        'book', 'umbrella', 'bottle', 'keyboard', 'remote'
    ]
}