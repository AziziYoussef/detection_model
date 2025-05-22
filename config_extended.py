# Configuration étendue avec 30 classes d'objets perdus
config = {
    # Paramètres du modèle
    'num_classes': 30,  # 30 classes au lieu de 10
    
    # Paramètres d'entraînement ajustés pour plus de classes
    'batch_size': 4,  # Garder petit pour éviter les erreurs mémoire
    'learning_rate': 0.003,  # Légèrement réduit pour stabilité
    'num_epochs': 20,  # Plus d'époques pour apprendre toutes les classes
    'image_size': (320, 320),
    'use_mixed_precision': True,
    'max_train_images': 5000,  # Plus d'images pour plus de classes
    'max_val_images': 500,
    
    # Paramètres d'optimisation
    'optimizer': 'sgd',
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'num_workers': 0,
    'pin_memory': False,
    
    # Chemins
    'coco_dir': 'c:/Users/ay855/Documents/detction_model/coco',
    'output_dir': 'output_extended_30',
    
    # 30 classes d'objets perdus courants
    'classes': [
        # === OBJETS PERSONNELS === (8 classes)
        'backpack',      # sac à dos
        'suitcase',      # valise
        'handbag',       # sac à main
        'tie',           # cravate
        'sunglasses',    # lunettes de soleil
        'watch',         # montre (pas dans COCO standard, on utilisera 'clock')
        'hair drier',    # sèche-cheveux
        'toothbrush',    # brosse à dents
        
        # === ÉLECTRONIQUE === (6 classes)
        'cell phone',    # téléphone portable
        'laptop',        # ordinateur portable
        'keyboard',      # clavier
        'mouse',         # souris d'ordinateur
        'remote',        # télécommande
        'tv',            # télévision
        
        # === USTENSILES ET CUISINE === (7 classes)
        'bottle',        # bouteille
        'cup',           # tasse
        'bowl',          # bol
        'knife',         # couteau
        'spoon',         # cuillère
        'fork',          # fourchette
        'wine glass',    # verre à vin
        
        # === OUTILS ET BUREAU === (3 classes)
        'scissors',      # ciseaux
        'book',          # livre
        'clock',         # horloge
        
        # === OBJETS DU QUOTIDIEN === (3 classes)
        'umbrella',      # parapluie
        'vase',          # vase
        'potted plant',  # plante en pot
        
        # === TRANSPORT ET SPORT === (3 classes)
        'bicycle',       # vélo
        'skateboard',    # skateboard
        'sports ball',   # ballon
    ],
    
    # Mapping pour l'affichage en français (optionnel)
    'class_names_fr': {
        'backpack': 'Sac à dos',
        'suitcase': 'Valise',
        'handbag': 'Sac à main',
        'tie': 'Cravate',
        'sunglasses': 'Lunettes de soleil',
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
}