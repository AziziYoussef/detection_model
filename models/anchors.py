import torch

# Génération des boîtes d'ancrage (anchors)
class AnchorGenerator:
    def __init__(self, sizes, aspect_ratios, strides):
        """
        Initialise le générateur d'anchors
        
        Args:
            sizes (list): Tailles de base des anchors pour chaque niveau
            aspect_ratios (list): Ratios hauteur/largeur
            strides (list): Pas pour chaque niveau de la pyramide
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        
    def generate_anchors(self, image_size):
        """
        Génère des boîtes d'ancrage pour une image de taille donnée
        
        Args:
            image_size (tuple): Taille de l'image (height, width)
            
        Returns:
            torch.Tensor: Anchors de forme [N, 4] où N est le nombre total d'anchors
                          et 4 correspond aux coordonnées [x1, y1, x2, y2] normalisées
        """
        anchors = []
        
        # Pour chaque niveau de la pyramide
        for level, (size, stride) in enumerate(zip(self.sizes, self.strides)):
            # Calculer le nombre de cellules dans la grille
            grid_height = image_size[0] // stride
            grid_width = image_size[1] // stride
            
            # Coordonnées du centre de chaque cellule
            centers_x = torch.arange(0, grid_width) * stride + stride // 2
            centers_y = torch.arange(0, grid_height) * stride + stride // 2
            
            # Créer une grille de centres
            grid_x, grid_y = torch.meshgrid(centers_x, centers_y, indexing='ij')
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)
            
            # Pour chaque centre, créer des anchors avec différentes tailles et ratios
            for ratio in self.aspect_ratios:
                for scale in [0.5, 1.0, 2.0]:
                    w = size * scale * torch.sqrt(torch.tensor(ratio))
                    h = size * scale / torch.sqrt(torch.tensor(ratio))
                    
                    # Créer les anchors [x1, y1, x2, y2]
                    x1 = grid_x - w / 2
                    y1 = grid_y - h / 2
                    x2 = grid_x + w / 2
                    y2 = grid_y + h / 2
                    
                    # Empiler les coordonnées
                    level_anchors = torch.stack([x1, y1, x2, y2], dim=1)
                    anchors.append(level_anchors)
        
        # Concaténer tous les anchors
        anchors = torch.cat(anchors, dim=0)
        
        # Normaliser les anchors par rapport à la taille de l'image
        anchors[:, [0, 2]] /= image_size[1]
        anchors[:, [1, 3]] /= image_size[0]
        
        # Limiter les anchors à [0, 1]
        anchors = torch.clamp(anchors, 0, 1)
        
        return anchors