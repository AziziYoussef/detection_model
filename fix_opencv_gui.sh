# Script pour réparer OpenCV avec support GUI sur Windows
# Exécutez ces commandes dans votre environnement conda 'detection'

echo "🔧 RÉPARATION OPENCV GUI POUR WINDOWS"
echo "====================================="

# Option 1: Réinstaller opencv-python avec conda (recommandé)
echo "Option 1: Réinstallation via conda..."
conda install -c conda-forge opencv -y

# Option 2: Si Option 1 ne marche pas, utiliser pip
echo "Option 2: Réinstallation via pip..."
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python

# Option 3: Version complète avec tous les codecs
echo "Option 3: Version complète..."
pip uninstall opencv-python opencv-python-headless -y  
pip install opencv-contrib-python

# Test de vérification
echo "🧪 Test OpenCV GUI..."
python -c "
import cv2
print('OpenCV version:', cv2.__version__)
try:
    # Test simple
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow('Test', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print('✅ GUI fonctionne!')
except Exception as e:
    print('❌ GUI ne fonctionne pas:', e)
"

echo "🎯 Si le test réussit, vous pouvez utiliser les scripts avec affichage"
echo "🎯 Sinon, utilisez headless_video_detection.py (fonctionne toujours)"