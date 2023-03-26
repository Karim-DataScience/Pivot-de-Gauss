import numpy as np

# Génération d'une matrice aléatoire 3x3
A = np.random.rand(3, 3)

# Génération d'un second membre aléatoire 3x1
b = np.random.rand(3, 1)

# Matrice augmentée
M = np.concatenate((A, b), axis=1)

# Affichage de la matrice augmentée
print("Matrice augmentée :")
print(M)
