import numpy as np

# Matrice augmentée du système d'équations linéaires
A = np.array([[1, 2, 3, 4],
              [2, 3, 4, 1],
              [3, 4, 1, 2],
              [4, 1, 2, 3]], dtype=float)

# Nombre d'inconnues
n = A.shape[0]

# Étape 1 : Élimination de Gauss
print("Étape 1 : Élimination de Gauss")
print("Matrice initiale :")
print(A)

for i in range(n):
    pivot = A[i, i]
    A[i, :] = A[i, :] / pivot
    print("Pivot : ", pivot)
    print("L{} <- L{} / {}".format(i + 1, i + 1, pivot))
    print("Résultat :")
    print(A)

    for j in range(i + 1, n):
        pivot2 = A[j, i]
        A[j, :] = A[j, :] - pivot2 * A[i, :]
        print("L{} <- L{} - {} * L{}".format(j + 1, j + 1, pivot2, i + 1))
        print("Résultat :")
        print(A)

# Étape 2 : Réduction de Gauss-Jordan
print("Étape 2 : Réduction de Gauss-Jordan")
for i in range(n - 1, -1, -1):
    for j in range(i - 1, -1, -1):
        pivot3 = A[j, i]
        A[j, :] = A[j, :] - pivot3 * A[i, :]
        print("L{} <- L{} - {} * L{}".format(j + 1, j + 1, pivot3, i + 1))
        print("Résultat :")
        print(A)

# Affichage de la matrice après pivot de Gauss
print("Matrice après pivot de Gauss :")
print(A)

# Vérification de la compatibilité du système
if A[-1, :-1].any() == 0 and A[-1, -1] != 0:
    print("Système incompatible")
else:
    print("Système compatible")

# Vérification de l'homogénéité du système
if A[-1, :-1].all() == 0 and A[-1, -1] == 0:
    print("Système homogène")
else:
    print("Système non homogène")

# Calcul du nombre de solutions
r = np.linalg.matrix_rank(A[:, :-1])
r_aug = np.linalg.matrix_rank(A)
n_variables = A.shape[1] - 1
n_ecart = n_variables - r

if r == r_aug:
    if n_ecart == 0:
        print("Le système admet une unique solution")
    else:
        print("Le système admet une infinité de solutions")
        
        # Identification des variables libres et liées
        var_libres = []
        var_liees = []
        for i in range(n_variables):
            if A[i, :-1].any() == 0:
                var_libres.append(i)
            else:
                var_liees.append(i)
        print("Variables libres :", var_libres)
        print("Variables liées :", var_liees)
else:
    print("Le système n'admet pas de solution")
