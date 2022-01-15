# TP-GPU-CUDA
## Speed CPU vs GPU
On a testé des multiplication matricielle N x N sur CPU et GPU (`speed_test.cu`). 

Pour N = 2000, le GPU a été 1100x plus rapide, pour N = 10000, le GPU a pris 4 secondes pour faire l'operation, dieu sait combien ça aurait pris pour un CPU ! 

Notre kernel  ``` __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n_l, int n_c) ``` utilise des thread blocks de 32 x 32, (limité à 1024 par thread block), comme l'illuste la figure suivante, `BLOCK_WIDTH = 32` pour notre cas. Chaque thread par bloc fera le produit d'une ligne de la matrice M1 et d'une colonne de M2.

![image](https://user-images.githubusercontent.com/37119086/149612529-3426a2ab-b193-4301-8795-dbd9b32a770b.png) 
![image](https://user-images.githubusercontent.com/37119086/149612487-e75d189c-9560-4055-b79f-4d1eca162b25.png)

## Inference
Les poids du modèle entrainé sont chargé dans `weights.h`.
Toute les couches sont parallelisable. 

Le programme à exécuter est `
```cpp
struct bbox_t
```
