# TP-GPU-CUDA HAMAM_HAKIMI
## Speed CPU vs GPU
On a testé des multiplication matricielle N x N sur CPU et GPU (`speed_test.cu`). 

Pour N = 2000, le GPU a été 1100x plus rapide, pour N = 10000, le GPU a pris 4 secondes pour faire l'operation, dieu sait combien ça aurait pris pour un CPU ! 

Notre kernel  ``` __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n_l, int n_c) ``` utilise des thread blocks de 32 x 32, (limité à 1024 par thread block), comme l'illuste la figure suivante.

BLOCK_WIDTH = 32` pour notre cas. Chaque thread par bloc fera le produit d'une ligne de la matrice M1 et d'une colonne de M2.

![image](https://user-images.githubusercontent.com/37119086/149612529-3426a2ab-b193-4301-8795-dbd9b32a770b.png) 
![image](https://user-images.githubusercontent.com/37119086/149612487-e75d189c-9560-4055-b79f-4d1eca162b25.png)

Le nombre de bloc par grid doit être au minimum (N/32 + 1, N/32 + 1), pour pouvoir balayer toutes les lignes et colonnes.
## Inference
Les poids du modèle entrainé sont chargé dans `weights.h`.
Toute les couches sont parallelisable. 

Le programme à exécuter est `LeNet5_CUDA.cu`. On fournit un index, et classifie l'image à cet index. Dans l'exemple qui suit, l'image à l'index 3 (4ème image du train set `train_x`) est un 1.

```
############### Predictions ############
Enter a number between 0 and 59999 : 3
                                                        
################ Conv1 ########################

################ averagePool2D ########################

################ Conv1 ########################

################ averagePool2D ########################

################ Dense ########################

################ Dense ########################

################ Dense ########################
label 0 with percentage : 0.000000
label 1 with percentage : 0.999946
label 2 with percentage : 0.000001
label 3 with percentage : 0.000000
label 4 with percentage : 0.000041
label 5 with percentage : 0.000000
label 6 with percentage : 0.000000
label 7 with percentage : 0.000002
label 8 with percentage : 0.000010
label 9 with percentage : 0.000000

################# Answer ###################

Model predicted 1 with probability 0.999946.

#############################################
```
