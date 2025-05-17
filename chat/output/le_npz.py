import numpy as np

# Carrega o array
arr = np.load('chat/output/dataset_best_weights_initial_W1.npy')

# Imprime informações gerais
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)

# Imprime todo o conteúdo (se não for muito grande)
print("Conteúdo inicial:\n", arr)

arr2 = np.load('chat/output/dataset_best_weights_final_W1.npy')

# Imprime informações gerais
print("Shape:", arr2.shape)
print("Dtype:", arr2.dtype)

# Imprime todo o conteúdo (se não for muito grande)
print("Conteúdo final:\n", arr2)