import numpy as np

# Função sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


x = np.array([0.2, 1.5, -2.0])  # exemplo de saída bruta
prob = sigmoid(x)   
print(prob)            
prob[0] = 0.5
print(prob)
 # converte para valores entre 0 e 1
saida_binaria = (prob >= 0.5).astype(int) 
print(saida_binaria)
print(type(saida_binaria))
print(saida_binaria[0])

#funcao saida = ReLU
# def saida(self,entradas):
#     return max(0,self.funcao_ativacao(entradas))



entradas = [0.2, 1.5, -2.0]
entradas = np.array(entradas)
sigmoid1 =  1 / (1 + np.exp(-entradas))
print(sigmoid1)
sigmoid1[0] = 0.5
print(sigmoid1)
# Recebe 1 caso seja >= 0.5
resp = np.where(sigmoid1 < 0.5, 0, 1)
print("RESPOST",resp)



sigmoi =  1 / (1 + np.exp(-1.5))
# converte para valores entre 0 e 1
resp = 1 if sigmoi >= 0.5 else  0
print(resp)

print(sigmoi)