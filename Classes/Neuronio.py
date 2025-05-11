import random
import numpy as np

class Neuronio:
    def __init__(self,n_entradas):
        self.pesos = [random.uniform(-1,1) for i in range(n_entradas)]
        self.bias = 1
        self.gradiente = 0
        self.y_in = 0

    #mudar dps baseado em erro
    def muda_bias(self):
        self.bias += random.uniform(-10,10)

    def muda_peso(self, numero):
        self.pesos = [max(-1, min(peso + random.uniform(-numero,numero),1)) for peso in self.pesos]

    def funcao_ativacao(self,entradas):
        soma = 0 
        for i in range(len(entradas)):
            soma += self.pesos[i] * entradas[i]
        soma += self.bias
        return soma
    
    #funcao saida = ReLU
    def saida(self,entradas):
        return max(0,self.funcao_ativacao(entradas))
    
    #funcao saida = SIGMOID
    def saidaFinal(self,entradas):
        self.y_in = self.funcao_ativacao(entradas)
        sigmoid =  1 / (1 + np.exp(-self.y_in))
        # converte para valores entre -1 e 1
        resp = 1 if sigmoid >= 0.5 else  -1
        return resp