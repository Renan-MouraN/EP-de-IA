from Classes.Neuronio import Neuronio

class Camada:
    # boolCamSaida indica que a camada é a de saída
    def __init__(self, father, tam, last_size, saidaEsperada = []):
        self.rede = father
        self.tam = tam
        self.last_size = last_size
        self.saidaEsperada = saidaEsperada
        self.neuronios: list[Neuronio] = []
        self.gradientes: list[Neuronio.gradiente] = []

        for i in range(tam):
            self.neuronios.append(Neuronio(last_size))

    def calcula_saidas(self, entradas:list, idx)->list:
        listaux = []
        if len(self.saidaEsperada) == 0:
            for i, neuronio in enumerate(self.neuronios):
                #neuronio.gradiente = somatorio de todos (gradiente k * peso do meu neuronio pra k * derivada da função de ativacao do meu neuronio no y_in) pra cada neuronio k da camada posterior
                '''
                neuronio.gradiente = 1
                for neuronio da camanda da atual:
                    for neuronio da camada da frente:
                        neuronio.gradiente =* peso do neuronio atual pro neuronio da frente * sigmoid_derivative(neuronio.y_in)
                '''
                listaux.append(neuronio.saida(entradas))
        else: 
            for i, neuronio in enumerate(self.neuronios):

                if self.tam == 1 : saida = self.saidaEsperada[idx] 
                else: saida = self.saidaEsperada[idx][i]

                # Já muda a saída para binário (1, -1)
                resposta = neuronio.saidaFinal(entradas)
                if resposta != saida:
                    #neuronio.gradiente = (resposta - self.saidaEsperada[i]) * sigmoid_derivative(neuronio.y_in)
                    #self.gradientes.append(neuronio.gradiente)
                    # neuronio.muda_peso()
                    neuronio.gradiente = resposta - saida
                    print(neuronio.gradiente)
                    
                    self.rede.mudaPesosCamadasAnteriores()
                    print("Saida Esperada: ", saida)
                listaux.append(neuronio.saidaFinal(entradas))
            #Checagem
        return listaux 
    
    def trocaPesos(self):
        print("TROCA PESOS")
        for neuronio in self.neuronios:
            neuronio.muda_peso(1)