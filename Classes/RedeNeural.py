from Classes.Camada import Camada
import polars as pl

class Rede_Neural:

    def __init__(self, tamanhos:list, learning_rate: int, epocas: int, acertoMin:float, data: pl.DataFrame):
        self.txAprendizado = learning_rate
        self.epocas = epocas
        self.acertoMin = acertoMin

        start_index = data.columns.index("Resposta")
        self.entries = data.select(data.columns[:start_index])
        self.respostaesperada = data.select(data.columns[start_index:])
        
        self.camadas_neuronios: list[Camada] = []

        # Inicializa as Camadas com os tamanhos dados e específica a última camada pela passagem de parâmetro a mais
        for i in range(1,len(tamanhos)):
            # Última camada
            if i == len(tamanhos)-1:
                self.camadas_neuronios.append(Camada(self, tam=tamanhos[i], last_size=tamanhos[i-1],saidaEsperada = self.respostaesperada.to_numpy()))
            else: 
                self.camadas_neuronios.append(Camada(self, tam=tamanhos[i], last_size=tamanhos[i-1]))

    def mudaPesosCamadasAnteriores(self):
        # Muda o peso de todas as camadas Ocultas
        for camada in self.camadas_neuronios[:-1]:
            camada.trocaPesos()

    def runEpoca(self):
        for i in range(self.epocas):
            print("\nRodando Época ", i, "\n")
            trocapesos = self.feedfoward()

    def feedfoward(self):

        listaresult = []

        for idx, row in enumerate(self.entries.iter_rows()):
            entrada = row
            print("\nEntrada da Rede", entrada)
            for camada in self.camadas_neuronios:
                entrada = camada.calcula_saidas(entrada,idx)
                print("Saída da camada: ", entrada)

            # Saída da última camada
            listaresult.append(entrada)

        print("\nResultado da Época: ", listaresult)
        self.calculaerro(listaresult)
        return

    # Compara as saídas  da época com a saída esperada
    def calculaerro(self, resultadoGeral:list):
        counterro = 0
        for i, resultlinha in enumerate(resultadoGeral):
            df_linha = list(self.respostaesperada.row(i))
            if df_linha != resultlinha:
                counterro += 1
                calculoerro = 0 
                for i in range(len(df_linha)):
                    calculoerro += df_linha[i] - resultlinha[i]
                print(calculoerro )
                print(f"Diferente na linha {i}: saidaEsperada={df_linha}, saida={resultlinha}")

        print("Houveram ", counterro , "erros em um total de " , len(resultadoGeral))
        return True
    

    

            