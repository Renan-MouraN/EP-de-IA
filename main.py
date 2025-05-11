from Classes.RedeNeural import Rede_Neural
import polars as pl

def lercsv(nomecsv:str = "conjuntos_teste/problemAND.csv", tamresp = 0)->pl.DataFrame:
    df = pl.read_csv(nomecsv, separator=',', has_header=False, encoding='utf8')
    #caso onde a resposta é composta por apenas uma coluna 
    ultima_coluna = df.columns[-tamresp]
    df = df.rename({ultima_coluna: "Resposta"})
    return df

#Teste com csv limpo 
# tamanhoRespostacsv = 2
# df = lercsv("conjuntos_teste/caracteres-limpo.csv", tamanhoRespostacsv)
# print(df)

tamanhoRespostacsv = 1
df = lercsv(tamresp= tamanhoRespostacsv)
print(df)

tamCamadaInicial = df.shape[1] - tamanhoRespostacsv
tamCamadaOculta = 7
tamCamadaFinal = tamanhoRespostacsv
learnig_rate = 0.8
num_epocas = 5

# Deixamos nesse formato para testar mais camadas ocultas depois 
tamCamadas = [tamCamadaInicial,tamCamadaOculta,tamCamadaFinal]
rede = Rede_Neural(tamanhos = tamCamadas, learning_rate = learnig_rate, epocas = num_epocas, acertoMin= 0.9, data = df)

print(f'Numero de epocas: {num_epocas}, Learning_rate: {learnig_rate}')

rede.runEpoca()