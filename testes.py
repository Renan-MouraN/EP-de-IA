import polars as pl 

def lercsv(nomecsv:str = "conjuntos_teste/problemAND.csv")->pl.DataFrame:
    df = pl.read_csv(nomecsv, separator=',', has_header=False, encoding='utf8')
    ultima_coluna = df.columns[-2]
    df = df.rename({ultima_coluna: "Resposta"})
    return df
     
df = lercsv("conjuntos_teste/caracteres-limpo.csv")
start_index = df.columns.index("Resposta")
print(df.select(df.columns[start_index:]))
print(df.select(df.columns[:start_index]))