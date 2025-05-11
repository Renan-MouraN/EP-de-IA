# 1 - Definição da Estrutura e Arquitetura da Rede

1.1 - Componentes básicos  
- **Camada de Entrada:** Definir a quantidade de neurônios de entrada  
- **Camada Oculta:** 1 camada oculta; definir o número de neurônios dela  
- **Camada de Saída:** Definir o número dessa camada  

1.2 - Representação dos dados  
- Pré-processamento  
- Testes rápidos

# 2 - Inicialização dos pesos e parâmetros

2.1 - Pesos e Bias  
- **Pesos iniciais:** pequenos e aleatórios  
- **Bias:** 1

2.2 - Parâmetros  
- Taxa de aprendizado  
- Épocas  
- Número de neurônios na camada oculta  
- Critério de parada

# 3 - Implementação do Algoritmo de Backpropagation

3.1 - Passo a Passo  
- **Cálculo da ativação da matriz oculta:** somatório (entrada * pesos) + bias  
- **Cálculo da saída:** mesma lógica da camada oculta  
- Usar função de erro  
- Fazer o backward propagation (ver detalhes posteriormente)

3.2 - Treinamento  
- Loop que percorre todas as amostras  
- Gravação do erro em cada iteração
