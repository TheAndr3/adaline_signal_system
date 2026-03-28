import numpy as np

def treinar_adaline(entradas_x, desejado_d, taxa_aprendizado, max_epocas, tolerancia):
    num_amostras = entradas_x.shape[0]
    
    bias = -np.ones((num_amostras, 1))
    # Adiciona o -1 para cada amostra, criando uma nova matriz com o Bias na primeira coluna e as entradas_x nas colunas seguintes
    x_com_bias = np.hstack((bias, entradas_x))
    
    pesos = np.random.rand(3)
    pesos_iniciais = np.copy(pesos) 
    
    historico_eqm = []
    
    for epoca in range(max_epocas):
        erro_quadratico_soma = 0
        
        for i in range(num_amostras):
            amostra_atual = x_com_bias[i]
            d_atual = desejado_d[i]

            u = np.dot(pesos, amostra_atual)
            erro = d_atual - u
            pesos = pesos + taxa_aprendizado * erro * amostra_atual   #Delta Rule
            erro_quadratico_soma += (erro ** 2)
            
        eqm = erro_quadratico_soma / num_amostras
        historico_eqm.append(eqm)
        
        if eqm <= tolerancia:
            break
            
    numero_de_epocas_que_rodou = epoca + 1
    return pesos_iniciais, pesos, numero_de_epocas_que_rodou, historico_eqm
