import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gera_tabela_cluster(path_diretorio_iduser):
    try:
        df = pd.read_csv(rf'{path_diretorio_iduser}\opt_outliers.csv', sep=';')
        resposta = df['opcao'].iloc[0]
    except:
        resposta = 'SIM'

    original = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', delimiter=';')
    #original['VALOR'].sum()

    #Definicao de Outliers
    q1 = original['VALOR'].quantile(0.25)
    q3 = original['VALOR'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr

    q99 = original['VALOR'].quantile(0.999)

    if q99 >  upper_bound:
        upper_bound_temp = q99
    else:
        upper_bound_temp =  upper_bound

    original['flag_1.5_iqr'] = np.where(original['VALOR']<upper_bound_temp,1,0)
    #original.to_csv(r'C:\Users\bruno.trindade\Documents\Analitics\externo\analise\ARQUIVO_clusterizado.csv', sep=';')
    # Definicao de Outliers

    if resposta == 'SIM':
        soma_VALOR_por_grupo = original.groupby('Clusters')['VALOR'].sum()
    else:
        original = original[original['flag_1.5_iqr']==1]
        soma_VALOR_por_grupo = original.groupby('Clusters')['VALOR'].sum()

    soma_vendas_por_grupo = original.groupby('Clusters')['VENDAS'].sum()
    soma_valor_recuperado = original[original['VENDAS']==1].groupby('Clusters')['VALOR'].sum()
    contagem_por_grupo = original['Clusters'].value_counts()

    resultado = pd.DataFrame({'VALOR Recuperado':soma_valor_recuperado,'VENDAS': soma_vendas_por_grupo, 'Contagem': contagem_por_grupo,'VALOR': soma_VALOR_por_grupo})
    resultado = resultado.rename_axis('Clusters').reset_index()
    categorias = ['Grupo1', 'Grupo2', 'Grupo3', 'Grupo4', 'Grupo5', 'Grupo6', 'Grupo7', 'Grupo8', 'Grupo9', 'Grupo10']
    resultado['Clusters'] = pd.Categorical(resultado['Clusters'], categories=categorias, ordered=True)
    resultado['% Conversão'] = resultado['VENDAS'] / resultado['Contagem'] * 100
    resultado['% Conversão'] = (resultado['VENDAS'] / resultado['Contagem'] * 100).round(2).astype(str) + '%'
    total_vendas = resultado['VENDAS'].sum()
    resultado['%Vendas'] = resultado['VENDAS'] / total_vendas * 100
    resultado['%Vendas'] = resultado['%Vendas'].round(2).astype(str) + '%'
    resultado['%Qtde'] = resultado['Contagem'] / resultado['Contagem'].sum() * 100
    resultado['%Qtde'] = resultado['%Qtde'].round(2).astype(str) + '%'

    total_soma_vendas = resultado['VENDAS'].sum()
    total_contagem = resultado['Contagem'].sum()
    total_VALOR = resultado['VALOR'].sum()
    total_VALOR_Recuperado = resultado['VALOR Recuperado'].sum()
    resultado['Index'] = resultado['VENDAS'] / resultado['Contagem'] / (total_soma_vendas / total_contagem)
    resultado['Index'] = resultado['Index'] * 100
    resultado['Index'] = resultado['Index'].round().astype(int).astype(str) + '%'
    resultado = resultado.sort_values('Clusters')

    resultado2 =  resultado.copy()
    total_sales = pd.DataFrame(
        {'Clusters': ['Total'], 'VALOR Recuperado':[total_VALOR_Recuperado],'VENDAS': [total_vendas], 'Contagem': [total_contagem],'VALOR':[total_VALOR], '%Vendas': ['100%'],
         '%Qtde': ['100%'], 'Index': ['100%']})

    resultado = resultado.append(total_sales, ignore_index=True)

    overall_conversion = total_vendas / total_contagem

    resultado.loc[len(resultado) - 1, '% Conversão'] = '{:.2%}'.format(overall_conversion)

    resultado.rename(columns={'Index':'Index CP','%Vendas': '%Cliente Pago','VENDAS': 'QTDE Cliente Pago', '% Conversão': '%Sucesso CP'}, inplace=True)
    #resultado.info()

    resultado2['VALOR em Aberto'] = resultado2['VALOR']

    resultado2['%VALOR em Aberto'] = resultado2['VALOR'] / resultado2['VALOR'].sum()

    resultado2['%VALOR Recuperado'] = resultado2['VALOR Recuperado'] / resultado2['VALOR Recuperado'].sum()

    resultadoTotal = resultado2.iloc[0]
    resultadoTotal['Clusters'] = 'TOTAL'
    resultado2 = resultado2.append(resultadoTotal, ignore_index=True)
    for col in ['VALOR Recuperado','VENDAS','Contagem','VALOR em Aberto']:
        resultado2[col][resultado2['Clusters']=='TOTAL'] = resultado2[col][resultado2['Clusters']!='TOTAL'].sum()

    resultado2['Recuperacao Média'] = resultado2['VALOR Recuperado'] / resultado2['Contagem']
    recuperacao_media_total = resultado2['Recuperacao Média'][resultado2['Clusters']=='TOTAL'].reset_index(drop=True)[0]
    resultado2['IndexCob (VALOR)'] = resultado2['Recuperacao Média'] / recuperacao_media_total
    resultado2['IndexCob (VALOR)'] = resultado2['IndexCob (VALOR)'].apply(lambda x: '{:.2f}%'.format(x * 100))

    resultado2['%VALOR em Aberto'][resultado2['Clusters'] == 'TOTAL'] = resultado2['%VALOR em Aberto'][
        resultado2['Clusters'] != 'TOTAL'].sum()
    resultado2['%VALOR em Aberto'] = resultado2['%VALOR em Aberto'].apply(lambda x: '{:.2f}%'.format(x * 100))

    resultado2['%VALOR Recuperado'][resultado2['Clusters'] == 'TOTAL'] = resultado2['%VALOR Recuperado'][
        resultado2['Clusters'] != 'TOTAL'].sum()
    resultado2['%VALOR Recuperado'] = resultado2['%VALOR Recuperado'].apply(lambda x: '{:.2f}%'.format(x * 100))
    #resultado2 = resultado2[['Clusters','VALOR em Aberto', '%VALOR em Aberto', 'VALOR Recuperado','Recuperacao Média', 'IndexCob (VALOR)']]

    resultado=resultado.drop(labels='VALOR Recuperado',axis=1)
    resultado=resultado.rename(columns={'VALOR': 'VALOR em Aberto'})
    resultado.to_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv',sep=';')
    #resultado2.info()

    #Colunas permitidas
    resultado2 = resultado2[['Clusters','VALOR em Aberto', '%VALOR em Aberto',
       'VALOR Recuperado', '%VALOR Recuperado', 'Recuperacao Média',
       'IndexCob (VALOR)','%Qtde']]
    resultado2.to_csv(rf'{path_diretorio_iduser}\resultado_cluster_2.csv', sep=';')

    #st.write(resultado)
    #st.write(resultado2)