import pandas as pd


def gera_cluster(path_diretorio_iduser,cluster_volume):

    "Clusters"
    opcao_cluster = cluster_volume

    #gravando o cluster selecionado
    data = [opcao_cluster]
    df = pd.DataFrame(data, columns=['opcao_cluster'])
    df.to_csv(rf'{path_diretorio_iduser}\selecao_cluster.csv', index=False)

    num_clusters = int(opcao_cluster)

    #avalia e verifica seusa randon ou xgboost
    df_result_forest = pd.read_csv(rf'{path_diretorio_iduser}\score_randomforest.csv', sep=';')
    df_result_xgb = pd.read_csv(rf'{path_diretorio_iduser}\score_xgboost.csv', sep=';')
    result_forest = df_result_forest['construcao']*0.3 + df_result_forest['validacao']*0.7
    result_xgb = df_result_xgb['construcao']*0.3 + df_result_xgb['validacao']*0.7

    #st.write(result_forest)
    #st.write(result_xgb)

    if result_forest.iloc[0] > result_xgb.iloc[0]:
        original = pd.read_csv(rf'{path_diretorio_iduser}\original_com_proba_random.csv', delimiter=';')
        modelo = [2]
        df = pd.DataFrame(modelo, columns=['modelo'])
        df.to_csv(rf'{path_diretorio_iduser}\modelo_won.csv', sep=';', index=None)

    else:
        original = pd.read_csv(rf'{path_diretorio_iduser}\original_com_proba_xg_boost.csv',
                               delimiter=';')
        modelo = [3]
        df = pd.DataFrame(modelo, columns=['modelo'])
        df.to_csv(rf'{path_diretorio_iduser}\modelo_won.csv', sep=';', index=None)
    #######

    try:
        bins = pd.qcut(original['probabilidade'], q=num_clusters,
                       labels=[f'Grupo{i + 1}' for i in range(num_clusters)][::-1])
        original['Clusters'] = bins
        original.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', sep=';', index=None)

        "converter em endpoints"

        # gera_tabela_cluster()
        original = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', delimiter=';')
        soma_vendas_por_grupo = original.groupby('Clusters')['VENDAS'].sum()
        contagem_por_grupo = original['Clusters'].value_counts()
        resultado = pd.DataFrame({'VENDAS': soma_vendas_por_grupo, 'Contagem': contagem_por_grupo})
        resultado = resultado.rename_axis('Clusters').reset_index()
        categorias = ['Grupo1', 'Grupo2', 'Grupo3', 'Grupo4', 'Grupo5', 'Grupo6', 'Grupo7', 'Grupo8', 'Grupo9',
                      'Grupo10']
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
        resultado['Index'] = resultado['VENDAS'] / resultado['Contagem'] / (total_soma_vendas / total_contagem)
        resultado['Index'] = resultado['Index'] * 100
        resultado['Index'] = resultado['Index'].round().astype(int).astype(str) + '%'
        resultado = resultado.sort_values('Clusters')

        total_sales = pd.DataFrame(
            {'Clusters': ['Total'], 'VENDAS': [total_vendas], 'Contagem': [total_contagem], '%Vendas': ['100%'],
             '%Qtde': ['100%'], 'Index': ['100%']})
        resultado = resultado.append(total_sales, ignore_index=True)
        overall_conversion = total_vendas / total_contagem
        resultado.loc[len(resultado) - 1, '% Conversão'] = '{:.2%}'.format(overall_conversion)

        resultado.to_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv', sep=';')

        #grafico_clus()
        #grafico_2()
        #grafico_waterfall()

    except Exception as e:
        print("Por favor diminua a quantidade de cluster, o modelo não suporte o volume solicitado")
        print(e)