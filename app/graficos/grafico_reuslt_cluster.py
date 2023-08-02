import streamlit as st
import pandas as pd
import altair as alt

def grafico_result_cluster(path_diretorio_iduser):

    data = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv',delimiter=';')
    data1 = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv', delimiter=';')
    data_table = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv', delimiter=';')

    data['Index'] = data['Index'].str.rstrip('%').astype(int)

    data = data[['Index', 'Clusters']]
    data['Index'] = data['Index'] / 100
    data = data.drop(data[data['Clusters'] == 'Total'].index)

    data1['%Vendas'] = data1['%Vendas'].str.rstrip('%')
    data1['%Vendas_graph'] = data1['%Vendas'].shift(1).fillna(0).astype(float)

    for i in range(2, data1.shape[0]):
        data1.at[i, '%Vendas_graph'] = data1.at[i - 1, '%Vendas_graph'] + float(data1.at[i - 1, '%Vendas'])

    data1['%Qtde'] = data1['%Qtde'].str.rstrip('%')
    data1['%Qtde_graph'] = data1['%Qtde'].shift(1).fillna(0).astype(float)

    for i in range(2, data1.shape[0]):
        data1.at[i, '%Qtde_graph'] = data1.at[i - 1, '%Qtde_graph'] + float(data1.at[i - 1, '%Qtde'])

    data1['%Qtde_graph'] = data1['%Qtde_graph'].round()

    data1 = data1[['%Vendas_graph', '%Qtde_graph']]

    data_table = data_table[["Clusters","VENDAS","Contagem","% Conversão","Index","%Vendas","%Qtde"]]

    #Gera graficos e tabelas
    dict_return = {'grafico1': '', 'grafico2': '', 'tabela': ''}
    dict_return['grafico1'] = data.to_dict(orient='records')
    dict_return['grafico2'] = data1.to_dict(orient='records')
    dict_return['tabela'] = data_table.to_dict(orient='records')
    #print(dict_return['tabela'])
    #print( dict_return)
    return dict_return
