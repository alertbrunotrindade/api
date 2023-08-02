import pandas as pd

def grafico_result_cluster_dispersao(path_diretorio_iduser):
    data = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv',delimiter=';')

    data['%Vendas'] = data['%Vendas'].str.rstrip('%')
    data['%Vendas_graph'] = data['%Vendas'].shift(1).fillna(0).astype(float)

    for i in range(2, data.shape[0]):
        data.at[i, '%Vendas_graph'] = data.at[i - 1, '%Vendas_graph'] + float(data.at[i-1, '%Vendas'])

    data['%Qtde'] = data['%Qtde'].str.rstrip('%')
    data['%Qtde_graph'] = data['%Qtde'].shift(1).fillna(0).astype(float)

    for i in range(2, data.shape[0]):
        data.at[i, '%Qtde_graph'] = data.at[i - 1, '%Qtde_graph'] + float(data.at[i-1, '%Qtde'])

    data['%Qtde_graph'] = data['%Qtde_graph'].round()

    data = data[['%Vendas_graph', '%Qtde_graph']]

    dict_return = {'grafico': ''}
    dict_return['grafico'] = data.to_dict(orient='records')

    return dict_return