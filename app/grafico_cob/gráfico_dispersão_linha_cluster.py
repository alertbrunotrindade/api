import pandas as pd

def grafico_clus(path_diretorio_iduser):
    for item in [(rf'{path_diretorio_iduser}\resultado_cluster.csv', '%Cliente Pago','%Cliente_Pago','graf_cliente_pago'),
                 (rf'{path_diretorio_iduser}\resultado_cluster_2.csv', '%VALOR Recuperado','%Valor_Pago','graf_valor_pago')]:

        try:
            path, variavel,y,nome_file = item
            data = pd.read_csv(path, delimiter=';')
            #data.drop(data[data['Clusters'] == 'TOTAL'].index, inplace=True)
            #data = pd.read_csv( path,delimiter=';')
            #data.info()
            data['%Vendas'] = data[variavel].str.rstrip('%')
            data[f'{y}_graph'] = data['%Vendas'].shift(1).fillna(0).astype(float)

            for i in range(2, data.shape[0]):
                data.at[i, f'{y}_graph'] = data.at[i - 1, f'{y}_graph'] + float(data.at[i-1, '%Vendas'])

            data['%Qtde'] = data['%Qtde'].str.rstrip('%')
            data['%Qtde_graph'] = data['%Qtde'].shift(1).fillna(0).astype(float)

            for i in range(2, data.shape[0]):
                data.at[i, '%Qtde_graph'] = data.at[i - 1, '%Qtde_graph'] + float(data.at[i-1, '%Qtde'])

            data['%Qtde_graph'] = data['%Qtde_graph'].round()
            data['%Qtde_graph'].iloc[-1] = 100.0

            data = data[[f'{y}_graph', '%Qtde_graph']]
            print(data)
            data.to_csv(rf'{path_diretorio_iduser}\{nome_file}.csv',sep=';',index=False)
            #
        except Exception as err:
            pass