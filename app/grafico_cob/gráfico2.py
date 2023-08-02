import pandas as pd


def grafico_2(path_diretorio_iduser):
    dataframes = []

    # Para cada arquivo, adicione o dataframe à lista
    for item in [(rf'{path_diretorio_iduser}\resultado_cluster.csv', 'Index CP','index_cp'),(rf'{path_diretorio_iduser}\resultado_cluster_2.csv', 'IndexCob (VALOR)','IndexCob_VALOR')]:
        try:
            path, variavel,nome_file = item
            #path=r'C:\GN_Analitycs_WEB_COB\arquivos\resultado_cluster_2.csv'
            data = pd.read_csv(path, delimiter=';')

            data.drop(data[data['Clusters'] == 'TOTAL'].index, inplace=True)
            try:
                data[variavel] = data[variavel].str.rstrip('%')
                try:
                    data[variavel] = data[variavel].astype('float64')
                except:
                    data[variavel] = data[variavel].astype(int)
            except:
                data[variavel] = data[variavel].astype(int)

            data = data[[variavel, 'Clusters']]
            data[variavel] = data[variavel] / 100
            data = data.drop(data[data['Clusters'] == 'Total'].index)

            # Padronize os valores para a coluna 'Clusters'
            data['Clusters'] = data['Clusters'].str.replace('Grupo', 'Grupo ')
            dataframes.append(data)
        except Exception as err:
            pass

        # Concatene os dataframes na lista em um só dataframe
        combined_data = pd.concat(dataframes,axis=1)
        combined_data = combined_data.iloc[:, :-1]
        data.to_csv(rf'{path_diretorio_iduser}\{nome_file}.csv',sep=';',index=False)