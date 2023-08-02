import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
from joblib import load

def prever_vendas(base_construcao,caminho_arquivo,path_diretorio_iduser):
    # Ocultar apos testes

    opcao_escolhida = pd.read_csv(rf'{path_diretorio_iduser}\selecao.csv',delimiter=';')
    df_Layout = pd.read_csv(base_construcao,delimiter=';',encoding='latin1')
    df = pd.read_csv(rf'{path_diretorio_iduser}\base_original_importada.csv',delimiter=';',encoding='latin1')
    qtd_total = len(df)

    flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
    flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
    retornar_0_em_percentual = flag_resposta_0 / qtd_total
    retornar_1_em_percentual = flag_resposta_1 / qtd_total

    original = pd.read_csv(caminho_arquivo, delimiter=';', encoding='latin1')
    original['ID_UNICO'] = range(1, len(original) + 1)
    original_list = original.columns.tolist()
    ########REMOVER
    df_list = df_Layout.columns.tolist()
    df_list_final = []
    for item in original_list:
        if item not in df_list:
            df_list_final.append(item)
    #######
    original = original[df_list]
    original['ID_UNICO'] = range(1, len(original) + 1)
    print("tamanho é igual ao total de linhas?:",len(original) == original['ID_UNICO'].max())

    #original.to_csv(r'C:\Users\bruno.trindade\Documents\Analitics\arquivo_novo_do_modelo_treinado_id_unico.csv',
    #                sep=';', index=None)

    for col in original.columns:
        if original[col].dtype == "object" or original[col].dtype == 'O':
            original[col] = original[col].astype(str)
            try:
                le = LabelEncoder()
                original.loc[:, col] = le.fit_transform(original[col])
                print(col)
            except Exception as err:
                print(err)
                pass

    X = original.drop(columns=["VENDAS", "ID_UNICO"], axis=1)
    y = original["VENDAS"]

    df = pd.read_csv(rf'{path_diretorio_iduser}\modelo_won.csv', sep=';')
    if df["modelo"].iloc[0] == 3:

        model = xgb.XGBClassifier()
        model.load_model(rf'{path_diretorio_iduser}\modelo.xgb')
    else :
        # Especifique o caminho completo para o arquivo do modelo
        caminho_arquivo_modelo = rf'{path_diretorio_iduser}\modelo_random_forest.joblib'

        # Carregue os parâmetros e atributos do modelo
        model = load(caminho_arquivo_modelo)

    y_probs = model.predict_proba(X.fillna(0))
    original["probabilidade"] = [p[1] for p in y_probs]

    if opcao_escolhida['opcao'][0] =='50%50':
        fator_ajuste = np.array([retornar_0_em_percentual/0.5, retornar_1_em_percentual/0.5])
        y_pred_ajustado = y_probs * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='70%30':
        fator_ajuste = np.array([retornar_0_em_percentual/0.7, retornar_1_em_percentual/0.3])
        y_pred_ajustado = y_probs * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='80%20':
        fator_ajuste = np.array([retornar_0_em_percentual/0.8, retornar_1_em_percentual/0.2])
        y_pred_ajustado = y_probs * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='Sem balanceamento':
        fator_ajuste = np.array([retornar_0_em_percentual / 0.99, retornar_1_em_percentual / 0.01])
        y_pred_ajustado = y_probs * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]


    original.to_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv',sep=';',index=None)

    texto1 = 'Taxa de Conversão da Base de Construção do Modelo: {:.2f}%'.format(retornar_1_em_percentual * 100)

    proba_origiginal = pd.read_csv(rf'{path_diretorio_iduser}\original_com_proba_xg_boost.csv', delimiter=';')
    prob = proba_origiginal['y_pred_ajustado'].mean()
    print(prob)

    proba_modelo_novo = pd.read_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv', delimiter=';')
    prob_novo = proba_modelo_novo['y_pred_ajustado'].mean()

    print(prob_novo)
    prob_text = (prob_novo / prob * 100)-100

    if prob_text>0:
        texto2 = 'Desempenho estimado em comparação com o mailing da Construção: Incremento de {:.2f}%'.format(prob_text)
    elif prob_text == 0:
        texto2 = 'Desempenho estimado em comparação com o mailing da Construção: Mesmo Desempenho {:.2f}%'.format(prob_text)
    else:
        texto2 = 'Desempenho estimado em comparação com o mailing da Construção: Decremento de {:.2f}%'.format(prob_text)

    texto3 = 'Estimativa da Taxa de Conversão da Base importada: {:.2f}%'.format(prob_novo / prob * retornar_1_em_percentual * 100)

    arquivo_clusterizado = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', delimiter=';')
    media_y_pred_por_cluster = arquivo_clusterizado.groupby('Clusters')['y_pred_ajustado'].mean().apply(lambda x: "{:.2%}".format(x))
    media_y_pred_por_cluste_original = media_y_pred_por_cluster
    #st.title("Média do arquivo original:")
    #st.table(media_y_pred_por_cluster.rename("Taxa de Conversao"))


    opcao_cluster = int(pd.read_csv(rf'{path_diretorio_iduser}\selecao_cluster.csv').iloc[0, 0])
    original = pd.read_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv', delimiter=';')
    bins = pd.qcut(original['probabilidade'], q=opcao_cluster,labels=[f'Grupo{i + 1}' for i in range(opcao_cluster)][::-1])
    original['Clusters'] = bins
    original.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv', sep=';', index=None)

    modelo = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv',delimiter=';',encoding='latin1')
    arquivo_clusterizado = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv', delimiter=';')
    media_y_pred_por_cluster = arquivo_clusterizado.groupby('Clusters')['y_pred_ajustado'].mean().apply(lambda x: "{:.2%}".format(x))



    #Recriado a tabela de conversao da base importada
    resul_cluster = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv', delimiter=';')
    resul_cluster.drop(resul_cluster.loc[resul_cluster['Clusters'] == 'Total'].index, inplace=True)
    resul_cluster.set_index('Clusters', inplace=True)

    #alterado a coluna para a conversao da base importada
    media_y_pred_por_cluster_unido = pd.concat([resul_cluster['% Conversão'],media_y_pred_por_cluste_original.rename("Taxa de Conversao Base Modelo"), media_y_pred_por_cluster.rename("Taxa de Conversao Base Importada")], axis=1)

    media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2']=((media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada'].str.replace('%','').astype('float64') / media_y_pred_por_cluster_unido['Taxa de Conversao Base Modelo'].str.replace('%','').astype('float64'))* media_y_pred_por_cluster_unido['% Conversão'].str.replace('%','').astype('float64')).fillna(0)
    media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'] = media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'].round(2)
    media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'] = media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'].apply('{:.2f}%'.format)

    table = media_y_pred_por_cluster_unido[['% Conversão','Taxa de Conversao Base Importada_v2']].rename(columns={'% Conversão': 'Taxa de Conversao Base Modelo', 'Taxa de Conversao Base Importada_v2': 'Taxa de Conversao Base Importada'})
    media_y_pred_por_cluster.rename("y_pred_ajustado")

    df_chart =  media_y_pred_por_cluster_unido[['% Conversão', 'Taxa de Conversao Base Importada_v2']].rename(
        columns={'% Conversão': 'Taxa de Conversao Base Modelo',
                 'Taxa de Conversao Base Importada_v2': 'Taxa de Conversao Base Importada'})


    df_chart = df_chart.rename(columns={"Taxa de Conversao Base Modelo": "Base Modelo"})
    df_chart = df_chart.rename(columns={"Taxa de Conversao Base Importada": "Base Importada"})
    df_chart = df_chart.reset_index()
    df_chart['Base Modelo'] = df_chart['Base Modelo'].str.replace('%','').astype('float64')/100
    df_chart['Base Importada'] = df_chart['Base Importada'].str.replace('%', '').astype('float64') / 100

    df_orig_mean = df_chart[['Clusters', 'Base Modelo']]
    df_novo_mean = df_chart[['Clusters', 'Base Importada']]

    df_merged = pd.merge(df_orig_mean, df_novo_mean, on='Clusters', suffixes=('_original', '_novo'))

    df_melted = pd.melt(df_merged, id_vars='Clusters', var_name='Arquivo', value_name='Média do y_pred_ajustado')

    df_sem_coluna = modelo.drop(['probabilidade','y_pred_ajustado'], axis=1)

    # ADICIONANDO AS COLUNAS FALTANTES DO ARQUIVO ORIGINAL
    original = pd.read_csv(caminho_arquivo, delimiter=';', encoding='latin1')
    for col in df_list_final:
        try:
            df_sem_coluna[col] = original[col]
        except:
            pass

    df_sem_coluna.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente.csv',sep=';',index=False)
    dict_return = {'tabela':table.to_dict(orient='records'),'grafico': '','texto':{'texto1':texto1,'texto2':texto2,'texto3':texto3}}
    dict_return['grafico'] = df_melted.to_dict(orient='records')

    return dict_return

def rodar_novo_arquivo(path_novo_arquivo,path_diretorio_iduser):
    # Ocultar apos testes
    #path_diretorio_iduser = r"C:\Users\planejamento\PycharmProjects\pythonProject1\arquivos\3/"
    "Carregue o novo arquivo"
    #path_novo_arquivo=r"C:\Users\planejamento\PycharmProjects\pythonProject1\arquivos\3\BaseHeadset_02.CSV"

    arquivo = path_novo_arquivo
    if arquivo is not None:
        df = pd.read_csv(arquivo, delimiter=';',encoding='latin1')
        df.columns = df.columns.str.upper()
        "Arquivo carregado com sucesso!"
        df.to_csv(rf'{path_diretorio_iduser}\base_importada_2.csv', sep=';', index=None)
        # try:
        base_construcao = pd.read_csv(rf'{path_diretorio_iduser}\base_construcao_usar.csv', delimiter=';')

        dict_prever = prever_vendas(base_construcao=base_construcao['base_construcao_usar'][0],caminho_arquivo=rf'{path_diretorio_iduser}\base_importada_2.csv',path_diretorio_iduser=path_diretorio_iduser)

        caminho_arquivo = rf'{path_diretorio_iduser}\resumo_arquivo_segundo.txt'

        # Salvar o dicionário como arquivo de texto
        with open(caminho_arquivo, "w") as arquivo:
            json.dump(dict_prever, arquivo)
        print('salvou')

