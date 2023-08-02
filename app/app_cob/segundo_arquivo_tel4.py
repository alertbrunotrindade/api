import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Adiciona o logotipo ao sidebar
import base64

def analise_segundoarquivo(path_diretorio_iduser, file_path,data_ref):


    def prever_vendas(path_diretorio_iduser,base_construcao, caminho_arquivo):

        # base_construcao = pd.read_csv(r'C:\GN_Analitycs_WEB_COB\arquivos\base_construcao_usar.csv', delimiter=';')
        # base_construcao = base_construcao['base_construcao_usar'][0]
        # caminho_arquivo =r'C:\GN_Analitycs_WEB_COB\arquivos\base_importada_2.csv'

        def grafic(df):
            # Criar gráfico de barras usando Altair
            df['% Recuperado'] = df['% Recuperado'].apply(lambda x: x.split('%')[0])
            df['% Recuperado'] = df['% Recuperado'].astype('float64')
            df_graf = df.loc[~df['Clusters'].str.contains('TOTAL')]

            # Mostrar gráfico na página do Streamlit
            return df_graf

        def tableCobAntes(resposta,path_diretorio_iduser):
            df_table_cob = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', delimiter=';',
                                       encoding='latin1')
            # recebe a resposta para usar Outliers

            # Definicao de Outliers
            q1 = df_table_cob['VALOR'].quantile(0.25)
            q3 = df_table_cob['VALOR'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr

            q99 = df_table_cob['VALOR'].quantile(0.999)

            if q99 > upper_bound:
                upper_bound_temp = q99
            else:
                upper_bound_temp = upper_bound

            df_table_cob['flag_1.5_iqr'] = np.where(df_table_cob['VALOR'] < upper_bound_temp, 1, 0)
            # Definicao de Outliers
            if resposta == 'SIM':
                pass
            else:
                df_table_cob = df_table_cob[df_table_cob['flag_1.5_iqr'] == 1]

            df_media_cob = df_table_cob.groupby('Clusters')['y_pred_ajustado'].mean().reset_index()
            df_media_cob.columns = ['Clusters', 'media_agrupamento']

            df_final = pd.merge(df_table_cob, df_media_cob, on='Clusters')
            df_final['Deflator'] = df_final['y_pred_ajustado'] / df_final['media_agrupamento']

            # df_final['VALOR'] = 1000
            df_final['Recuperado'] = df_final['Deflator'] * df_final['media_agrupamento'] * df_final['VALOR']
            df_final.info()
            # Agrupamento
            df_final = df_final.groupby(['Clusters']).agg(
                {'VENDAS': ['count'], 'VALOR': ['sum'], 'Recuperado': ['sum']}).reset_index(drop=False)
            df_final.columns = ['Clusters', 'Qtde', 'VALOR', 'Recuperado']
            df_final['Recuperado'] = df_final['Recuperado'].astype(int)

            # adicionando totais
            resultadoTotal = df_final.iloc[0]
            resultadoTotal['Clusters'] = 'TOTAL'
            df_final = df_final.append(resultadoTotal, ignore_index=True)

            df_final['VALOR'][df_final['Clusters'] == 'TOTAL'] = df_final['VALOR'][
                df_final['Clusters'] != 'TOTAL'].sum()
            df_final['Recuperado'][df_final['Clusters'] == 'TOTAL'] = df_final['Recuperado'][
                df_final['Clusters'] != 'TOTAL'].sum()

            df_final['% Recuperado'] = df_final['Recuperado'] / df_final['VALOR']
            df_final['% Recuperado'] = df_final['% Recuperado'].apply(lambda x: '{:.2f}%'.format(x * 100))

            df_final['Qtde'][df_final['Clusters'] == 'TOTAL'] = df_final['Qtde'][df_final['Clusters'] != 'TOTAL'].sum()
            df_final['Recuperação Média'] = df_final['Recuperado'] / df_final['Qtde']
            df_final['Recuperação Média'] = df_final['Recuperação Média'].astype(int)
            return df_final

        def tableCobPos(resposta,path_diretorio_iduser):
            df_table_cob = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv', delimiter=';',
                                       encoding='latin1')
            # recebe a resposta para usar Outliers

            # Definicao de Outliers
            q1 = df_table_cob['VALOR'].quantile(0.25)
            q3 = df_table_cob['VALOR'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr

            q99 = df_table_cob['VALOR'].quantile(0.999)

            if q99 > upper_bound:
                upper_bound_temp = q99
            else:
                upper_bound_temp = upper_bound

            df_table_cob['flag_1.5_iqr'] = np.where(df_table_cob['VALOR'] < upper_bound_temp, 1, 0)
            # Definicao de Outliers
            if resposta == 'SIM':
                pass
            else:
                df_table_cob = df_table_cob[df_table_cob['flag_1.5_iqr'] == 1]

            df_media_cob = df_table_cob.groupby('Clusters')['y_pred_ajustado'].mean().reset_index()
            df_media_cob.columns = ['Clusters', 'media_agrupamento']

            df_final = pd.merge(df_table_cob, df_media_cob, on='Clusters')
            df_final['Deflator'] = df_final['y_pred_ajustado'] / df_final['media_agrupamento']
            df_final['Recuperado'] = df_final['Deflator'] * df_final['media_agrupamento'] * df_final['VALOR']

            # SALVA o ARQUIVO
            df_final.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente_2.csv', sep=';',
                            index=False)

            # Agrupamento
            df_final = df_final.groupby(['Clusters']).agg(
                {'VENDAS': ['count'], 'VALOR': ['sum'], 'Recuperado': ['sum']}).reset_index(drop=False)
            df_final.columns = ['Clusters', 'Qtde', 'VALOR', 'Recuperado']
            df_final['Recuperado'] = df_final['Recuperado'].astype(int)

            # adicionando totais
            resultadoTotal = df_final.iloc[0]
            resultadoTotal['Clusters'] = 'TOTAL'
            df_final = df_final.append(resultadoTotal, ignore_index=True)

            df_final['VALOR'][df_final['Clusters'] == 'TOTAL'] = df_final['VALOR'][
                df_final['Clusters'] != 'TOTAL'].sum()
            df_final['Recuperado'][df_final['Clusters'] == 'TOTAL'] = df_final['Recuperado'][
                df_final['Clusters'] != 'TOTAL'].sum()

            df_final['% Recuperado'] = df_final['Recuperado'] / df_final['VALOR']
            df_final['% Recuperado'] = df_final['% Recuperado'].apply(lambda x: '{:.2f}%'.format(x * 100))

            df_final['Qtde'][df_final['Clusters'] == 'TOTAL'] = df_final['Qtde'][df_final['Clusters'] != 'TOTAL'].sum()
            df_final['Recuperação Média'] = df_final['Recuperado'] / df_final['Qtde']
            df_final['Recuperação Média'] = df_final['Recuperação Média'].astype(int)
            # df_final['Recuperação Média']= df_final['Recuperação Média']
            return df_final

        opcao_escolhida = pd.read_csv(rf'{path_diretorio_iduser}\selecao.csv', delimiter=';')

        df_Layout = pd.read_csv(base_construcao, delimiter=';', encoding='latin1')
        df = pd.read_csv(rf'{path_diretorio_iduser}\base_original_importada.csv', delimiter=';',
                         encoding='latin1')
        qtd_total = len(df)

        flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
        flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
        retornar_0_em_percentual = flag_resposta_0 / qtd_total
        retornar_1_em_percentual = flag_resposta_1 / qtd_total

        original = pd.read_csv(caminho_arquivo, delimiter=';', encoding='latin1')

        original_volume_total_de_colunas = pd.read_csv(caminho_arquivo, delimiter=';', encoding='latin1')

        lista_antiga = original_volume_total_de_colunas.columns.tolist()

        original = original[df_Layout.columns.tolist()]
        try:
            original['ID_UNICO']
        except:
            original['ID_UNICO'] = range(1, len(original) + 1)

        try:
            original_volume_total_de_colunas['ID_UNICO']
        except:
            original_volume_total_de_colunas['ID_UNICO'] = range(1, len(original) + 1)

        for col in original.columns:
            if original[col].dtype == "object":
                le = LabelEncoder()
                original.loc[:, col] = le.fit_transform(original[col])

        X = original.drop(columns=["ID_UNICO", "VENDAS"], axis=1)
        y = original["VENDAS"]

        model = xgb.XGBClassifier()
        model.load_model(rf"{path_diretorio_iduser}\modelo.xgb")

        y_probs = model.predict_proba(X)
        original["probabilidade"] = [p[1] for p in y_probs]

        if opcao_escolhida['opcao'][0] == '50%50':
            fator_ajuste = np.array([retornar_0_em_percentual / 0.5, retornar_1_em_percentual / 0.5])
            y_pred_ajustado = y_probs * fator_ajuste
            y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
            original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
            original["VALOR RECUPERADO AJUSTADO"] = original["y_pred_ajustado"] * original["VALOR"]
        if opcao_escolhida['opcao'][0] == '70%30':
            fator_ajuste = np.array([retornar_0_em_percentual / 0.7, retornar_1_em_percentual / 0.3])
            y_pred_ajustado = y_probs * fator_ajuste
            y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
            original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
            original["VALOR RECUPERADO AJUSTADO"] = original["y_pred_ajustado"] * original["VALOR"]
        if opcao_escolhida['opcao'][0] == '80%20':
            fator_ajuste = np.array([retornar_0_em_percentual / 0.8, retornar_1_em_percentual / 0.2])
            y_pred_ajustado = y_probs * fator_ajuste
            y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
            original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
            original["VALOR RECUPERADO AJUSTADO"] = original["y_pred_ajustado"] * original["VALOR"]
        if opcao_escolhida['opcao'][0] == 'Sem balanceamento':
            fator_ajuste = np.array([retornar_0_em_percentual / 0.99, retornar_1_em_percentual / 0.01])
            y_pred_ajustado = y_probs * fator_ajuste
            y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
            original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
            original["VALOR RECUPERADO AJUSTADO"] = original["y_pred_ajustado"] * original["VALOR"]

        original.to_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv', sep=';',
                        index=None)

        try:
            df_priori_resp = pd.read_csv(rf'{path_diretorio_iduser}\priorizar_pergunta_cob.csv', sep=';')
            retorno = df_priori_resp['priorizar'].iloc[0]
        except:
            retorno = 'Probabilidade de Sucesso'

        if retorno == 'Probabilidade de Sucesso':
            var_prioridade = 'y_pred_ajustado'
        else:
            var_prioridade = 'VALOR RECUPERADO AJUSTADO'

        #st.write('Taxa de Conversão da Base de Construção do Modelo: {:.2f}%'.format(retornar_1_em_percentual * 100))

        proba_origiginal = pd.read_csv(rf'{path_diretorio_iduser}\original_com_proba_xg_boost.csv',
                                       delimiter=';')
        prob = proba_origiginal[var_prioridade].mean()
        # print(prob)

        proba_modelo_novo = pd.read_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv',
                                        delimiter=';')
        prob_novo = proba_modelo_novo[var_prioridade].mean()
        # print(prob_novo)
        prob_text = (prob_novo / prob * 100) - 100
        #if prob_text > 0:
        #    st.write('Desempenho estimado em comparação com o mailing da Construção: Incremento de {:.2f}%'.format(
        #        prob_text))
        #elif prob_text == 0:
        #    st.write('Desempenho estimado em comparação com o mailing da Construção: Mesmo Desempenho {:.2f}%'.format(
        #        prob_text))
        #else:
        #    st.write('Desempenho estimado em comparação com o mailing da Construção: Decremento de {:.2f}%'.format(
        #        prob_text))

        #st.write('Estimativa da Taxa de Conversão da Base importada: {:.2f}%'.format(
        #    prob_novo / prob * retornar_1_em_percentual * 100))

        arquivo_clusterizado = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', delimiter=';')
        media_y_pred_por_cluster = arquivo_clusterizado.groupby('Clusters')[var_prioridade].mean().apply(
            lambda x: "{:.2%}".format(x))
        media_y_pred_por_cluste_original = media_y_pred_por_cluster
        # st.title("Média do arquivo original:")
        # st.table(media_y_pred_por_cluster.rename("Taxa de Conversao"))

        opcao_cluster = int(pd.read_csv(rf'{path_diretorio_iduser}\selecao_cluster.csv').iloc[0, 0])
        original = pd.read_csv(rf'{path_diretorio_iduser}\arquivo_novo_do_modelo_treinado.csv',
                               delimiter=';')

        if retorno == 'Probabilidade de Sucesso':
            var_bins = 'probabilidade'
        else:
            var_bins = 'VALOR RECUPERADO AJUSTADO'

        bins = pd.qcut(original[var_bins], q=opcao_cluster,
                       labels=[f'Grupo{i + 1}' for i in range(opcao_cluster)][::-1])
        original['Clusters'] = bins
        original.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv', sep=';', index=None)

        modelo = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv', delimiter=';',
                             encoding='latin1')

        arquivo_clusterizado = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo.csv',
                                           delimiter=';')
        media_y_pred_por_cluster = arquivo_clusterizado.groupby('Clusters')[var_prioridade].mean().apply(
            lambda x: "{:.2%}".format(x))
        #st.title("Modelo x Importado")


        # Recriado a tabela de conversao da base importada
        resul_cluster = pd.read_csv(rf'{path_diretorio_iduser}\resultado_cluster.csv', delimiter=';')
        resul_cluster.drop(resul_cluster.loc[resul_cluster['Clusters'] == 'Total'].index, inplace=True)
        resul_cluster.set_index('Clusters', inplace=True)

        # alterado a coluna para a conversao da base importada
        media_y_pred_por_cluster_unido = pd.concat(
            [resul_cluster['%Sucesso CP'], media_y_pred_por_cluste_original.rename("Taxa de Conversao Base Modelo"),
             media_y_pred_por_cluster.rename("Taxa de Conversao Base Importada")], axis=1)
        media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'] = (media_y_pred_por_cluster_unido[
                                                                                     'Taxa de Conversao Base Importada'].str.replace(
            '%', '').astype('float64') / media_y_pred_por_cluster_unido['Taxa de Conversao Base Modelo'].str.replace(
            '%', '').astype('float64')) * media_y_pred_por_cluster_unido['%Sucesso CP'].str.replace('%', '').astype(
            'float64')
        media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'] = media_y_pred_por_cluster_unido[
            'Taxa de Conversao Base Importada_v2'].round(2)
        media_y_pred_por_cluster_unido['Taxa de Conversao Base Importada_v2'] = media_y_pred_por_cluster_unido[
            'Taxa de Conversao Base Importada_v2'].apply('{:.2f}%'.format)

        #st.table(media_y_pred_por_cluster_unido[['%Sucesso CP', 'Taxa de Conversao Base Importada_v2']].rename(
        #    columns={'%Sucesso CP': 'Taxa de Conversao Base Modelo',
        #             'Taxa de Conversao Base Importada_v2': 'Taxa de Conversao Base Importada'}))
        media_y_pred_por_cluster.rename(var_prioridade)

        df_chart = media_y_pred_por_cluster_unido[['%Sucesso CP', 'Taxa de Conversao Base Importada_v2']].rename(
            columns={'%Sucesso CP': 'Taxa de Conversao Base Modelo',
                     'Taxa de Conversao Base Importada_v2': 'Taxa de Conversao Base Importada'})

        df_chart = df_chart.rename(columns={"Taxa de Conversao Base Modelo": "Base Modelo"})
        df_chart = df_chart.rename(columns={"Taxa de Conversao Base Importada": "Base Importada"})
        df_chart = df_chart.reset_index()
        df_chart['Base Modelo'] = df_chart['Base Modelo'].str.replace('%', '').astype('float64') / 100
        df_chart['Base Importada'] = df_chart['Base Importada'].str.replace('%', '').astype('float64') / 100

        df_orig_mean = df_chart[['Clusters', 'Base Modelo']]
        df_novo_mean = df_chart[['Clusters', 'Base Importada']]

        df_merged = pd.merge(df_orig_mean, df_novo_mean, on='Clusters', suffixes=('_original', '_novo'))

        df_melted = pd.melt(df_merged, id_vars='Clusters', var_name='Arquivo', value_name=f'Média do {var_prioridade}')

        ##########montar uma saida para o grafico

        "df_melted"

        ##########

        df_sem_coluna = modelo.drop(['probabilidade', 'y_pred_ajustado'], axis=1)
        df_sem_coluna.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente.csv', sep=';',
                             index=False)

        # Cria um tabelao do modelo treinado e do novo arquivo
        try:
            df_resp = pd.read_csv(rf'{path_diretorio_iduser}\opt_outliers.csv', sep=';')
            resp = df_resp['opcao'].iloc[0]
        except:
            resp = 'SIM'

        ###################### fazer saida de graficos

        #st.table(tableCobAntes(resp,path_diretorio_iduser))

        #st.write('Importado')
        #st.table(tableCobPos(resp,path_diretorio_iduser))




        ############################


        # ORDENAR CONFORME ESCOLHA DO CLIENTE

        # botao_finalizar =  st.button("Gerar Arquivo", key=312042023)
        # if botao_finalizar:
        df = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente_2.csv', sep=';')
        df_priorizar = pd.read_csv(rf'{path_diretorio_iduser}\priorizar_pergunta_cob.csv', sep=';')
        try:
            status_priori = df_priorizar["priorizar"].iloc[0]
        except:
            status_priori = 'Valor Recuperado'

        if status_priori == 'Valor Recuperado':
            df = df.sort_values(['Recuperado', var_prioridade], ascending=[False, False])
        else:
            df = df.sort_values([var_prioridade, 'Recuperado'], ascending=[False, False])

        lista_atual = df_Layout.columns.tolist()
        for item in lista_antiga:
            if item not in lista_atual:
                try:
                    df[item] = original_volume_total_de_colunas[item]
                except:
                    pass

        df.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente_2.csv', sep=';', index=False)
        # df_final = pd.read_csv(r'C:\GN_Analitycs_WEB_COB\arquivos\ARQUIVO_clusterizado_novo_cliente_2.csv')
        # st.markdown(download_link(df_final, 'dados.csv', 'Clique aqui para baixar os dados em CSV'), unsafe_allow_html=True)

        return original

    arquivo = file_path
    df = pd.read_csv(arquivo, delimiter=';')
    df.columns = df.columns.str.upper()
    df['ID_UNICO'] = range(1, len(df) + 1)

    df.to_csv(rf'{path_diretorio_iduser}\base_importada_2.csv', sep=';', index=None)

    base_construcao = pd.read_csv(rf'{path_diretorio_iduser}\base_construcao_usar.csv', delimiter=';')
    data_ref_1 = data_ref
    # data_ref = '2023-04-18'
    data_ref_1 = pd.to_datetime(data_ref_1)
        # CARREGA O MAILING ORIGINAL
    original = pd.read_csv(rf'{path_diretorio_iduser}\base_importada_2.csv', sep=';')

    # CONVERTE A COLUNA DATA PARA O FORMATO DATA
    original['DATA VENCIMENTO'] = pd.to_datetime(original['DATA VENCIMENTO'], format='%d/%m/%Y',
                                                 errors='coerce')
    try:
        original['DATA PAGAMENTO'] = pd.to_datetime(original['DATA PAGAMENTO'], format='%d/%m/%Y',
                                                    errors='coerce')
    except Exception as err:
        print(err)

    original['DATA PAGAMENTO_TEMP'] = original['DATA PAGAMENTO']
    original['DATA PAGAMENTO'] = original['DATA PAGAMENTO'].fillna(data_ref_1)
    # CRIA UMA COLUNA COM A DIFERENCA DE DIAS

    original['Diferença em Dias'] = (
                original['DATA PAGAMENTO'] - original['DATA VENCIMENTO']).dt.days
    original['Diferença em meses'] = original['Diferença em Dias'] / 30
    original['Diferença em meses'] = original['Diferença em meses'].apply(lambda x: round(x, 0))
    original['Diferença em meses'] = original['Diferença em meses'].astype(int)
    original['DATA PAGAMENTO'] = original['DATA PAGAMENTO_TEMP']

    original.drop(labels='DATA PAGAMENTO_TEMP', axis=1, inplace=True)

    original.to_csv(rf'{path_diretorio_iduser}\base_importada_2.csv', sep=';',
                    index=False)

    prever_vendas(path_diretorio_iduser,base_construcao=base_construcao['base_construcao_usar'][0],caminho_arquivo=rf'{path_diretorio_iduser}\base_importada_2.csv')

    'Download do arquivo'
    df_final = pd.read_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado_novo_cliente_2.csv',sep=';')
    df_final = df_final.drop(['y_pred_ajustado','flag_1.5_iqr','media_agrupamento','Deflator','Recuperado'],axis=1)
    df_final.to_csv(rf'{path_diretorio_iduser}\arquivo_tratado_download.csv',sep=';',index=False)

    #retornar o df_final para p d
