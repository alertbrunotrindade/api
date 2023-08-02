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


def outliers_e_cluster(path_diretorio_iduser, opcao_outliers="SIM",cluster = 4):

    df_outliers = pd.DataFrame([opcao_outliers], columns=['opcao'])
    df_outliers.to_csv(rf'{path_diretorio_iduser}\opt_outliers.csv', index=False)


    "Clusters"
    opcao_cluster = cluster
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


    #AJUSTE PARA RESPEITAR A PRIORIDADE SELECIONADA
    try:
        df_priori_resp = pd.read_csv(rf'{path_diretorio_iduser}\priorizar_pergunta_cob.csv',sep=';')
        retorno = df_priori_resp['priorizar'].iloc[0]
    except:
        retorno = 'Probabilidade de Sucesso'

    if retorno == 'Probabilidade de Sucesso':
        original_dupl = original['probabilidade'].duplicated()

        # Adiciona um número aleatório entre 0.00000001 e 0.00000000000001 aos valores duplicados
        original.loc[original_dupl, 'probabilidade'] += np.random.uniform(0.00000001, 0.00000000000001)

        bins = pd.qcut(original['probabilidade'], q=num_clusters,
                       labels=[f'Grupo{i + 1}' for i in range(num_clusters)][::-1])
    else:
        bins = pd.qcut(original['VALOR RECUPERADO AJUSTADO'], q=num_clusters,
                       labels=[f'Grupo{i + 1}' for i in range(num_clusters)][::-1])
    #SALVANDO OS CLUSTERS
    original['Clusters'] = bins
    original.to_csv(rf'{path_diretorio_iduser}\ARQUIVO_clusterizado.csv', sep=';', index=None)
    #st.write('Foi particionado em ', opcao_cluster)





