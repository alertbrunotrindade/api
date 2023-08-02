import pandas as pd
import os
import numpy as np
import re
import json

def analise_exploratoria_tel1(path_diretorio_iduser, file_path):


    arquivo = file_path

    def is_numeric(x):
        try:
            float(x)
            return True
        except Exception as err:
            return False

    def all_rows_numeric(df, column):
        return df[column].apply(is_numeric).all()

    def iv_woe(data, target, bins=10, show_woe=False):
        newDF, woeDF = pd.DataFrame(), pd.DataFrame()
        df_ = {'target':[],'value':[]}
        cols = data.columns

        for ivars in cols[~cols.isin([target])]:
            if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
                binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
            else:
                d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
            d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
            d.columns = ['Cutoff', 'N', 'Events']
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
            d['Non-Events'] = d['N'] - d['Events']
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
            d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
            d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
            d.insert(loc=0, column='Variable', value=ivars)
            print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
            df_['target'].append(ivars)
            df_['value'].append(round(d['IV'].sum(), 6))

            temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
            newDF = pd.concat([newDF, temp], axis=0)
            woeDF = pd.concat([woeDF, d], axis=0)

            if show_woe == True:
                print(d)
        return newDF, woeDF, pd.DataFrame.from_dict(df_)

    #inicio da analise


    df = pd.read_csv(arquivo, delimiter=';',encoding='latin1')
    df.columns = df.columns.str.upper()
    df = df.fillna(0)
    df.to_csv(rf'{path_diretorio_iduser}/base_original_importada.csv',sep=';',index=None)
    qtd_total = len(df)
    qtd_colunas = len(df.columns)
    flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
    flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
    flag_resposta_em_percentual = len(df['VENDAS'][df['VENDAS'] == 1]) / len(df['VENDAS']) * 100

    #analisa as variaveis
    "analise_exploratoria(df=df)"

    #Analisando as colunas... Aguarde
    colunas = df.columns
    colunas_numericas = []
    colunas_categoricas = []
    colunas_aprovadas_categoricas = []
    colunas_removidas_1000 = []
    colunas_removidas_50_per = []
    colunas_removidas_nan = []
    colunas_removidas_unico = []
    colunas_removidas_exp_regular = []

    colunas = df.columns.str.upper().to_list()
    df.columns = df.columns.str.upper().to_list()

    for coluna in colunas:
        colunas_buscadas = ['CPF', 'CNPJ', 'CEP', 'DDD', 'TEL', 'CEL','CD_ALERT']
        for colunas in colunas_buscadas:
            match = re.search(colunas, coluna)
            if match:
                similaridade = int(100 * (len(match.group(0)) / len(colunas)))
                print(f'Coluna encontrada: {coluna} - Similaridade: {similaridade}%')
                colunas_removidas_exp_regular.append(coluna)

        porcentagem_nan = df[coluna].isnull().mean()

        valor_unico = len(set(df[coluna]))
        if coluna in colunas_removidas_exp_regular:
            pass
        elif coluna == 'flag_resposta' or str(coluna).upper() == 'VENDA' and df[coluna][df[coluna]==0].count()<500:
            print("PROCESSO ENCERRADO POIS O VOLUME DE FLAG RESPOSTA 0 É INFERIOR A 500")
            break
        elif coluna == 'flag_resposta' or str(coluna).upper() == 'VENDA' and df[coluna].sum()<100:
            print("PROCESSO ENCERRADO POIS O VOLUME DE FLAG RESPOSTA É INFERIOR A 100")
            break
        elif len(df[coluna])<500:
            print("PROCESSO ENCERRADO POIS O VOLUME DE REGISTRO É INFERIOR A 500")
            break
        elif valor_unico == 1:
            print(f" {coluna} contém apenas um valor único")
            colunas_removidas_unico.append(coluna)
        elif porcentagem_nan >= 0.90:
            print(f" {coluna} contém + de 90% nulo")
            colunas_removidas_nan.append(coluna)
        elif all_rows_numeric(df.fillna(0), coluna) == True:
            colunas_numericas.append(coluna)
        else:
            colunas_categoricas.append(coluna)
            if len(df[f'{coluna}'].unique()) >=1000 and all_rows_numeric(df.fillna(0), coluna) == False:
                print("Variavel com muitas categorias:", coluna)
                colunas_removidas_1000.append(coluna)
            else:
                if len(df[f'{coluna}'].unique()) / len(df[f'{coluna}']) > 0.5 and all_rows_numeric(df.fillna(0), coluna) == False:

                    print('o conjunto possui mais de 50% de variaveis distintas para a quantidade total')

                    colunas_removidas_50_per.append(coluna)
                else:
                    colunas_aprovadas_categoricas.append(coluna)
                    print('o conjunto possui menos de 50% de variaveis distintas para a quantidade total')


    colunas = df.columns.str.upper().to_list()
    num_iterations = len(colunas)
    df_ajustado = df[colunas_numericas + colunas_aprovadas_categoricas]
    df_ajustado.to_csv(rf'{path_diretorio_iduser}/variaveis_aprovadas_original.csv',sep=';',encoding='latin1',index=None)
    #'DF_AJUSTADO!'

    iv, woe, dfw = iv_woe(data=df_ajustado, target='VENDAS')

    #"Importância das variáveis"
    df_woe = dfw.sort_values('value', ascending=False)
    df_woe2 = df_woe[df_woe['value'] >= 0.02]
    last_row_index = df_woe2.index[-1]
    df_woe2.at[last_row_index, 'target'] = 'VENDAS'
    df_grafico = df_woe[df_woe['value'] >= 0.01]
    ####REMOVER
    lista_remover_depois = df_woe2['target'].to_list()

    ################
    base_aprovada_final_analise = df_ajustado[lista_remover_depois]
    df_woe.to_csv(rf'{path_diretorio_iduser}/df_woe_original.csv',sep=';',encoding='latin1',index=None)
    df_woe2.to_csv(rf'{path_diretorio_iduser}/df_woe_0_02.csv',sep=';',encoding='latin1',index=None)
    df_grafico.to_csv(rf'{path_diretorio_iduser}/df_woe_0_01_grafico.csv',sep=';',encoding='latin1',index=None)
    #base_aprovada_final_analise.to_csv(r'C:\GN_Analitycs_WEB\arquivos\base_aprovada_final_woe_0_02.csv',sep=';', encoding='utf-8',index=None)
    ## CORRELACAO DAS VARIAVEIS
    df_independent_variables = base_aprovada_final_analise
    corr_matrix = df.corr().abs()
    threshold = 0.70
    corr_vars = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                corr_vars.add(colname)

    for item in corr_vars:
        colname = item.split()[0]
        try:
            df_independent_variables.drop(columns=colname, inplace=True)
        except:
            pass

    df_independent_variables.to_csv(rf'{path_diretorio_iduser}/base_aprovada_final_woe_0_02.csv',sep=';', encoding='utf-8',index=None)
    #corr_matrix.to_csv(r'C:\GN_Analitycs_WEB\arquivos\correla_teste.csv',sep=';', encoding='utf-8',index=None)
    ##
    #Criando o json
    original = pd.read_csv(rf'{path_diretorio_iduser}/df_woe_0_02.csv', delimiter=';')
    lista_variaveis = original['target'].to_list()

    dict_return = {'grafico':'','lista_variaveis':lista_variaveis}
    dict_return['grafico'] = df_woe.to_dict(orient='records')
    #dict_return['grafico'] = df_woe.head(25).to_json()
    return dict_return





