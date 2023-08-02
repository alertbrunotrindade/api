import pandas as pd
import os
import numpy as np
import re

def variaveis_escolhidas(paht_arquivos_iduser,lista_escolhas=[]):

    if type(lista_escolhas) == 'list':
        escolhas = lista_escolhas
    else:
        escolhas = []

    pasta = f'{paht_arquivos_iduser}'

    if escolhas:
        arquivos =[]
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".csv"):
                arquivos.append(pasta+'\\'+arquivo)

        for arq in arquivos:
            try:
                df_temp = pd.read_csv(arq,sep=';',encoding='latin1')
                df_temp = df_temp[~df_temp['target'].isin(escolhas)]
                df_temp.to_csv(arq, sep=';', encoding='latin1', index=None)
                #print(df_woe['target'])
            except:
                pass
            for esc in escolhas:
                try:
                    df_temp = pd.read_csv(arq, sep=';', encoding='latin1')
                    df_temp = df_temp.drop(esc, axis=1)
                    df_temp.to_csv(arq, sep=';', encoding='latin1', index=None)
                    # print(df_woe['target'])
                except Exception as err:
                     print(err)
                     #pass






