import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import dump


def modelo_xg(construcao,validacao,path_arquivos_user):

    #df_bcu =pd.read_csv(rf'{path_arquivos_user}\base_construcao_usar.csv')
    #construcao = df_bcu['base_construcao_usar'][0]
    #validacao = df_bcu['base_construcao_usar'][0]
    opcao_escolhida = pd.read_csv(rf'{path_arquivos_user}\selecao.csv',delimiter=';')

    df = pd.read_csv(rf'{path_arquivos_user}\base_original_importada.csv',delimiter=';',encoding='latin1')
    qtd_total = len(df)
    flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
    flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
    retornar_0_em_percentual = flag_resposta_0 / qtd_total
    retornar_1_em_percentual = flag_resposta_1 / qtd_total

    construcao = pd.read_csv(construcao,delimiter=';',encoding='latin1')
    validacao = pd.read_csv(validacao,delimiter=';',encoding='latin1')
    original = pd.read_csv(rf'{path_arquivos_user}\base_aprovada_final_woe_0_02.csv',delimiter=';')

    for col in construcao.columns:
        if construcao[col].dtype == "object":
            le = LabelEncoder()
            construcao.loc[:, col] = le.fit_transform(construcao[col])

    for col in validacao.columns:
        if validacao[col].dtype == "object":
            le = LabelEncoder()
            validacao.loc[:, col] = le.fit_transform(validacao[col])

    for col in original.columns:
        if original[col].dtype == "object":
            le = LabelEncoder()
            original.loc[:, col] = le.fit_transform(original[col])

    X1 = construcao.drop("VENDAS", axis=1)
    y1 = construcao["VENDAS"]

    X2 = validacao.drop("VENDAS", axis=1)
    y2 = validacao["VENDAS"]

    X3 = original.drop("VENDAS", axis=1)
    y3 = original["VENDAS"]

    X_train = X1
    y_train = y1
    X_test = X2
    y_test = y2
    X_origin = X3
    y_origin = y3

    random_state = 42

    # Cria a instância do modelo com o argumento random_state definido
    modelxgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.7, eta=0.3, random_state=random_state)
    modelxgb.fit(X_train, y_train)

    # Avaliar o modelo
    scorexgb = modelxgb.score(X_train, y_train)
    score2xgb = modelxgb.score(X_test, y_test)
    score3xgb = modelxgb.score(X_origin, y_origin)
    dict_result_xgb = {'construcao': ["{:.2}".format(scorexgb)],
                          'validacao': ["{:.2}".format(score2xgb)],
                          'original': ["{:.2}".format(score3xgb)]}
    df_result_xgb = pd.DataFrame.from_dict(dict_result_xgb)

    y_probs_train = modelxgb.predict_proba(X_train)
    y_probs_test = modelxgb.predict_proba(X_test)
    y_probs_ori = modelxgb.predict_proba(X_origin)

    construcao["probabilidade"] = [p[1] for p in y_probs_train]

    validacao["probabilidade"] = [p[1] for p in y_probs_test]

    original["probabilidade"] = [p[1] for p in y_probs_ori]

    if opcao_escolhida['opcao'][0] =='50%50':
        fator_ajuste = np.array([retornar_0_em_percentual/0.5, retornar_1_em_percentual/0.5])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='70%30':
        fator_ajuste = np.array([retornar_0_em_percentual/0.7, retornar_1_em_percentual/0.3])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='80%20':
        fator_ajuste = np.array([retornar_0_em_percentual/0.8, retornar_1_em_percentual/0.2])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='Sem balanceamento':
        fator_ajuste = np.array([retornar_0_em_percentual/0.99, retornar_1_em_percentual/0.01])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]

    modelxgb.save_model(rf"{path_arquivos_user}\modelo.xgb")

    construcao.to_csv(rf'{path_arquivos_user}\construcao_com_proba_xg_boost.csv', sep=';', index=None)
    validacao.to_csv(rf'{path_arquivos_user}\validacao_com_proba_xg_boost.csv', sep=';', index=None)
    original.to_csv(rf'{path_arquivos_user}\original_com_proba_xg_boost.csv', sep=';', index=None)
    df_result_xgb.to_csv(rf'{path_arquivos_user}\score_xgboost.csv', sep=';', index=False)

    "Modelo 3:"
    "Acuracia do modelo de construção é: {:.2%}".format(scorexgb)
    "Acuracia do modelo de validação é: {:.2%}".format(score2xgb)
    "Acuracia do modelo original é: {:.2%}".format(score3xgb)

def random(construcao,validacao,path_arquivos_user):

    opcao_escolhida = pd.read_csv(rf'{path_arquivos_user}\selecao.csv',delimiter=';')

    df = pd.read_csv(rf'{path_arquivos_user}\base_original_importada.csv',delimiter=';',encoding='latin1')
    qtd_total = len(df)
    flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
    flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
    retornar_0_em_percentual = flag_resposta_0 / qtd_total
    retornar_1_em_percentual = flag_resposta_1 / qtd_total

    construcao = pd.read_csv(construcao,delimiter=';',encoding='latin1')
    validacao = pd.read_csv(validacao,delimiter=';',encoding='latin1')
    original = pd.read_csv(rf'{path_arquivos_user}\base_aprovada_final_woe_0_02.csv',delimiter=';')

    for col in construcao.columns:
        if construcao[col].dtype == "object":
            le = LabelEncoder()
            construcao.loc[:, col] = le.fit_transform(construcao[col])

    for col in validacao.columns:
        if validacao[col].dtype == "object":
            le = LabelEncoder()
            validacao.loc[:, col] = le.fit_transform(validacao[col])

    for col in original.columns:
        if original[col].dtype == "object":
            le = LabelEncoder()
            original.loc[:, col] = le.fit_transform(original[col])

    X1= construcao.drop("VENDAS", axis=1)
    y1 = construcao["VENDAS"]

    X2 = validacao.drop("VENDAS", axis=1)
    y2 = validacao["VENDAS"]

    X3 = original.drop("VENDAS", axis=1)
    y3 = original["VENDAS"]

    X_train = X1
    y_train = y1
    X_test = X2
    y_test = y2
    X_origin = X3
    y_origin = y3

    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=10, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,  # min_impurity_split=None,#
                                 min_samples_leaf=30, min_samples_split=15,
                                 min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                 oob_score=False, random_state=42, verbose=0,
                                 warm_start=False)
    model.fit(X_train, y_train)

    "Modelo 2:"
    score = model.score(X_train, y_train)
    score2 = model.score(X_test, y_test)
    score3 = model.score(X_origin, y_origin)
    "Acuracia do modelo de construção é: {:.2%}".format(score)
    "Acuracia do modelo de validação é: {:.2%}".format(score2)
    "Acuracia do modelo original é: {:.2%}".format(score3)

    dict_result_forest = {'construcao':[ "{:.2}".format(score)],
                          'validacao':["{:.2}".format(score2)],
                          'original':["{:.2}".format(score3)]}
    df_result_forest = pd.DataFrame.from_dict(dict_result_forest)

    y_probs_train = model.predict_proba(X_train)
    y_probs_test = model.predict_proba(X_test)
    y_probs_ori = model.predict_proba(X_origin)

    construcao["probabilidade"] = [p[1] for p in y_probs_train]

    validacao["probabilidade"] = [p[1] for p in y_probs_test]

    original["probabilidade"] = [p[1] for p in y_probs_ori]

    original.sort_values('probabilidade', ascending=False)

    if opcao_escolhida['opcao'][0] =='50%50':
        fator_ajuste = np.array([retornar_0_em_percentual/0.5, retornar_1_em_percentual/0.5])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='70%30':
        fator_ajuste = np.array([retornar_0_em_percentual/0.7, retornar_1_em_percentual/0.3])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='80%20':
        fator_ajuste = np.array([retornar_0_em_percentual/0.8, retornar_1_em_percentual/0.2])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]

    if opcao_escolhida['opcao'][0] =='Sem balanceamento':
        fator_ajuste = np.array([retornar_0_em_percentual / 0.99, retornar_1_em_percentual / 0.01])
        y_pred_ajustado = y_probs_ori * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]

    #salvando o modelo randown
    dump(model, rf'{path_arquivos_user}\modelo_random_forest.joblib')

    try:
        construcao.to_csv(rf'{path_arquivos_user}\construcao_com_proba_random.csv',sep=';',index=None)
        validacao.to_csv(rf'{path_arquivos_user}\validacao_com_proba_random.csv',sep=';',index=None)
        original.to_csv(rf'{path_arquivos_user}\original_com_proba_random.csv',sep=';',index=None)
        df_result_forest.to_csv(rf'{path_arquivos_user}\score_randomforest.csv',sep=';',index=False)

    except Exception as error:
        print(error)

def regressao_logistica(construcao,validacao,path_arquivos_user):

    opcao_escolhida = pd.read_csv(rf'{path_arquivos_user}\selecao.csv',delimiter=';')

    df = pd.read_csv(rf'{path_arquivos_user}\base_original_importada.csv',delimiter=';',encoding='latin1')
    qtd_total = len(df)
    flag_resposta_0 = len(df['VENDAS'][df['VENDAS'] == 0])
    flag_resposta_1 = len(df['VENDAS'][df['VENDAS'] == 1])
    retornar_0_em_percentual = flag_resposta_0 / qtd_total
    retornar_1_em_percentual = flag_resposta_1 / qtd_total
    construcao = pd.read_csv(construcao,
                     delimiter=';')
    validacao = pd.read_csv(validacao,delimiter=';')

    base_original = pd.read_csv(rf'{path_arquivos_user}\base_aprovada_final_woe_0_02.csv',delimiter=';')

    for col in construcao.columns:
        if construcao[col].dtype == "object":
            le = LabelEncoder()
            construcao.loc[:, col] = le.fit_transform(construcao[col])

    for col in validacao.columns:
        if validacao[col].dtype == "object":
            le = LabelEncoder()
            validacao.loc[:, col] = le.fit_transform(validacao[col])

    for col in base_original.columns:
        if base_original[col].dtype == "object":
            le = LabelEncoder()
            # base_original.loc[:, col] = le.fit_transform(base_original[col]) (comentado para parte melhorada)
            base_original.loc[:, col] = le.fit_transform(pd.to_numeric(base_original[col], errors='coerce'))

    train = construcao
    test = validacao
    original = base_original


    X_train = train.drop("VENDAS", axis=1)
    y_train = train["VENDAS"]

    X_test = test.drop("VENDAS", axis=1)
    y_test = test["VENDAS"]

    X_test2 = original.drop("VENDAS", axis=1)
    y_test2 = original["VENDAS"]

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_original = model.predict(X_test2)

    y_probs_train = model.predict_proba(X_train)
    y_probs_test = model.predict_proba(X_test)
    y_probs_original = model.predict_proba(X_test2)

    accuracy = accuracy_score(y_train, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_orig = accuracy_score(y_test2, y_pred_original)

    train["probabilidade"] = [p[1] for p in y_probs_train]

    test["probabilidade"] = [p[1] for p in y_probs_test]

    original["probabilidade"] = [p[1] for p in y_probs_original]

    if opcao_escolhida['opcao'][0] =='50%50':
        fator_ajuste = np.array([retornar_0_em_percentual/0.5, retornar_1_em_percentual/0.5])
        y_pred_ajustado = y_probs_original * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='70%30':
        fator_ajuste = np.array([retornar_0_em_percentual/0.7, retornar_1_em_percentual/0.3])
        y_pred_ajustado = y_probs_original * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]
    if opcao_escolhida['opcao'][0] =='80%20':
        fator_ajuste = np.array([retornar_0_em_percentual/0.8, retornar_1_em_percentual/0.2])
        y_pred_ajustado = y_probs_original * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]

    if opcao_escolhida['opcao'][0] =='Sem balanceamento':
        fator_ajuste = np.array([retornar_0_em_percentual / 0.99, retornar_1_em_percentual / 0.01])
        y_pred_ajustado = y_probs_original * fator_ajuste
        y_pred_ajustado /= np.sum(y_pred_ajustado, axis=1, keepdims=True)
        original["y_pred_ajustado"] = [p[1] for p in y_pred_ajustado]

    train.to_csv(rf'{path_arquivos_user}\construcao_regressao.csv',sep=';',encoding='latin1',index=None)
    test.to_csv(rf'{path_arquivos_user}\validacao_regressao.csv',sep=';',encoding='latin1',index=None)
    original.to_csv(rf'{path_arquivos_user}\original_regressao.csv',sep=';',encoding='latin1',index=None)

    "Modelo 1"
    "Acuracia do modelo de construção é: {:.2f}%".format(accuracy * 100)
    "Acuracia do modelo de validação é: {:.2f}%".format(accuracy_test * 100)
    "Acuracia do modelo original é: {:.2f}%".format(accuracy_orig * 100)

def particiona(arquivo,particionamento,path_arquivos_user):

    df = pd.read_csv(arquivo,delimiter=';',encoding='latin1')

    if particionamento =='30%':
        particionamento_1 = 0.7
        particionamento_2 = 0.7
        opcao = 'validação'

    if particionamento =='70%':
        particionamento_1 = 0.3
        particionamento_2 = 0.3
        opcao = 'construção'

    df_vendas_1 = df[df['VENDAS'] == 1]
    df_vendas_0 = df[df['VENDAS'] == 0]

    train_vendas_1, test_vendas_1 = train_test_split(df_vendas_1, test_size=particionamento_1)
    train_vendas_0, test_vendas_0 = train_test_split(df_vendas_0, test_size=particionamento_2)

    modelo_de_treino = pd.concat([train_vendas_1, train_vendas_0])

    modelo_de_teste = pd.concat([test_vendas_1, test_vendas_0])

    ones = modelo_de_treino[modelo_de_treino['VENDAS'] == 1]
    zeros = modelo_de_treino[modelo_de_treino['VENDAS'] == 0].iloc[:len(ones) * 4, :]

    if opcao =='construção':
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_construção.csv',sep=';',index=None)
        base_balanceada_80_20_construcao = pd.concat([ones, zeros])
        base_balanceada_80_20_construcao.to_csv(rf'{path_arquivos_user}\base_balanceada_80_20_construcao.csv', sep=';',index=None)


    else:
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_validação.csv',sep=';',index=None)
        base_balanceada_80_20_validação = pd.concat([ones, zeros])
        base_balanceada_80_20_validação.to_csv(rf'{path_arquivos_user}\base_balanceada_80_20_validação.csv', sep=';',index=None)


    ones = modelo_de_treino[modelo_de_treino['VENDAS'] == 1]
    zeros = modelo_de_treino[modelo_de_treino['VENDAS'] == 0].iloc[:len(ones) * 1, :]

    if opcao == 'construção':
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_construção.csv',sep=';',index=None)
        base_balanceada_80_20_construcao = pd.concat([ones, zeros])
        base_balanceada_80_20_construcao.to_csv(rf'{path_arquivos_user}\base_balanceada_50_50_construcao.csv', sep=';',index=None)


    else:
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_validação.csv',sep=';',index=None)
        base_balanceada_80_20_validação = pd.concat([ones, zeros])
        base_balanceada_80_20_validação.to_csv(rf'{path_arquivos_user}\base_balanceada_50_50_validação.csv', sep=';',index=None)


    ones = modelo_de_treino[modelo_de_treino['VENDAS'] == 1]
    zeros = modelo_de_treino.loc[modelo_de_treino['VENDAS'] == 0].iloc[:int(len(ones) * 2.33), :]


    if opcao == 'construção':
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_construção.csv',sep=';',index=None)
        base_balanceada_80_20_construcao = pd.concat([ones, zeros])
        base_balanceada_80_20_construcao.to_csv(rf'{path_arquivos_user}\base_balanceada_70_30_construcao.csv', sep=';',index=None)


    else:
        modelo_de_treino.to_csv(rf'{path_arquivos_user}\modelo_validação.csv',sep=';',index=None)
        base_balanceada_80_20_validação = pd.concat([ones, zeros])
        base_balanceada_80_20_validação.to_csv(rf'{path_arquivos_user}\base_balanceada_70_30_validação.csv', sep=';',index=None)
    #EVENTO RARO E BALANCEAMENTO

def rotina_balanceamento(path_arquivos_user,balanceamento):
    'Evento raro?'
    opcao1 = 'Não'
    if opcao1 == 'Não':
        particiona(arquivo=rf'{path_arquivos_user}\base_aprovada_final_woe_0_02.csv', particionamento='70%',path_arquivos_user=path_arquivos_user)
        particiona(arquivo=rf'{path_arquivos_user}\base_aprovada_final_woe_0_02.csv', particionamento='30%',path_arquivos_user=path_arquivos_user)

    elif opcao1 =='Sim':
        'Modelo em construção/ Selecione a opção não e confirmar.'

    else:
        "Selecione um balanceamento"

    'Selecione o metodo de balanceamento (Recomendamos Balancear)'
    opcao = balanceamento

    print("opcao:",opcao,"balanceamento:",balanceamento,"comparação opcao == '70%30'",opcao == "70%30")

    if opcao =='50%50':
        base_construcao_usar = rf'{path_arquivos_user}\base_balanceada_50_50_construcao.csv'
        base_validacao_usar = rf'{path_arquivos_user}\base_balanceada_50_50_validação.csv'

    if opcao =='80%20':
        base_construcao_usar = rf'{path_arquivos_user}\base_balanceada_50_50_construcao.csv'
        base_validacao_usar = rf'{path_arquivos_user}\base_balanceada_50_50_validação.csv'

    if opcao == '70%30':
        print("entrou opcao:", opcao, "balanceamento:", balanceamento)
        base_construcao_usar = rf'{path_arquivos_user}\base_balanceada_70_30_construcao.csv'
        base_validacao_usar = rf'{path_arquivos_user}\base_balanceada_70_30_validação.csv'

    if opcao =='Sem balanceamento':
        base_construcao_usar = rf'{path_arquivos_user}\modelo_construção.csv'
        base_validacao_usar = rf'{path_arquivos_user}\modelo_construção.csv'

    data = [opcao]
    base_con_usar = [base_construcao_usar]
    df_bcu = pd.DataFrame(base_con_usar, columns=['base_construcao_usar'])
    df_bcu.to_csv(rf'{path_arquivos_user}\base_construcao_usar.csv', index=False)

    df = pd.DataFrame(data, columns=['opcao'])
    df.to_csv(rf'{path_arquivos_user}\selecao.csv', index=False)

    #regressao_logistica
    regressao_logistica(construcao=base_construcao_usar,validacao=base_validacao_usar,path_arquivos_user=path_arquivos_user)
    #random
    random(construcao=base_construcao_usar,validacao=base_validacao_usar,path_arquivos_user=path_arquivos_user)
    #modelo_xg
    modelo_xg(construcao=base_construcao_usar,validacao=base_validacao_usar,path_arquivos_user=path_arquivos_user)




