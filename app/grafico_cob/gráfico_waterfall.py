import streamlit as st
import altair as alt
import pandas as pd
import shap
import streamlit.components.v1 as components
import xgboost

def grafico_waterfall():
    #data = pd.read_csv(r'C:\GN_Analitycs_WEB_COB\arquivos\resultado_cluster.csv', delimiter=';')
    try:
        base_construcao_waterfall = pd.read_csv(r'C:\GN_Analitycs_WEB_COB\arquivos\base_construcao_usar.csv', delimiter=';')
        data = pd.read_csv(base_construcao_waterfall['base_construcao_usar'][0], delimiter=';')
        #data.info()
        if len(data) >50000:
            st.title("Impossivel calcular devido a quantidade de linhas superior a 50 mil")
        else:
            #data.info()
            X3 = data.drop("VENDAS", axis=1)
            X3 = X3[X3.select_dtypes(exclude=['object']).columns]
            y3 = data["VENDAS"]
            #data.info()
            #X3.info()

            #X_origin = X_origin.drop(["Clusters", '% Conversão', '%Vendas', '%Qtde', 'Index'], axis=1, inplace=True)
            X_origin = X3
            y_origin = y3
            #DSC_PLANO, SGL_ESTADO, DSC_CLASSIFICACAO, CAMPAIGN_CD
            # train XGBoost model
            model = xgboost.XGBClassifier().fit(X_origin, y_origin)

            # compute SHAP values
            explainer = shap.Explainer(model, X_origin)
            shap_values = explainer(X_origin)

            # visualize the training set predictions
            # Exibe o gráfico no Streamlit

            st.title("Importância Variáveis Numéricas")
            st.pyplot(shap.plots.waterfall(shap_values[0], max_display=20), use_container_width=True)
    except:
        st.title("Sem variaveis numéricas!")

