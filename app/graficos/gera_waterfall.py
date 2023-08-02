import time
import base64
import streamlit as st
import altair as alt
import pandas as pd
import shap
import streamlit.components.v1 as components
import xgboost
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def grafico_waterfall(path_diretorio_iduser):
    # data = pd.read_csv(r'C:\GN_Analitycs_WEB\arquivos\resultado_cluster.csv', delimiter=';')
    try:
        #path_diretorio_iduser=r'C:\Users\planejamento\PycharmProjects\pythonProject1\arquivos\1123456'
        base_construcao_waterfall = pd.read_csv(rf'{path_diretorio_iduser}\base_construcao_usar.csv', delimiter=';')
        data = pd.read_csv(base_construcao_waterfall['base_construcao_usar'][0], delimiter=';')
        # data.info()
        if len(data) > 5000000:
            return {'retorno': "Impossivel calcular devido a quantidade de linhas superior a 5 mi"}
        else:
            # data.info()
            X3 = data.drop("VENDAS", axis=1)
            X3 = X3[X3.select_dtypes(exclude=['object']).columns]
            y3 = data["VENDAS"]
            # data.info()
            # X3.info()

            # X_origin = X_origin.drop(["Clusters", '% Conversão', '%Vendas', '%Qtde', 'Index'], axis=1, inplace=True)
            X_origin = X3
            y_origin = y3
            # DSC_PLANO, SGL_ESTADO, DSC_CLASSIFICACAO, CAMPAIGN_CD
            # train XGBoost model
            model = xgboost.XGBClassifier().fit(X_origin, y_origin)

            # compute SHAP values
            explainer = shap.Explainer(model, X_origin)
            shap_values = explainer(X_origin)

            # visualize the training set predictions
            # Exibe o gráfico no Streamlit

            # Defina o caminho e o nome do arquivo de imagem
            image_path = rf'{path_diretorio_iduser}\waterfall_plot.png'

            #shap.plots.waterfall(shap_values[0], max_display=20)

            def create_figure():
                fig = plt.figure(figsize=(8, 6))  # Defina o tamanho da figura conforme necessário

                shap.plots.waterfall(shap_values[0], max_display=20, show=False)
                plt.tight_layout()  # Ajusta a disposição do gráfico
                return fig

              # Desativar o aviso de depreciação

            """
             # Defina o caminho e o nome do arquivo de imagem
            image_path = rf'{path_diretorio_iduser}\waterfall_plot.png'

            fig, ax = plt.subplots()
            # Gerar o gráfico waterfall
            st.pyplot(shap.plots.waterfall(shap_values[0], max_display=20), use_container_width=True)

            # Salvar o gráfico como uma imagem

            st.set_option('deprecation.showPyplotGlobalUse', False)  # Desativar o aviso de depreciação
            fig.savefig(image_path, dpi=300)


            # Converter a imagem em Base64
            def gerar_base64(image_path):
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                return encoded_string

            # Retornar a imagem em Base64
            return gerar_base64(image_path)

            """

            def gerar_base64(image_path):
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                return encoded_string

            figs = create_figure()
            figs.savefig(image_path, dpi=300)
            # Retornar a imagem em Base64
            return  gerar_base64(image_path)
    except Exception as e:
        print(e)
        return [e]
