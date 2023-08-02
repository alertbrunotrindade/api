import os
from flask import Flask, request, jsonify,send_file
from carga_e_analiseexploratoria_tela1 import analise_exploratoria_tel1
from coleta_das_variaveis_tela2 import variaveis_escolhidas
from balanceamento_tela3 import rotina_balanceamento
from coleta_cluster_tela4 import gera_cluster
from rodar_novo_arquivo_tela5 import rodar_novo_arquivo
from graficos.grafico_result_cluster_dispersao import grafico_result_cluster_dispersao
from graficos.grafico_reuslt_cluster import grafico_result_cluster
from graficos.revisao_segundo_arquivo import revisao_segundo_arquivo
from graficos.gera_waterfall import grafico_waterfall
import time
import shutil
import json

from app_cob.carga_e_analise_exploratoria_tel1 import analise_exploratoria__tel1
from app_cob.ajustedata_e_priorizar_tel2 import ajuste_e_priorizar
from app_cob.outliers_e_cluster_tel3 import outliers_e_cluster
from app_cob.segundo_arquivo_tel4 import analise_segundoarquivo

#Gera um id com base no timestamp
timestamp = str(time.time()).split('.')[0]  # Obter o timestamp atual

app = Flask(__name__)


TOKEN = "akiva#340340"
#UPLOAD_FOLDER = r"C:\Users\Pichau\Documents\ALERT\PROJETO_ANALI\GN_Analitycs_WEB_files\arquivos/"
UPLOAD_FOLDER = r"C:\Users\planejamento\PycharmProjects\pythonProject1\arquivos/"
UPLOAD_FOLDER_COB = r"C:\Users\planejamento\PycharmProjects\pythonProject1\arquivos\cob/"
@app.route('/upload', methods=['POST'])
def upload_file():
    def delete_files_only(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):  # Verifica se é um arquivo (não é diretório)
                try:
                    os.remove(file_path)
                    #print(f"Arquivo deletado: {file_path}")
                except Exception as e:
                    #print(f"Erro ao deletar o arquivo {file_path}: {e}")
                    pass
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    # Verificar se o arquivo CSV foi enviado
    if 'file' not in request.files:
        return jsonify({'status': 'Nenhum arquivo fornecido'}), 400

    file = request.files['file']
    userid = request.form.get('userid')
    folder_id = str(userid)
    print("Solicitação:", "Arquivo:", file, "usuário", userid)
    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Obter a data e o timestamp atual
    #now = datetime.datetime.now()
    #date_folder = now.strftime("%Y-%m-%d")
    #timestamp = now.strftime("%H-%M-%S")

    # Criar o diretório se não existir
    #folder_path = os.path.join(UPLOAD_FOLDER, folder_id, date_folder, timestamp)
    folder_path = os.path.join(UPLOAD_FOLDER, folder_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Limpa a pasta do usuario mantendo apenas o historico
    delete_files_only(folder_path)

    # Salvar o arquivo na pasta de uploads
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)

    grafico = analise_exploratoria_tel1(folder_path,file_path)
    return jsonify({'status': 'arquivo_recebido', 'caminho_do_arquivo': file_path,'retorno':grafico}), 200

@app.route('/lista_variaveis', methods=['POST'])
def lista_variaveis():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER, str(userid))
    if not os.path.exists(folder_path):
        return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404

    # Verificar se o campo "lista_variaveis" está presente e é uma lista
    lista_variaveis = request.form.get('lista_variaveis')

    if not lista_variaveis:
        return jsonify({'status': 'Campo "lista_variaveis" não fornecido'}), 400

    try:
        lista_variaveis = eval(lista_variaveis)  # Avaliar a string como uma lista
        if not isinstance(lista_variaveis, list):
            raise ValueError
    except (NameError, SyntaxError, ValueError):
        return jsonify({'status': 'Campo "lista_variaveis" inválido. Deve ser uma lista.'}), 400

    variaveis_escolhidas(folder_path, lista_variaveis)
    return jsonify({'status': 'success', 'lista_variaveis': lista_variaveis}), 200

@app.route('/coleta_balanceamento', methods=['POST'])
def coleta_balanceamento():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER, str(userid))
    if not os.path.exists(folder_path):
        return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404

    # Verificar se o campo "balanceamento" está preenchido
    balanceamento = request.form.get('balanceamento')
    if not balanceamento:
        return jsonify({'status': 'Campo "balanceamento" não fornecido'}), 400
    print( balanceamento)
    rotina_balanceamento(folder_path,balanceamento)
    return jsonify({'status': 'success', 'balanceamento': balanceamento}), 200

@app.route('/coleta_cluster', methods=['POST'])
def coleta_cluster():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

    if not os.path.exists(folder_path):
        return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404

    # Verificar se o campo "balanceamento" está preenchido
    cluster = request.form.get('cluster')
    if not cluster:
        return jsonify({'status': 'Campo "cluster" não fornecido'}), 400

    print(cluster)

    gera_cluster(folder_path,cluster)
    return jsonify({'status': 'success', 'cluster': cluster}), 200

@app.route('/upload_segundo', methods=['POST'])
def upload_file_segundo():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    # Verificar se o arquivo CSV foi enviado
    if 'file' not in request.files:
        return jsonify({'status': 'Nenhum arquivo fornecido'}), 400

    file = request.files['file']
    userid = request.form.get('userid')

    print("Solicitação:", "Arquivo:", file, "usuário", userid)
    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404

    # Salvar o arquivo na pasta de uploads
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)

    rodar_novo_arquivo(file_path,folder_path)
    return jsonify({'status': 'arquivo_recebido', 'caminho_do_segundo_arquivo': file_path}), 200

@app.route('/grafico_result_cluster_dispersao', methods=['POST'])
def gera_result_cluster_dispersao():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)
        print("novo caminho:",folder_path)
        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404


    grafico_cluster_disper = grafico_result_cluster_dispersao(folder_path)

    return jsonify({'status': 'success', 'retorno': grafico_cluster_disper}), 200

@app.route('/grafico_result_cluster', methods=['POST'])
def gera_result_cluster():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404


    grafico_cluster = grafico_result_cluster(folder_path)
    json_data = json.dumps({'status': 'success', 'retorno': grafico_cluster}, sort_keys=False)
    return json_data, 200, {'Content-Type': 'application/json'}

@app.route('/gerar_waterfall', methods=['POST'])
def gera_waterfall():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404

    gera_grafico_waterfall = grafico_waterfall(folder_path)
    #print(gera_grafico_waterfall)
    return jsonify({'imagem': gera_grafico_waterfall}), 200
    #return 'ok',200

@app.route('/gera_revisao', methods=['POST'])
def gerar_revisao():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        #valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404



    rev_segundo_arquivo = revisao_segundo_arquivo(folder_path)
    json_data = json.dumps({'status': 'success', 'retorno': rev_segundo_arquivo}, sort_keys=False)
    return json_data, 200, {'Content-Type': 'application/json'}


@app.route('/download',methods=['POST'])
def download_file():
    # Caminho para o arquivo que deseja disponibilizar para download
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404

    arquivo_file = r'\ARQUIVO_clusterizado_novo_cliente.csv'
    arquivo = folder_path+arquivo_file
    print(arquivo)

    # Retorna o arquivo para o cliente
    return send_file(arquivo, as_attachment=True)

@app.route('/salvar_modelo',methods=['POST'])
def salvar_modelos():
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

    # Obter o timestamp atual como string
    timestamp_string = str(int(time.time()))

    # Criar o diretório de destino com base no timestamp
    diretorio_destino = rf"{folder_path}\{timestamp_string}"
    os.makedirs(diretorio_destino)

    # Copiar apenas os arquivos da pasta "arquivos" para o diretório de destino
    origem = rf"{folder_path}"
    destino = diretorio_destino
    for arquivo in os.listdir(origem):
        caminho_arquivo = os.path.join(origem, arquivo)
        if os.path.isfile(caminho_arquivo):
            shutil.copy2(caminho_arquivo, destino)
    print("Arquivos copiados com sucesso!")
    return jsonify({'status': 'success', 'historico_criado': destino}), 200

@app.route('/listar_historico',methods=['POST'])
def listar_historicos():
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER, str(userid))

    # Copiar apenas os arquivos da pasta "arquivos" para o diretório de destino
    origem = rf"{folder_path}"
    lista_pastas = []
    for arquivo in os.listdir(origem):
        caminho_arquivo = os.path.join(origem, arquivo)
        if os.path.isfile(caminho_arquivo):
            pass
        else:
            lista_pastas.append(os.path.basename(caminho_arquivo))
    print("Arquivos copiados com sucesso!")
    return jsonify({'status': 'success', 'lista_historico': lista_pastas}), 200

#########Metodos Cobrança

@app.route('/upload_cob', methods=['POST'])
def upload_file_cob():
    def delete_files_only(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):  # Verifica se é um arquivo (não é diretório)
                try:
                    os.remove(file_path)
                    #print(f"Arquivo deletado: {file_path}")
                except Exception as e:
                    #print(f"Erro ao deletar o arquivo {file_path}: {e}")
                    pass
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    # Verificar se o arquivo CSV foi enviado
    if 'file' not in request.files:
        return jsonify({'status': 'Nenhum arquivo fornecido'}), 400

    file = request.files['file']
    userid = request.form.get('userid')
    folder_id = str(userid)
    print("Solicitação:", "Arquivo:", file, "usuário", userid)
    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Criar o diretório se não existir
    #folder_path = os.path.join(UPLOAD_FOLDER, folder_id, date_folder, timestamp)
    folder_path = os.path.join(UPLOAD_FOLDER_COB, folder_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Limpa diretorio
    delete_files_only(folder_path)

    # Salvar o arquivo na pasta de uploads
    file_path = os.path.join(folder_path, file.filename)
    #print("brunoTeste:",file_path)
    #print("brunoTeste:",folder_path)
    file.save(file_path)

    grafico = analise_exploratoria__tel1(folder_path,file_path)
    return jsonify({'status': 'arquivo_recebido', 'caminho_do_arquivo': file_path,'retorno':grafico}), 200

@app.route('/coleta_balanceamento_cob', methods=['POST'])
def coleta_balanceamento__cob():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid))
    if not os.path.exists(folder_path):
        return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404

    # Verificar se o campo "balanceamento" está preenchido
    balanceamento = request.form.get('balanceamento')
    if not balanceamento:
        return jsonify({'status': 'Campo "balanceamento" não fornecido'}), 400

    data_ref = request.form.get('data_ref')
    if not data_ref:
        return jsonify({'status': 'Campo "data_ref" não fornecido'}), 400

    priorizacao = request.form.get('priorizacao')
    if not data_ref:
        return jsonify({'status': 'Campo "priorizacao" não fornecido'}), 400

    ajuste_e_priorizar(folder_path, data_ref,priorizacao=priorizacao,balanceamento=balanceamento)
    return jsonify({'status': 'success'}), 200


@app.route('/coleta_cluster_cob', methods=['POST'])
def coleta_cluster__cob():
    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid))

    if not os.path.exists(folder_path):
        return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404

    # Verificar se o campo "balanceamento" está preenchido
    cluster = request.form.get('cluster')
    if not cluster:
        return jsonify({'status': 'Campo "cluster" não fornecido'}), 400

    outliers = request.form.get('outliers')
    if not outliers:
        return jsonify({'status': 'Campo "outliers" não fornecido'}), 400

    print(cluster)
    outliers_e_cluster(folder_path, opcao_outliers="SIM", cluster=cluster)

    return jsonify({'status': 'success'}), 200


@app.route('/upload_segundo_cob', methods=['POST'])
def upload_file_segundo_cob():

    # Verificar o token
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    # Verificar se o arquivo CSV foi enviado
    if 'file' not in request.files:
        return jsonify({'status': 'Nenhum arquivo fornecido'}), 400

    file = request.files['file']
    userid = request.form.get('userid')

    print("Solicitação:", "Arquivo:", file, "usuário", userid)
    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    id_modelo = request.form.get('id_modelo')

    # Verificar se o id_modelo foi enviado
    if not id_modelo:
        id_modelo = ''
        folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid))

        # valida se o caminho deste usuario existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'Usuário com ID {userid} não encontrado.'}), 404
    else:
        folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid), id_modelo)

        # valida se o caminho deste usuario com este id_modelo existe
        if not os.path.exists(folder_path):
            return jsonify({'status': f'id_modelo com ID {id_modelo} não encontrado.'}), 404

    data_ref = request.form.get('data_ref')
    if not data_ref:
        return jsonify({'status': 'Campo "data_ref" não fornecido'}), 400

    # Salvar o arquivo na pasta de uploads
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)

    analise_segundoarquivo(folder_path, file_path, data_ref)
    return jsonify({'status': 'arquivo_recebido', 'caminho_do_segundo_arquivo': file_path}), 200


@app.route('/salvar_modelo_cob',methods=['POST'])
def salvar_modelos_cob():
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid))

    # Obter o timestamp atual como string
    timestamp_string = str(int(time.time()))

    # Criar o diretório de destino com base no timestamp
    diretorio_destino = rf"{folder_path}\{timestamp_string}"
    os.makedirs(diretorio_destino)

    # Copiar apenas os arquivos da pasta "arquivos" para o diretório de destino
    origem = rf"{folder_path}"
    destino = diretorio_destino
    for arquivo in os.listdir(origem):
        caminho_arquivo = os.path.join(origem, arquivo)
        if os.path.isfile(caminho_arquivo):
            shutil.copy2(caminho_arquivo, destino)
    print("Arquivos copiados com sucesso!")
    return jsonify({'status': 'success', 'historico_criado': destino}), 200


@app.route('/listar_historico_cob',methods=['POST'])
def listar_historicos_cob():
    token = request.headers.get('Authorization')
    if token != TOKEN:
        return jsonify({'status': 'Não autorizado'}), 401

    userid = request.form.get('userid')

    # Verificar se o nome de usuário foi enviado
    if not userid:
        return jsonify({'status': 'Faltou o campo "userid"'}), 400

    # Verificar se o diretório do usuário existe
    folder_path = os.path.join(UPLOAD_FOLDER_COB, str(userid))

    # Copiar apenas os arquivos da pasta "arquivos" para o diretório de destino
    origem = rf"{folder_path}"
    lista_pastas = []
    for arquivo in os.listdir(origem):
        caminho_arquivo = os.path.join(origem, arquivo)
        if os.path.isfile(caminho_arquivo):
            pass
        else:
            lista_pastas.append(os.path.basename(caminho_arquivo))
    print("Arquivos copiados com sucesso!")
    return jsonify({'status': 'success', 'lista_historico': lista_pastas}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=35189)
