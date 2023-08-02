import json
def revisao_segundo_arquivo(path_diretorio_iduser):
    caminho_arquivo = rf'{path_diretorio_iduser}\resumo_arquivo_segundo.txt'
    # Coletar o conteúdo do arquivo em uma variável
    with open(caminho_arquivo, "r") as arquivo:
        conteudo = arquivo.read()
    return json.loads(conteudo)