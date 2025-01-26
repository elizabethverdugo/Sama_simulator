#code extracted from "extraindo_resultados.ipynb" by Nicholas

 #Bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
import pickle
import numpy as np
import statistics as st
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import cumfreq
from osgeo import gdal,ogr,osr
import rasterio
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import gc
from itertools import cycle,product


# Encontrando arquivos pickle e filtrando por arquivos específicos:
def get_files(pasta, extensao):
    busca = f'{pasta}/**/*{extensao}'
    return glob.glob(busca, recursive=True)


def find_files(arquivos, palavras):
    # Filtrar a LISTA de arquivos que contém qualquer uma das palavras no nome. Palavras precisa ser lista por causa do any()
    arquivos_filtrados = [arquivo for arquivo in arquivos if
                          any(palavra in os.path.basename(arquivo) for palavra in palavras)]
    return arquivos_filtrados


def get_data_dict(file,type):
    with open(file[0], 'rb') as file:
        data = pickle.load(file)
    if type == 'downlink':
        raw_dict = data[0]['downlink_data']
    elif type == 'uplink':
        raw_dict = data[0]['uplink_data']
    else:
        raise ValueError(f"Type must be 'downlink' or 'uplink'")
    return raw_dic


def get_raw_dict(file,type):
    with open(file[0], 'rb') as file:
        data = pickle.load(file)
    if type == 'downlink':
        raw_dict = data[0]['downlink_data']['raw_data']
    elif type == 'uplink':
        raw_dict = data[0]['uplink_data']['raw_data']
    else:
        raise ValueError(f"Type must be 'downlink' or 'uplink'")
    return raw_dict


# Adaptado do sama
def extract_parameter_from_raw(raw_data, parameter_name, data_index, subindex=None, calc=None, concatenate=True):
    """
    Extrai e organiza dados de raw_data para um parâmetro específico, com suporte a subíndices e cálculos.

    Args:
        raw_data: Lista de dados brutos.
        parameter_name: Nome do parâmetro a ser extraído.
        data_index: Índice correspondente ao número de BSs simuladas.
        subindex: Índice do parâmetro, caso seja necessário (ex.: para acessar x[parameter_name][subindex]).
        calc: Operação a ser aplicada ('avg', 'std' ou None para retornar todos os dados).
        concatenate: Se True, concatena os dados em um array único; caso contrário, mantém como lista.

    Returns:
        Dados extraídos e possivelmente processados.
    """
    if calc is None:
        if concatenate:
            if subindex is not None:
                # Concatenar com subíndice
                extracted_data = np.concatenate([x[parameter_name][subindex] for x in raw_data[data_index]])
            else:
                # Concatenar sem subíndice
                extracted_data = np.concatenate([x[parameter_name] for x in raw_data[data_index]])
        else:
            if subindex is not None:
                # Manter como lista com subíndice
                extracted_data = [x[parameter_name][subindex] for x in raw_data[data_index]]
            else:
                # Manter como lista sem subíndice
                extracted_data = [x[parameter_name] for x in raw_data[data_index]]
    elif calc == 'avg':
        if subindex is not None:
            extracted_data = [x[parameter_name][subindex].mean() for x in raw_data[data_index]]
        else:
            extracted_data = [x[parameter_name].mean() for x in raw_data[data_index]]
    elif calc == 'std':
        if subindex is not None:
            extracted_data = [x[parameter_name][subindex].std() for x in raw_data[data_index]]
        else:
            extracted_data = [x[parameter_name].std() for x in raw_data[data_index]]
    else:
        raise ValueError(f"Operação desconhecida: {calc}")

    return extracted_data


# Função para calcular a CDF
def calculate_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf



def group_ue(data_dict, iter_dict_name, data_index=None):
    # this function will pick the output simulation data dict and will group the UEs by beam and sector and
    # also indicates the UEs that was not connected to the network
    dict = []
    if data_index is None:
        # bs_list = range(data_dict['BSs'].__len__())
        iter_list = range(data_dict[iter_dict_name].__len__())
    else:
        iter_list = [data_index]
    for bs_data_index in iter_list:
        nactive_ue_cnt = []  # UEs non-connected to the network
        ue_per_beam = []  # ues grouped by beam
        ue_per_sector = []  # ues grouped by sector
        active_ues = []  # UEs connected to the network
        ue_bs_tables = [x['ue_bs_table'] for x in data_dict['raw_data'][bs_data_index]]
        for i, ue_bs_tb in enumerate(ue_bs_tables):
            beam_comb = np.array(list(product(ue_bs_tb['bs_index'].unique(), ue_bs_tb['beam_index'].unique(), ue_bs_tb['sector_index'].unique())))
            sec_comb = np.array(list(product(ue_bs_tb['bs_index'].unique() , ue_bs_tb['sector_index'].unique())))
            act_beams = beam_comb[(beam_comb[:, 0] != -1) & (beam_comb[:, 1] != - 1) & (beam_comb[:, 2] != -1)]
            act_sec = sec_comb[(sec_comb[:, 0] != -1) & (sec_comb[:, 1] != - 1)]
            nactive_ue_cnt.append(np.sum(ue_bs_tb['bs_index'] == -1))
            active_ues.append(np.where(ue_bs_tb['bs_index'] != -1)[0])
            ue_bs_tb = np.array(ue_bs_tb)
            dummy_beam = []
            dummy_sec = []
            for index in act_beams:
                dummy_beam.append(np.where((ue_bs_tb[:, 0] == index[0]) & (ue_bs_tb[:, 1] == index[1]) &
                                            (ue_bs_tb[:, 2] == index[2]))[0])
            ue_per_beam.append(dummy_beam)
            for index in act_sec:
                dummy_sec.append(np.where((ue_bs_tb[:, 0] == index[0]) & (ue_bs_tb[:, 2] == index[1]))[0])
            ue_per_sector.append(dummy_sec)

        dict.append({'nactive_ue_cnt': nactive_ue_cnt, 'active_ues': active_ues, 'ue_per_beam':ue_per_beam,
                    'ue_per_sector': ue_per_sector})

    return dict


def generate_inactive_ue_curve(datasets, iter_dict_name, datacolor, datalinestyle, iter_range, xlabel, path):
    # Dicionário para armazenar as UEs desconectadas médias por cenário
    avg_disconn_ues_dict = {}

    # Processar cada dataset separadamente
    for label, file_path in datasets.items():
        print(f"Processando o cenário: {label}")

        # Obter os dados do dataset atual usando a função get_data_dict
        data = get_data_dict(file=file_path, type='downlink')  # Ou 'uplink', dependendo do seu caso

        # Agrupar os UEs por beam e sector
        beam_sec_groupings = [
            group_ue(data_dict=data, iter_dict_name=iter_dict_name, data_index=bs - 1)[0]
            for bs in iter_range
        ]

        # Calcular a média de UEs desconectados
        avg_disconn_ues = [np.mean(x['nactive_ue_cnt']) for x in beam_sec_groupings]

        # Armazenar as UEs desconectadas médias para o cenário atual
        avg_disconn_ues_dict[label] = avg_disconn_ues

        # Liberar a memória do dataset processado
        del data
        gc.collect()  # Forçar a liberação de memória

    # Criar o gráfico
    plt.figure(figsize=(12, 8))

    # Plotar cada linha para cada cenário (média das UEs desconectadas)
    #  for (label, capacities), color, linestyle in zip(mean_capacities.items(), datacolor.values(), datalinestyle.values()):

    for (label, avg_ues), color, linestyle in zip(avg_disconn_ues_dict.items(), datacolor.values(),
                                                  datalinestyle.values()):
        color = datacolor.get(label, 'black')  # Cor para o cenário
        linestyle = datalinestyle.get(label, '-')  # Estilo da linha para o cenário

        sns.lineplot(
            x=iter_range,
            y=avg_ues,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

    # Configurações do gráfico
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Média de UEs Desconectados", fontsize=12)
    plt.title("Comparação da Média de UEs Desconectados por Número de Base Stations", fontsize=14)
    plt.xticks(iter_range)  # Mostrar todos os valores no eixo X
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar legenda
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(f'{path}/inactive_ues_comparison.png')

    # Exibir o gráfico
    plt.show()


# inactive UEs - this graphics is not different for uplink/downlink
    if data_dict['downlink_data']['BSs']:
        beam_sec_groupings = group_ue(data_dict=data_dict['downlink_data'], iter_dict_name=iter_dict_name)
    elif data_dict['uplink_data']['BSs']:
        beam_sec_groupings = group_ue(data_dict=data_dict['uplink_data'], iter_dict_name=iter_dict_name)
    avg_disconn_ues = [np.mean(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    std_disconn_ues = [np.std(x['nactive_ue_cnt']) for x in beam_sec_groupings]
    title = 'UEs not connected to the RAN'
    default_curve_plt(n_bs_vec=iter_list, data=avg_disconn_ues, std=std_disconn_ues, xlabel=xlabel, title=title,
                      path=path, save=True, save_name='not_connected_ues')


file_path = "12_09_24-10_54_17"

correcao_pasta = get_files('output','.pkl')
corre = find_files(correcao_pasta,['12_09_24-10_54_17.pkl'])
corre

# Obter os dados do dataset atual usando a função get_data_dict
data = get_data_dict(file=corre, type='downlink')  # Ou 'uplink', dependendo do seu caso