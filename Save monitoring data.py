#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/09 22:16
# @Author  : wsh
# @File    : Delta Rds.py
"""
每个数据点都给他保留下来—数据量太大了
取每隔10个点
"""
from scipy.io import loadmat, savemat
import re
import numpy as np

device_num = ['4', '5', '6', '7', '9', '10', "11"]
devicenum = []
cyclenum = []
oringal_Rds = []
Delta_Rds_dic = {}
Rds_dic = {}
delta_Rds = []

path_steady = 'D:\OneDrive\SCUT\IGBT实验平台\MOSFET Aging Data\\' + device_num[6] + "\\" + device_num[6] + '_Steady (Including package temperature).mat'
path_temp = 'E:\MOSFET Aging Data\\11号器件（老化）\温度数据\\11号\\'
path_transient = 'D:\OneDrive\SCUT\IGBT实验平台\MOSFET Aging Data\\' + device_num[6] + "\\" + device_num[6] + '_Transient (Time alignment).mat'
struct_var_pattern = re.compile(r'^Test_(\d+)_run_(\d+)$')

mat_content = loadmat(path_steady, squeeze_me=True)


def cut_and_store_array(A, B, C, Div):
    """
    Cuts an array A into 10 smaller arrays of length B,
    discards any remaining data, and stores them in a dictionary.

    Args:
        A (list or numpy.ndarray): The input array to be cut.
        B (int): The desired length of each of the 10 resulting arrays.

    Returns:
        dict: A dictionary containing the 10 cut arrays, or None if the
              input array is too short.
    """

    # Calculate the minimum required length of array A to get 10 arrays of length B
    required_length = Div * B

    # Check if the input array is long enough
    if len(A) < required_length:
        print(f"Error: The input array's length ({len(A)}) is less than the required length ({required_length}).")
        return None

    # Initialize an empty dictionary to store the results
    result_dict = {}

    # Cut the array and populate the dictionary
    for i in range(Div):
        # Calculate the start and end indices for the current segment
        start_index = i * B
        end_index = start_index + B

        # Get the current segment and store it in the dictionary
        result_dict[f'Run_{C}_{i + 1}'] = A[start_index:end_index]

    return result_dict

all_cyclenums = []
for var_name, var_data in mat_content.items():
    # 忽略 MATLAB 自动生成的元数据


    if var_name.startswith('__'):
        continue
    match = struct_var_pattern.match(var_name)
    if match:
        devicenum = match.group(1)
        cyclenum = int(match.group(2))

        all_cyclenums.append(cyclenum)

max_cyclenum = max(all_cyclenums) if all_cyclenums else None

# Vgs = []
# Vds = []
# Ids = []

Test_Steady = {}
Test_Transient = {}
All_dic = {}
for i in range(int(max_cyclenum)):
    # 提取温度数据
    current_struct_temp = 'Temp_run_' + str(i + 1)
    Temp_Data = loadmat(path_temp + current_struct_temp)
    # 提取电压、电流数据
    current_struct = 'Test_' + devicenum + "_run_" + str(i + 1)

    try:
        # 获取Vds的数据长度
        b = devicenum
        Datalength = len(loadmat(path_steady)[current_struct]['Vds'][0][0][0]['data'][0])

        DeviceNum = np.array([devicenum])
        CycleNum = np.array([i + 1])
        Model = loadmat(path_steady)[current_struct]['Vds'][0][0][0]['model'][0]
        # WaveFormSource = loadmat(path)[current_struct]['Vds'][0][0][0]['waveformSource'][0]
        HorizontalUnits = loadmat(path_steady)[current_struct]['Vds'][0][0][0]['horizontalUnits'][0]
        SampleInterval = loadmat(path_steady)[current_struct]['Vds'][0][0][0]['sampleInterval'][0]

        Vds = cut_and_store_array(loadmat(path_steady)[current_struct]['Vds'][0][0][0]['data'][0],
                                      int(Datalength / 10), i + 1, 10)
        Ids = cut_and_store_array(loadmat(path_steady)[current_struct]['Ids'][0][0][0]['data'][0],
                                      int(Datalength / 10), i + 1, 10)
        Vgs = cut_and_store_array(loadmat(path_steady)[current_struct]['Vgs'][0][0][0]['data'][0],
                                      int(Datalength / 10), i + 1, 10)
        # if i == 1:
        #     Vds = cut_and_store_array(loadmat(path_steady)[current_struct]['Vds'][0][0][0]['data'][0], int(Datalength / 6), i + 1, 6)
        #     Ids = cut_and_store_array(loadmat(path_steady)[current_struct]['Ids'][0][0][0]['data'][0], int(Datalength / 6), i + 1, 6)
        #     Vgs = cut_and_store_array(loadmat(path_steady)[current_struct]['Vgs'][0][0][0]['data'][0], int(Datalength / 6), i + 1, 6)
        # if i == 2:
        #     Vds = cut_and_store_array(loadmat(path_steady)[current_struct]['Vds'][0][0][0]['data'][0], int(Datalength / 5), i + 1, 5)
        #     Ids = cut_and_store_array(loadmat(path_steady)[current_struct]['Ids'][0][0][0]['data'][0], int(Datalength / 5), i + 1, 5)
        #     Vgs = cut_and_store_array(loadmat(path_steady)[current_struct]['Vgs'][0][0][0]['data'][0], int(Datalength / 5), i + 1, 5)

        # savemat('Vds', Vds)
        Test_Steady['Run_'+str(i+1)] = {'DeviceNum': DeviceNum,
                                    'CycleNum': CycleNum,
                                    'Model': Model,
                                    # 'WaveFormSource': WaveFormSource,
                                    'HorizontalUnits': HorizontalUnits,
                                    'SampleInterval': SampleInterval,
                                    'GateSourceVoltage': Vgs,
                                    'DrainSourceVoltage': Vds,
                                    'DrainCurrent': Ids,
                                    'PackageTemperature': Temp_Data}

        # 提取瞬态电压、电流数据
        Vds_tr = loadmat(path_transient)[current_struct]['Vds'][0][0][0]['data'][0]
        Ids_tr = loadmat(path_transient)[current_struct]['Ids'][0][0][0]['data'][0]
        Vgs_tr = loadmat(path_transient)[current_struct]['Vgs'][0][0][0]['data'][0]
        time_tr = loadmat(path_transient)[current_struct]['Vds'][0][0][0]['time'][0]
        SampleInterval_tr = loadmat(path_transient)[current_struct]['Vds'][0][0][0]['sampleInterval'][0]
        Test_Transient['Run_' + str(i + 1)] = {'DeviceNum': DeviceNum,
                                              'CycleNum': CycleNum,
                                              'Model': Model,
                                              # 'WaveFormSource': WaveFormSource,
                                              'HorizontalUnits': HorizontalUnits,
                                              'SampleInterval': SampleInterval_tr,
                                              'Time': time_tr,
                                              'GateSourceVoltage': Vgs_tr,
                                              'DrainSourceVoltage': Vds_tr,
                                              'DrainCurrent': Ids_tr}
        # savemat('Test_6_steady', Test_Steady)

    except KeyError:
        # 当 KeyError 发生时
        print(f"错误: 无法获取。")
        # 立即退出整个循环
        break


A = 1
All_dic = {'Transient': Test_Transient,
           'Steady': Test_Steady}
    # for j in range(int(Datalength)):
    #     # 获取当前值
    #
    #     CurrentTestNum = np.array([(j + 1) * 10])
    #     # CurrentTime = np.array([float(i * 10 + (j + 1)) * 0.1])
    #     CurrentTime = np.array([float(i * 10 + (j + 1))])
    #
    #     Vds = loadmat(path)[current_struct]['Vds'][0][0][0]['data'][0][j]
    #
    #
    #
    #
    # Dic_cycle[str(i + 1)] = Dic
    # Dic = {}
savemat('Test_11.mat', All_dic)
#     # 获取初始值
#     Run_1_Vds = loadmat(path)['Test_'+ devicenum + '_run_1']['Vds'][0][0][0]['data'][0]
#     Run_1_Ids = loadmat(path)['Test_'+ devicenum + '_run_1']['Ids'][0][0][0]['data'][0]
#     r_1_mask = (Run_1_Ids > 14)
#     Org_Rds = (Run_1_Vds/Run_1_Ids)[r_1_mask].reshape(-1, 1)
#
#     # Vds, Ids, Vgs包含了10次功率循环的，需要进行拆分，只保留导通的时候
#
#     r_mask = (Ids > 14)
#     Rds = (Vds/Ids)[r_mask].reshape(-1, 1)[1617:1817, 0]
#     print(Rds.shape)
#     Rds_dic[current_struct] = np.array(Rds)
#
#     if i == 0:
#         oringal_Rds = np.mean(Rds[0:100])
#     else:
#         oringal_Rds = np.mean(Org_Rds)
#
#     # 计算Delta Rds
#     for j in range(0, len(Rds)):
#         # diff = Rds_mid[j] - oringal_Rds
#         diff = (Rds[j] - Rds[0])
#         # diff = Rds_mid[j] - Rds_mid[j-1]
#         delta_Rds.append(diff)
#     # 将他们拼接在一个字典中
#     Delta_Rds_dic[current_struct] = np.array(delta_Rds)
#     delta_Rds = []
#
# # 1. 获取字典中所有的值，并把它们放到一个列表中
# list_of_arrays = list(Delta_Rds_dic.values())
# one_dimensional_array = np.concatenate(list_of_arrays)
# savemat('Test_5_Rds', Rds_dic)
# savemat('Test_5_Delta_Rds', Delta_Rds_dic)
# savemat('Test_5_Delta_Rds_flatten', {'my_array': one_dimensional_array})


# Vds = loadmat(path)['Test_5_run_1']['Vds'][0][0][0]['data'][0]
# Ids = loadmat(path)['Test_5_run_1']['Ids'][0][0][0]['data'][0]





