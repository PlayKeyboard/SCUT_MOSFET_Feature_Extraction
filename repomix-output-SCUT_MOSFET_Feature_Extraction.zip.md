This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
DP_load_mat_v2.py/
  DP_load_mat_v2.py
load_mat.py/
  load_mat.py
Save monitoring data.py/
  Save monitoring data.py
```

# Files

## File: DP_load_mat_v2.py/DP_load_mat_v2.py
```python
import os
import re
from scipy.io import loadmat, savemat
import numpy as np

"""提取瞬态MOSFET数据"""

def process_and_organize_data(folder_path):
    """
    遍历指定文件夹，按循环数和通道号组织所有数据，并进行重命名。

    Args:
        folder_path (str): 包含数据文件的文件夹路径。

    Returns:
        dict: 一个字典，键是新的命名（'Test_run_1'等），值是组织好的数据。
    """
    # 1. 定义文件名的正则表达式模式，捕获循环数和通道号
    pattern = re.compile(r'^(\d{8})_(PC|DP)_(\d+)_(\d+)_(ch\d+)\.mat$')

    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹路径 '{folder_path}' 不存在。")
        return None

    # 2. 定义通道号和新名称的映射关系
    channel_name_map = {
        'ch2': 'Vds',
        'ch3': 'Vgs',
        'ch5': 'Ids'
    }

    # 3. 临时存储组织好的数据
    organized_data_dict = {}

    # 4. 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)

        if match:
            # 提取文件名中的关键信息
            _, _, cycle_num_str, _, channel_num = match.groups()
            cycle_num = int(cycle_num_str)

            file_path = os.path.join(folder_path, filename)

            try:
                # 加载 .mat 文件
                mat_data = loadmat(file_path)

                # 创建一个字典来存储当前文件的所有变量
                current_file_data = {}
                for var_name, data in mat_data.items():
                    # 忽略 loadmat 自动添加的元数据
                    if not var_name.startswith('__'):
                        current_file_data[var_name] = data

                # 获取新的通道名称，如果不存在则使用原始名称
                new_channel_name = channel_name_map.get(channel_num, channel_num)

                if cycle_num not in organized_data_dict:
                    organized_data_dict[cycle_num] = {}

                organized_data_dict[cycle_num][new_channel_name] = current_file_data

                print(f"已处理文件: {filename}，数据已按循环数 {cycle_num} 和通道 {new_channel_name} 合并。")

            except Exception as e:
                print(f"错误: 处理文件 {filename} 时发生异常：{e}")

    print("\n--- 所有文件处理完毕 ---")

    # 5. 根据排序后的循环数，将数据重新组织到一个新的字典中
    final_output_data = {}
    sorted_cycles = sorted(organized_data_dict.keys())
    for i, cycle_num in enumerate(sorted_cycles):
        var_name = f"Test_11_run_{i + 1}"
        final_output_data[var_name] = organized_data_dict[cycle_num]

    print("数据已按要求命名并合并完毕。")
    return final_output_data


# 6. 调用函数并接收返回值
if __name__ == "__main__":
    current_folder = 'E:\MOSFET Aging Data\\11号器件（老化）\\20250907_DP'
    processed_data = process_and_organize_data(current_folder)

    savemat('11_Transient.mat', processed_data)

    if processed_data:
        print("\n处理完成的数据已成功返回。")
        print("现在你可以使用 'processed_data' 变量来访问数据。")

        # 示例：访问 'Test_run_1' 中的 'Vds' 数据
        if 'Test_run_1' in processed_data:
            vds_data = processed_data['Test_run_1']['Vds']['data']
            print(f"Test_run_1 的 Vds 数据形状为: {vds_data.shape}")
```

## File: load_mat.py/load_mat.py
```python
import os
import re
from scipy.io import loadmat, savemat
import numpy as np

"""提取稳态MOSFET数据"""

def process_and_organize_data(folder_path):
    """
    遍历指定文件夹，按循环数和通道号组织所有数据，并进行重命名。

    Args:
        folder_path (str): 包含数据文件的文件夹路径。

    Returns:
        dict: 一个字典，键是新的命名（'Test_run_1'等），值是组织好的数据。
    """
    # 1. 定义文件名的正则表达式模式，捕获循环数和通道号
    pattern = re.compile(r'^(\d{8})_(PC|DP)_(\d+)_(\d+)_(ch\d+)\.mat$')

    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹路径 '{folder_path}' 不存在。")
        return None

    # 2. 定义通道号和新名称的映射关系
    channel_name_map = {
        'ch1': 'Vds',
        'ch3': 'Vgs',
        'ch5': 'Ids',
        'ch6': 'Gate'
    }

    # 3. 临时存储组织好的数据
    organized_data_dict = {}

    # 4. 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)

        if match:
            # 提取文件名中的关键信息
            _, _, cycle_num_str, _, channel_num = match.groups()
            cycle_num = int(cycle_num_str)

            file_path = os.path.join(folder_path, filename)

            try:
                # 加载 .mat 文件
                mat_data = loadmat(file_path)

                # 创建一个字典来存储当前文件的所有变量
                current_file_data = {}
                for var_name, data in mat_data.items():
                    # 忽略 loadmat 自动添加的元数据
                    if not var_name.startswith('__'):
                        current_file_data[var_name] = data

                # 获取新的通道名称，如果不存在则使用原始名称
                new_channel_name = channel_name_map.get(channel_num, channel_num)

                if cycle_num not in organized_data_dict:
                    organized_data_dict[cycle_num] = {}

                organized_data_dict[cycle_num][new_channel_name] = current_file_data

                print(f"已处理文件: {filename}，数据已按循环数 {cycle_num} 和通道 {new_channel_name} 合并。")

            except Exception as e:
                print(f"错误: 处理文件 {filename} 时发生异常：{e}")

    print("\n--- 所有文件处理完毕 ---")

    # 5. 根据排序后的循环数，将数据重新组织到一个新的字典中
    final_output_data = {}
    sorted_cycles = sorted(organized_data_dict.keys())
    for i, cycle_num in enumerate(sorted_cycles):
        var_name = f"Test_11_run_{i + 1}"
        final_output_data[var_name] = organized_data_dict[cycle_num]

    print("数据已按要求命名并合并完毕。")
    return final_output_data


def add_temperature_data_to_final_data(final_data, temperature_file_path):
    """
    读取温度文件，并将其所有内容作为一个新的key添加到 final_data 的顶层。

    Args:
        final_data (dict): 之前处理好的 final_organized_data 字典。
        temperature_file_path (str): 包含温度数据的新 .mat 文件的路径。

    Returns:
        dict: 更新后的 final_data 字典。
    """
    if not os.path.exists(temperature_file_path):
        print(f"错误: 温度文件未找到，路径为 {temperature_file_path}")
        return final_data

    try:
        # 加载温度文件中的所有变量
        temp_data_raw = loadmat(temperature_file_path)

        # 移除 MATLAB 的元数据，只保留实际的数组
        temp_variables = {key: value for key, value in temp_data_raw.items() if not key.startswith('__')}

        print(f"\n成功加载温度文件。找到 {len(temp_variables)} 个温度数组。")

        # 将所有温度变量作为一个新的键 'PackageTemperature' 添加到顶层字典
        final_data['PackageTemperature'] = temp_variables

        print(f"已将温度文件内容添加到 'PackageTemperature' 键下。")

    except Exception as e:
        print(f"错误: 处理温度文件时发生异常 - {e}")

    return final_data


# 6. 调用函数并接收返回值
if __name__ == "__main__":
    current_folder = 'E:\MOSFET Aging Data\\11号器件（老化）\\20250907_PC'
    processed_data = process_and_organize_data(current_folder)

    # savemat('4_Steady', processed_data)

    if processed_data:
        # --- 第2步：添加新的温度数据 ---
        # 替换为你的温度文件路径
        temperature_file_path = 'E:\MOSFET Aging Data\\11号器件（老化）\温度数据\\11号\\Temp_1_40_11.mat'

        updated_processed_data = add_temperature_data_to_final_data(processed_data, temperature_file_path)

        # --- 第3步：保存最终结果（可选） ---
        if updated_processed_data:
            output_filename = '11_Steady.mat'
            try:
                savemat(output_filename, updated_processed_data)
                print(f"\n成功将所有数据保存到文件: {output_filename}")
            except Exception as e:
                print(f"\n错误: 保存文件时出错 - {e}")
```

## File: Save monitoring data.py/Save monitoring data.py
```python
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
```
