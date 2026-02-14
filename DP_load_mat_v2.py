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