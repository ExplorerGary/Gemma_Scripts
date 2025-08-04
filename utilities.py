# utilities:
import os
import torch
import numpy as np
import logging
'''
1. scan_pt(base_dir:str)
    扫描特定盘下所有的.pt文件，然后返回一个整理好的绝对路径列表
2. read_pt(pt_path:str)
    读入一个.pt文件
    然后展平成一个巨大的flatten tensor
3. to_int(pt: torch.Tensor, scale: float = 1e8) -> np.ndarray:
    将 float16 tensor 安全地放大为 int32，默认放大倍率为1e8
'''

def scan_pt(base_dir:str):
    '''
    扫描特定盘下所有的.pt文件，然后返回一个整理好的绝对路径列表
    '''
    avail_pt = []

    for file in os.listdir(base_dir):
        # if file.endswith("pt"):
        if file.endswith(".pt"):
            file_path = os.path.join(base_dir,file)
            avail_pt.append(file_path)

    return avail_pt

def read_pt(pt_path) -> np.ndarray:
    '''
    返回 array ，等待使用 EG 编码。
        如果 pt 是 dict，则集体展平成一个一维 array。然后返回。
    '''
    pt = torch.load(pt_path, map_location="cpu")
    if isinstance(pt, dict):
        arrays = []
        for v in pt.values():
            arrays.append(v.flatten())
        # print(f"array:{len(arrays)}")
        pt_array = np.concatenate(arrays)
    else:
        pt_array = pt.flatten()
    return pt_array

def read_pt_tensor(pt_path) -> torch.Tensor:
    '''
    返回 array ，等待使用 EG 编码。
        如果 pt 是 dict，则集体展平成一个一维 torch.Tensor 叫做pt_array。然后返回。
    '''
    pt = torch.load(pt_path, map_location="cpu")
    if isinstance(pt, dict):
        tensors = []
        for v in pt.values():
            tensors.append(v.flatten())
        # print(f"array:{len(arrays)}")
        pt_array = torch.cat(tensors)
    else:
        pt_array = pt.flatten()
    return pt_array

def to_int_fuct(pt: np.ndarray, scale: float = 1e8) -> np.ndarray:
    """
    将 float16 np.ndarray 安全地放大为 int32，默认放大倍率为1e8
    使用 float32 进行缩放以避免精度问题。
    """
    return np.round(pt.astype(np.float32) * scale).astype(np.int32)

def to_int(pt: np.ndarray, scale: float = 1e6) -> np.ndarray:
    try:
        return to_int_fuct(pt=pt,scale=scale)
    except Exception as e:
        print(e)
        return None
    
def quantlization_fuct_np(pt_array:np.ndarray,scaling:int = 2**8, fp64_enable:bool = False, debug:bool = False) -> np.ndarray:
    """
    对输入的pt_array进行量化处理：
    先将pt_array乘以2的8次方，四舍五入后再除以2的8次方，返回量化后的数组。
    :param pt_array: 输入的numpy数组
    :param scaling: 量化的缩放因子，默认为2的8次方
    :return: 量化后的numpy数组
    """
    if not isinstance(pt_array, np.ndarray):
        return TypeError("pt_array must be a numpy array")
    if fp64_enable and debug:
        print("fp64_enable",fp64_enable)

    pt_array = pt_array.astype(np.float64) if fp64_enable else pt_array
    
    if debug:
        print(pt_array.dtype)
        print(pt_array)
    quantized = np.round(pt_array * scaling) / scaling
    # print(quantized)
    return quantized

def quantlization_np(pt_array:np.ndarray,scaling:int = 2**2,fp64_enable:bool = False,debug:bool = False) -> np.ndarray:
    try:
        return quantlization_fuct_np(pt_array=pt_array,scaling=scaling,fp64_enable = fp64_enable,debug=debug)
    except Exception as e:
        print(e)
        return None

def quantlization_fuct_pt(pt_array:torch.Tensor,scaling:int = 2**8, fp64_enable:bool = False, debug:bool = False) -> torch.Tensor:
    """
    对输入的pt_array进行量化处理：
    先将pt_array乘以2的8次方，四舍五入后再除以2的8次方，返回量化后的数组。
    :param pt_array: 输入的numpy数组
    :param scaling: 量化的缩放因子，默认为2的8次方
    :return: 量化后的numpy数组
    """
    if not isinstance(pt_array, torch.Tensor):
        return TypeError("pt_array must be a torch.Tensor")
    if fp64_enable and debug:
        print("fp64_enable",fp64_enable)
    if debug:
        print(pt_array.dtype)
        print(pt_array)
    
    
    working_tensor = pt_array.to(dtype=torch.float64) if fp64_enable else pt_array
    quantized = torch.round(working_tensor * scaling) / scaling
    
    
    if debug:
        print(quantized)
    return quantized

def quantlization_pt(pt_array:torch.Tensor,scaling:int = 2**2,fp64_enable:bool = False,debug:bool = False) -> torch.Tensor:
    try:
        return quantlization_fuct_pt(pt_array=pt_array,scaling=scaling,fp64_enable = fp64_enable,debug=debug)
    except Exception as e:
        print(e)
        return None



def scan_csv(base_dir:str):
    '''
    扫描特定盘下所有的.pt文件，然后返回一个整理好的绝对路径列表
    '''
    avail_csv = []

    for file in os.listdir(base_dir):
        # if file.endswith("pt"):
        if file.endswith(".csv"):
            file_path = os.path.join(base_dir,file)
            avail_csv.append(file_path)

    return avail_csv

def read_pt_from_csv(csv_path:str):
    '''
    从 csv 中读取 pt 文件路径
    返回一个列表，包含所有的 pt 文件路径
    '''
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "name" in df.columns:
        avail_pt = df["name"].tolist()
        return avail_pt
    else:
        raise ValueError("CSV file must contain a 'name' column with pt file paths.")


def test():
#     base_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\COMPRESS_COMPETITION\\packed\\"
#     file_name = "dummy.pt"
#     t1 = torch.tensor([[1,2,3],[4,5,6]])
#     t2 = torch.tensor([[1,2,3],[4,5,6]])
#     torch.save({"1": t1, "2": t2}, os.path.join(base_dir, file_name))
#     logging.info("tensor saved")
    
#     pt_array = read_pt(os.path.join(base_dir,file_name))
#     logging.info("tensor read!")
#     print(pt_array)
#     print(len(pt_array),pt_array.dtype)
#     print(f"""
# calculated byte = {8 * len(pt_array)}

# os byte = {os.path.getsize(os.path.join(base_dir,file_name))}
# """)
#     print()
    local_dir = "D:\\NYU_Files\\2025 SPRING\\Summer_Research\\新\\PYTHON\\QWEN\\dummy_files\\"
    avail_pt = scan_pt(local_dir)
    # print(len(avail_pt))
    for pt in avail_pt:
        pt_array = read_pt(pt)
        logging.info("tensor read!")
        # print(pt_array if len(pt_array)<1e2 else len(pt_array))
        print(len(pt_array),pt_array.dtype)
        print(f"""
calculated byte = {2 * len(pt_array)}

os byte = {os.path.getsize(pt)}
""")
# test()


    