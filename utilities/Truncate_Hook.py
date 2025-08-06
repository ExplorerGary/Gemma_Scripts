import torch
import torch.distributed as dist

DEBUG = True
FP4_CODEBOOKS = {
    "e1m2": torch.tensor([
        -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.0,
         0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5
    ]),
    "e2m1": torch.tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
         0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0
    ]),
    "e3m0": torch.tensor([
       -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.0,
         0.0,  0.25,  0.5,  1.0,  2.0,  4.0,  8.0, 16.0
    ]),
}
FP8_FORMATBOOKS = {
    "e4m3": {
        "exp_bits": 4,
        "mant_bits": 3,
        "bias_fp8": 7,
    },
    "e5m2": {
        "exp_bits": 5,
        "mant_bits": 2,
        "bias_fp8": 15,
    }
}

# --- helper functions ---
def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def print_bits(tensor_uint16):
    # tensor_uint16 是torch.uint16类型的tensor
    bits_list = [format(x.item(), '016b') for x in tensor_uint16]
    for i, bits in enumerate(bits_list):
        print(f"Index {i}: {bits}")

def debug_output(x_uint,x_trunc):
    if DEBUG:
        print("原始uint16位：")
        print_bits(tensor_uint16=x_uint)
        print("截断后uint16位：", x_trunc)
        print_bits(tensor_uint16=x_trunc)

# Truncate Hook:
def make_Truncate_Hook(Mode):
    '''
    接受Mode, 启用对应的Truncate模式。
    fp12: 简单截断（低位抹零）
    fp8、fp4: 提供不同的格式，将数据转化后再按照原始格式输出
    
    Mode的例子： fp12/fp4_e1m2/fp8_e4m3等
    '''
    
    def to_fp4(input_tensor:torch.Tensor,Mode:str) -> torch.Tensor:
        """
        将张量 x 量化为指定 FP4 格式，并用原始 dtype 表示还原后的值。
        """
        encode_format = Mode.split("_")[-1]
        assert encode_format in FP4_CODEBOOKS, f"Unsupported mode: {encode_format}"
        
        codebook = FP4_CODEBOOKS[encode_format].to(input_tensor.device) # 导入codebook
        x_fp32 = input_tensor.to(torch.float32) # 临时转为fp32

        # clip 一下
        min_val, max_val = codebook.min(), codebook.max()
        x_clipped = torch.clamp(x_fp32, min=min_val.item(), max=max_val.item())

        # 计算 x 与每个 codebook 值的差值，得到对应的划分
        diff = (x_clipped.unsqueeze(-1) - codebook.view(1, -1)).abs()
        indices = torch.argmin(diff, dim=-1)
        
        quantized = codebook[indices] # 得到经过量化的数据
        
        return quantized.to(input_tensor.dtype)
        
    def to_fp8(input_tensor:torch.Tensor,Mode):
        if Mode == "fp8_direct" and input_tensor.dtype == torch.float16:
            pass # 这个就是直接截断
        else:
            encode_format = Mode.split("_")[-1]
            assert encode_format in FP8_FORMATBOOKS, f"Unsupported mode: {encode_format}"
            fmt = FP8_FORMATBOOKS[encode_format]
            exp_bits = fmt["exp_bits"]
            mant_bits = fmt["mant_bits"]
            bias_fp8 = fmt["bias_fp8"]
            
            # 转成fp32做中间
            x_fp32 = input_tensor.to(torch.float32)
            x_bits = x_fp32.view(torch.int32)

            sign = (x_bits >> 31) & 0x1
            exp = (x_bits >> 23) & 0xFF
            mantissa = x_bits & 0x7FFFFF  # 取23位尾数

            # exponent 调整
            new_exp = exp - 127 + bias_fp8
            new_exp = torch.clamp(new_exp, 0, (1 << exp_bits) - 1)

            # mantissa 截断
            new_mant = mantissa >> (23 - mant_bits)

            # 反量化近似值（模拟）
            out = (1.0 + new_mant / (2 ** mant_bits)) * (2.0 ** (new_exp - bias_fp8))
            out = out * (-1.0) ** sign

            return out.to(input_tensor.dtype)
                    
        
    def to_fp12(input_tensor:torch.Tensor):
        # 注意，这个钩子只支持fp16/bf16
        assert input_tensor.dtype in [torch.float16, torch.bfloat16], \
            f"Unsupported dtype: {input_tensor.dtype}"  

        # -- basic configs ---
        if input_tensor.dtype == torch.float16:
            m_total = 10
            exp_bits = 5
        else:  # bfloat16
            m_total = 7
            exp_bits = 8
            
        keep_m_bits = max(2, m_total - 4)  # 截断后保留 6 or 3 位尾数
        dtype = input_tensor.dtype
            
        # 转成uint16，然后进行bitwise operation
        x_uint = input_tensor.view(torch.uint16)
        
        total_bits = 1 + exp_bits + m_total
        mask = ((1 << (exp_bits + keep_m_bits + 1)) - 1) << (m_total - keep_m_bits)

        # 应用掩码，抹零
        x_trunc = x_uint & mask
        
        return x_trunc.view(dtype)
        
    def truncate(input_tensor,Mode):
        with torch.no_grad():
            if "fp4" in Mode:
                return to_fp4(input_tensor=input_tensor,Mode=Mode)
            elif "fp8" in Mode:
                return to_fp8(input_tensor=input_tensor,Mode=Mode)
            else:
                return to_fp12(input_tensor=input_tensor)
        
    def Truncate_Hook(process_group: dist.ProcessGroup,
                      bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        input_tensor = bucket.buffer()
        truncated = truncate(input_tensor,Mode = Mode)
        bucket.set_buffer(truncated)
        return _allreduce_fut(process_group, bucket.buffer())
    
    return Truncate_Hook



# if __name__ == "__main__":
#     # 创建一些测试数据
#     x_fp16 = torch.tensor([1.5, -2.75, 0.1, 65504.0, 0.001, 0.0012, 0.00001, 0.0005], dtype=torch.float16)
#     x_bf16 = x_fp16.to(torch.bfloat16)
#     Mode = "fp8_e4m3"
#     to_12_1 = to_fp8(input_tensor=x_fp16,Mode = Mode)
#     to_12_2 = to_fp8(input_tensor=x_bf16,Mode = Mode)
#     print(to_12_1)
#     print(to_12_2)
#     print()
#     Mode = "fp8_e5m2"
#     to_12_1 = to_fp8(input_tensor=x_fp16,Mode = Mode)
#     to_12_2 = to_fp8(input_tensor=x_bf16,Mode = Mode)
#     print(to_12_1)
#     print(to_12_2)

    
    # """
    # tensor([ 1.5000e+00, -2.7500e+00,  9.3750e-02,  4.8000e+02,  7.8125e-03,
    #         8.7891e-03,  9.7656e-03,  7.8125e-03], dtype=torch.float16)
    # tensor([ 1.5000e+00, -2.7500e+00,  9.3750e-02,  2.5600e+02,  7.8125e-03,
    #         8.7891e-03,  9.7656e-03,  7.8125e-03], dtype=torch.bfloat16)

    # tensor([ 1.5000e+00, -2.5000e+00,  9.3750e-02,  5.7344e+04,  9.7656e-04,
    #         9.7656e-04,  3.8147e-05,  4.8828e-04], dtype=torch.float16)
    # tensor([ 1.5000e+00, -2.5000e+00,  9.3750e-02,  6.5536e+04,  9.7656e-04,
    #         9.7656e-04,  3.8147e-05,  4.8828e-04], dtype=torch.bfloat16)
    # """
