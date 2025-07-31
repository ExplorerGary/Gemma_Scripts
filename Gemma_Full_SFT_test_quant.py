# Gemma_SFT.py

'''
基于Gemma_SFT.ipynb的代码，进行分布式训练的SFT Trainer。
跟笔记本相比，支持分布式训练
提供钩子抓取数据
'''

# 启动命令： torchrun --nproc_per_node=2 Gemma_SFT.py
# "--scaling": scaling的参数
# "--save_bucket": 是否保存bucket
# "--pioneer": 挑选少部分子集进行测试

# 0. 导包：
import os
from datasets import load_dataset
from datasets import load_from_disk
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTTrainer
from dist_check import dist_check
from trl import SFTConfig
from transformers import DefaultFlowCallback # 导入默认的东西
from transformers import TrainerCallback
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook 
import csv
import argparse



# 0. 准备工具函数：
def quantlization_fuct(flat_tensor:torch.Tensor,
                       scaling:int = None,
                       fp64_enable:bool = False):
    '''
    观察记录：
    1. fp16的最高数字约为为6.5e5，也就意味着我们最好不要使用1e3及以上的tensor，不然就变成inf了(因为有情况下时会出现Xe2的数量级的)
        但不知道为什么，经过测试后发现原来的scaling是可行的。
    
    '''
    global Pioneer
    if Pioneer:
        print(f"doing quantlization, scaling = {scaling}")
    
    try:
        if fp64_enable:
            flat_tensor = flat_tensor.to(dtype=torch.float64)
            
        quantilized = torch.round(flat_tensor * scaling) / scaling
        if scaling is None:
            quantilized = flat_tensor
        return quantilized
    
    except Exception as e:
        raise e    

# System message for the assistant
system_message = "You are an expert product description writer for Amazon."

# User prompt that combines the user query and the schema
user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""
    # Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# 1. 准备数据集
def prepare_dataset(pioneer:bool = False):

    # Load dataset from the hub/local disk
    # dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
    dataset_path = "/gpfsnyu/home/zg2598/datasets/philschmid_amazon-product-descriptions-vlm/"
    full_dataset = load_from_disk(dataset_path)
    dataset = full_dataset["train"]
    if pioneer:
        dataset = dataset.select(range(10))  # 只取前10个样本，用于功能测试
    # Convert dataset to OAI messages
    # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
    dataset = [format_data(sample) for sample in dataset]

    # if dist.get_rank() == 0: # 只在主进程打印信息
    print(f"example data:\n{dataset[345]['messages'] if not pioneer else dataset[-1]['messages']}")


    return dataset



# 2. 准备模型
# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
model_path = "/gpfsnyu/home/zg2598/Gemma/gemma-3-4b-pt/" # working on local
processor_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/gemma-3-4b-it-processor/gemma-3-4b-it-processor" # using local processor
# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")



# 修改过的版本：
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # torchrun会设置这个环境变量
device_map = {"": local_rank}

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map=device_map,  # ✅ 重点修复！
)
# 移除Lora 量化策略

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
processor = AutoProcessor.from_pretrained(processor_path)

# if dist.get_rank() == 0: # 只在主进程打印信息
print("Model loaded, Processor loaded...\nDONE!!!")




# 3. 准备SFTConfig和损失函数：


args = SFTConfig(
    output_dir="gemma-product-description",     # directory to save and repository id
    num_train_epochs=1,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                         # save checkpoint every epoch when doing actual experiment, but for debugging, we save nothing
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    
    push_to_hub=False,                           # don't push model to hub !!!!!!!!!!
    
    report_to="tensorboard",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch



# 5. 准备装设了钩子的HookedSFTTrainer


# EPOCH 和 STEP 怎么找：


class EPOCH_STEP_HANDLER(TrainerCallback):
# class EPOCH_STEP_HANDLER(DefaultFlowCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        global CURRENT_EPOCH
        CURRENT_EPOCH = int(state.epoch or 0)
        
        return super().on_epoch_begin(args, state, control, **kwargs)
        
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        global CURRENT_STEP
        CURRENT_STEP = state.global_step
        
        return super().on_step_begin(args, state, control, **kwargs)



# DDP钩子：

#           dummy hook： allreduce_hook (默认DDP钩子，不改变行为）


#           mod_allreduce_hook: 添加读取和保存的信息：

# --- helper function ---
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
    
# --- hook本体 ---
def mod_allreduce_hook_EG(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    '''
    由mod_allreduce_hook_base修改而来：
    本体允许结合名为EPOCH_STEP_HANDLER的TrainerCallback，实现：
        1. 知晓这个bucket的meta数据 {
            1. rank
            2. epoch
            3. step
            4. index
        }
        2. 记录每一个bucket里的内容
        3. 根据save_Bucket变量决定是否保存GradBucket里面的数据
    
    更新后支持：
        1. 使用一个scaling参数对数据进行quantlization -- 2025年7月29日实现
    '''
    
    
    # --- 导入东西 --- 
    global CURRENT_EPOCH
    global CURRENT_STEP
    global save_Bucket
    global Scaling

    # --- 缓冲区里的扁平向量 --- 
    flat_tensor = bucket.buffer()
    # --- 基本信息 --- 
    global param_name_map
    output_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results/"
    
    # 1. 知道这个是哪个rank:
    rank = dist.get_rank()
    
    # 2. 知道这是这个batch(或者step)第几个bucket:
    idx = bucket.index()
    
    # 3. 知道存储的数据类型：
    data_type = flat_tensor.dtype
    
    # 4. 知道这个桶里面塞了什么？然后存下来！
    params = bucket.parameters()  # List[Tensor]
    grads = bucket.gradients()  # List[Tensor]，对应顺序应该和 params 一致 -- [已确认]

    
    # 4.1 知道这个桶属于哪个step和epoch
    the_epoch = CURRENT_EPOCH
    the_step = CURRENT_STEP

    
    #### DEBUGING ####
    print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}")
    ##################
    
    
    ### 更新 ###
    
    # 1. 量化
    if Scaling is not None:
        quantized = quantlization_fuct(flat_tensor=flat_tensor,
                                       scaling=Scaling,
                                       fp64_enable=False)
        # set_buffer
        bucket.set_buffer(quantized) # 2025年7月29日：测试量化后的表现 
    # 2. val2index
    
    # 3. EG Encoding
    
    

    
    
    # bucket.set_buffer(codes) # 将bucket的内容更改为EG encoding的结果: codes
    
    ############
    
    
    
    
    # 4.1.1:
    
    # 文件名称：
    file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
    # 保存路径
    save_dir = os.path.join(output_path, "COMMUNICATION_DATA_EG") # 更换了保存路径
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    

    
    # 4.2 具体保存
    try:
        param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]

        # print("save_bucket",save_Bucket)
        if save_Bucket:
            grad_dict = {}
            for name,grad_tensor in zip(param_names,grads):            
                # 将这个bucket的所有grad，按照name:grad_tensor的键对值形式保存进一个.pt文件里，日后备用
                # pt_file_name = f"R_{rank}_E_{epoch}_S_{step}_B_{idx}.pt"
                if grad_tensor is not None:
                    grad_dict[name] = grad_tensor  # .cpu()  # 先转 cpu，避免 GPU 阻塞
                else: # 一般情况下不会发生
                    print(f"[Rank {rank}] WARNING: Gradient for {name} is None")
                pass
            
                torch.save(grad_dict, save_path) # 分开保存
                # torch.save(flat_tensor,save_path) # 整体保存
            
    except Exception as e:
        print(f"[Rank {rank}] Error accessing bucket parameters: {e}")
        param_names = "ERROR!!!"
        
        
    # 保存调试信息：
    INFO = f"""
===========
[INFO]
rank: {rank}
epoch: {the_epoch}
step: {the_step}
bucket_idx: {idx}
    ---
contents:
{param_names}
===========
    """ 
    if the_epoch == 0 or 1: # 只保存前两个epoch的debug信息
            
        with open(f"{output_path}000_EG_Full_DEBUG_INFO_{rank}.txt","a") as DEBUG_FILE:
            DEBUG_FILE.write(INFO)
    

    # --- 原本的逻辑 ---
    return _allreduce_fut(process_group, bucket.buffer())

# HookedSFTTrainer类：

class HookedSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_data = []  # Store communication data
        self.hook_registered = False  # Track hook registration
        self.param_name_map = None
        self.checked = False
        
        # 一定有更好的方法解决这个问题
        self.epoch_step_config_0 = None
        self.epoch_step_config_1 = None
        self.output_path = None
        
    def training_step(
        self, model, inputs, num_items_in_batch=None
    ):
        # input args: model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        # --- DDP 钩子 ---
        if not self.checked:
            dist_check()
            self.checked = True
            
        if self.hook_registered == False: # initializing
            # print(model.module)
            print(f"Hooked??? --- {self.hook_registered}")
            # print(f"dist.is_initiallized --- {dist.is_initialized()}")
            # print(model.type)


        # Make sure allreduce_hook is defined or imported before using it
        # from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        if dist.is_initialized() and self.hook_registered == False:
            try:
                global param_name_map
                global epoch_step_config_0
                global epoch_step_config_1


                ###### debug info: #######
                try:
                    model_info = f'''
                model.type:
                {model.type}
                ====================================================================================
                model.module.type:
                {model.module.type}
                '''
                    file_path = os.path.join(self.output_path, f"001_model_info_rank_{dist.get_rank()}.txt")
                    with open(file_path, "a") as f:
                        f.write(model_info)
                    print("model structure saved to", file_path)
                except Exception as e:
                    print(f"model structure unable to save...\n{e}")

                
                param_name_map = {id(p): name for name, p in model.named_parameters()}
                self.param_name_map = param_name_map
                
                epoch_step_config_0 =  {"epoch":0,"step":0}   
                self.epoch_step_config_0 = epoch_step_config_0
                
                epoch_step_config_1 = {"epoch":0,"step":0}
                self.epoch_step_config_1 = epoch_step_config_1
                
                print("config initiallized!!!")
                print("registering HOOKS")
                model.register_comm_hook(state=None, hook=mod_allreduce_hook_EG)
                self.hook_registered = True
                print("HOOKED!!!")
            except Exception as e:
                print(f"Something bad happened: {e}")



                
        # --- 发现 ---
        # 经过试验，明确 self.model_wrapped才是我们需要处理的东西，用这个注册DDP钩子准备抓取数据！
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(f"self.model type in training_step: {type(self.model)}")
        #     print(f"self.model_wrapped type in training_step: {type(self.model_wrapped)}") # 已知这个才是我们要找的对象。
        #     # print(self.model == model)
        #     # print(self.model_wrapped == model)
        # 因此，_wrap_model就没必要修改了


        
        # ---调用本家的东西 --- 
        return super().training_step(model,inputs,num_items_in_batch)

def main(save_bucket = False,scaling = None,pioneer = False, output_dir_name = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results"):
    global save_Bucket
    global Scaling
    global Pioneer
    save_Bucket = save_bucket
    Scaling = scaling
    Pioneer = pioneer
    print(f"SAVING BUCKET???\n--{save_Bucket}")
    
    dataset = prepare_dataset(pioneer=pioneer)
    # 6. 准备训练器
    hooked_trainer = HookedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks = [EPOCH_STEP_HANDLER()]
    )
    hooked_trainer.output_path = f"/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results_{output_dir_name}/"
    print(f"results will be saved to\n{hooked_trainer.output_path}")
    # 7. 开始训练
    # if dist.get_rank() == 0: # 只在主进程打印信息
    print("Training begin...")
    hooked_trainer.train()
    hooked_trainer.save_model(args.output_dir)
    dist.destroy_process_group() # 结束分布式


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=float, default=None, required=False) # scaling的参数
    parser.add_argument("--save_bucket", action="store_true", default=False) #是否保存bucket
    parser.add_argument("--pioneer", action="store_true", default=False) # 用非常小的子集对进行新的feature测试
    parser.add_argument("--scaling_str", type=str, required=False, help="Original string of scaling (for dir name)", default=None)
    args_ = parser.parse_args()

    save_bucket = args_.save_bucket
    scaling = args_.scaling
    pioneer = args_.pioneer
    output_dir_name = args_.scaling_str


    main(save_bucket=save_bucket, scaling=scaling, pioneer=pioneer,output_dir_name = output_dir_name)

# # Gemma_SFT.py

# '''
# 基于Gemma_SFT.ipynb的代码，进行分布式训练的SFT Trainer。
# 跟笔记本相比，支持分布式训练
# 提供钩子抓取数据
# '''

# # 启动命令： torchrun --nproc_per_node=2 Gemma_SFT.py
# # "--scaling": scaling的参数
# # "--save_bucket": 是否保存bucket
# # "--pioneer": 挑选少部分子集进行测试

# # 0. 导包：
# import os
# from datasets import load_dataset
# from datasets import load_from_disk
# from PIL import Image
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
# from trl import SFTTrainer
# from dist_check import dist_check
# from trl import SFTConfig
# from transformers import DefaultFlowCallback # 导入默认的东西
# from transformers import TrainerCallback
# import torch.distributed as dist
# from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook 
# import csv
# import argparse



# # 0. 准备工具函数：
# def quantlization_fuct(flat_tensor:torch.Tensor,
#                        scaling:int = None,
#                        fp64_enable:bool = False):
#     '''
#     观察记录：
#     1. fp16的最高数字约为为6.5e5，也就意味着我们最好不要使用1e3及以上的tensor，不然就变成inf了(因为有情况下时会出现Xe2的数量级的)
#         但不知道为什么，经过测试后发现原来的scaling是可行的。
    
#     '''
#     global Pioneer
#     if Pioneer:
#         print(f"doing quantlization, scaling = {scaling}")
    
#     try:
#         if fp64_enable:
#             flat_tensor = flat_tensor.to(dtype=torch.float64)
            
#         quantilized = torch.round(flat_tensor * scaling) / scaling
#         if scaling is None:
#             quantilized = flat_tensor
#         return quantilized
    
#     except Exception as e:
#         raise e    

# # System message for the assistant
# system_message = "You are an expert product description writer for Amazon."

# # User prompt that combines the user query and the schema
# user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
# Only return description. The description should be SEO optimized and for a better mobile search experience.

# <PRODUCT>
# {product}
# </PRODUCT>

# <CATEGORY>
# {category}
# </CATEGORY>
# """
#     # Convert dataset to OAI messages
# def format_data(sample):
#     return {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text": system_message}],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": user_prompt.format(
#                             product=sample["Product Name"],
#                             category=sample["Category"],
#                         ),
#                     },
#                     {
#                         "type": "image",
#                         "image": sample["image"],
#                     },
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": sample["description"]}],
#             },
#         ],
#     }
# def process_vision_info(messages: list[dict]) -> list[Image.Image]:
#     image_inputs = []
#     # Iterate through each conversation
#     for msg in messages:
#         # Get content (ensure it's a list)
#         content = msg.get("content", [])
#         if not isinstance(content, list):
#             content = [content]

#         # Check each content element for images
#         for element in content:
#             if isinstance(element, dict) and (
#                 "image" in element or element.get("type") == "image"
#             ):
#                 # Get the image and convert to RGB
#                 if "image" in element:
#                     image = element["image"]
#                 else:
#                     image = element
#                 image_inputs.append(image.convert("RGB"))
#     return image_inputs

# # 1. 准备数据集
# def prepare_dataset(pioneer:bool = False):

#     # Load dataset from the hub/local disk
#     # dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
#     dataset_path = "/gpfsnyu/home/zg2598/datasets/philschmid_amazon-product-descriptions-vlm/"
#     full_dataset = load_from_disk(dataset_path)
#     dataset = full_dataset["train"]
#     if pioneer:
#         dataset = dataset.select(range(10))  # 只取前10个样本，用于功能测试
#     # Convert dataset to OAI messages
#     # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
#     dataset = [format_data(sample) for sample in dataset]

#     # if dist.get_rank() == 0: # 只在主进程打印信息
#     print(f"example data:\n{dataset[345]['messages'] if not pioneer else dataset[-1]['messages']}")


#     return dataset



# # 2. 准备模型
# # Hugging Face model id
# model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
# model_path = "/gpfsnyu/home/zg2598/Gemma/gemma-3-4b-pt/" # working on local
# processor_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/gemma-3-4b-it-processor/gemma-3-4b-it-processor" # using local processor
# # Check if GPU benefits from bfloat16
# if torch.cuda.get_device_capability()[0] < 8:
#     raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")



# # 修改过的版本：
# local_rank = int(os.environ.get("LOCAL_RANK", 0))  # torchrun会设置这个环境变量
# device_map = {"": local_rank}

# model_kwargs = dict(
#     attn_implementation="eager",
#     torch_dtype=torch.bfloat16,
#     device_map=device_map,  # ✅ 重点修复！
# )
# # 移除Lora 量化策略

# # Load model and tokenizer
# model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
# processor = AutoProcessor.from_pretrained(processor_path)

# # if dist.get_rank() == 0: # 只在主进程打印信息
# print("Model loaded, Processor loaded...\nDONE!!!")




# # 3. 准备SFTConfig和损失函数：


# args = SFTConfig(
#     output_dir="gemma-product-description",     # directory to save and repository id
#     num_train_epochs=1,                         # number of training epochs
#     per_device_train_batch_size=1,              # batch size per device during training
#     gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,                # use gradient checkpointing to save memory
#     optim="adamw_torch_fused",                  # use fused adamw optimizer
#     logging_steps=5,                            # log every 5 steps
#     save_strategy="epoch",                         # save checkpoint every epoch when doing actual experiment, but for debugging, we save nothing
#     learning_rate=2e-4,                         # learning rate, based on QLoRA paper
#     bf16=True,                                  # use bfloat16 precision
#     max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
#     warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
#     lr_scheduler_type="constant",               # use constant learning rate scheduler
    
#     push_to_hub=False,                           # don't push model to hub !!!!!!!!!!
    
#     report_to="tensorboard",                    # report metrics to tensorboard
#     gradient_checkpointing_kwargs={
#         "use_reentrant": False
#     },  # use reentrant checkpointing
#     dataset_text_field="",                      # need a dummy field for collator
#     dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
# )
# args.remove_unused_columns = False # important for collator

# # Create a data collator to encode text and image pairs
# def collate_fn(examples):
#     texts = []
#     images = []
#     for example in examples:
#         image_inputs = process_vision_info(example["messages"])
#         text = processor.apply_chat_template(
#             example["messages"], add_generation_prompt=False, tokenize=False
#         )
#         texts.append(text.strip())
#         images.append(image_inputs)

#     # Tokenize the texts and process the images
#     batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

#     # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
#     labels = batch["input_ids"].clone()

#     # Mask image tokens
#     image_token_id = [
#         processor.tokenizer.convert_tokens_to_ids(
#             processor.tokenizer.special_tokens_map["boi_token"]
#         )
#     ]
#     # Mask tokens for not being used in the loss computation
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     labels[labels == image_token_id] = -100
#     labels[labels == 262144] = -100

#     batch["labels"] = labels
#     return batch



# # 5. 准备装设了钩子的HookedSFTTrainer


# # EPOCH 和 STEP 怎么找：


# class EPOCH_STEP_HANDLER(TrainerCallback):
# # class EPOCH_STEP_HANDLER(DefaultFlowCallback):
#     def on_epoch_begin(self, args, state, control, **kwargs):
#         """
#         Event called at the beginning of an epoch.
#         """
#         global CURRENT_EPOCH
#         CURRENT_EPOCH = int(state.epoch or 0)
        
#         return super().on_epoch_begin(args, state, control, **kwargs)
        
    
#     def on_step_begin(self, args, state, control, **kwargs):
#         """
#         Event called at the beginning of a training step. If using gradient accumulation, one training step might take
#         several inputs.
#         """
#         global CURRENT_STEP
#         CURRENT_STEP = state.global_step
        
#         return super().on_step_begin(args, state, control, **kwargs)



# # DDP钩子：

# #           dummy hook： allreduce_hook (默认DDP钩子，不改变行为）


# #           mod_allreduce_hook: 添加读取和保存的信息：

# # --- helper function ---
# def _allreduce_fut(
#     process_group: dist.ProcessGroup, tensor: torch.Tensor
# ) -> torch.futures.Future[torch.Tensor]:
#     """Average the input gradient tensor by allreduce and returns a future."""
#     group_to_use = process_group if process_group is not None else dist.group.WORLD

#     # Apply the division first to avoid overflow, especially for FP16.
#     tensor.div_(group_to_use.size())

#     return (
#         dist.all_reduce(tensor, group=group_to_use, async_op=True)
#         .get_future()
#         .then(lambda fut: fut.value()[0])
#     )
    
# # --- hook本体 ---
# def mod_allreduce_hook_EG(
#     process_group: dist.ProcessGroup, bucket: dist.GradBucket
# ) -> torch.futures.Future[torch.Tensor]:
#     '''
#     由mod_allreduce_hook_base修改而来：
#     本体允许结合名为EPOCH_STEP_HANDLER的TrainerCallback，实现：
#         1. 知晓这个bucket的meta数据 {
#             1. rank
#             2. epoch
#             3. step
#             4. index
#         }
#         2. 记录每一个bucket里的内容
#         3. 根据save_Bucket变量决定是否保存GradBucket里面的数据
    
#     更新后支持：
#         1. 使用一个scaling参数对数据进行quantlization -- 2025年7月29日实现
#     '''
    
    
#     # --- 导入东西 --- 
#     global CURRENT_EPOCH
#     global CURRENT_STEP
#     global save_Bucket
#     global Scaling

#     # --- 缓冲区里的扁平向量 --- 
#     flat_tensor = bucket.buffer()
#     # --- 基本信息 --- 
#     global param_name_map
#     output_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results/"
    
#     # 1. 知道这个是哪个rank:
#     rank = dist.get_rank()
    
#     # 2. 知道这是这个batch(或者step)第几个bucket:
#     idx = bucket.index()
    
#     # 3. 知道存储的数据类型：
#     data_type = flat_tensor.dtype
    
#     # 4. 知道这个桶里面塞了什么？然后存下来！
#     params = bucket.parameters()  # List[Tensor]
#     grads = bucket.gradients()  # List[Tensor]，对应顺序应该和 params 一致 -- [已确认]

    
#     # 4.1 知道这个桶属于哪个step和epoch
#     the_epoch = CURRENT_EPOCH
#     the_step = CURRENT_STEP

    
#     #### DEBUGING ####
#     print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}")
#     ##################
    
    
#     ### 更新 ###
    
#     # 1. 量化
#     if Scaling is not None:
#         quantized = quantlization_fuct(flat_tensor=flat_tensor,
#                                        scaling=Scaling,
#                                        fp64_enable=False)
#         # set_buffer
#         bucket.set_buffer(quantized) # 2025年7月29日：测试量化后的表现
#     else:
#         print("No_scaling_happened")  
#     # 2. val2index
    
#     # 3. EG Encoding
    
    

    
    
#     # bucket.set_buffer(codes) # 将bucket的内容更改为EG encoding的结果: codes
    
#     ############
    
    
    
    
#     # 4.1.1:
    
#     # 文件名称：
#     file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
#     # 保存路径
#     save_dir = os.path.join(output_path, "COMMUNICATION_DATA_EG") # 更换了保存路径
#     os.makedirs(save_dir,exist_ok=True)
#     save_path = os.path.join(save_dir, file_name)
    

    
#     # 4.2 具体保存
#     try:
#         param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]

#         # print("save_bucket",save_Bucket)
#         if save_Bucket:
#             grad_dict = {}
#             for name,grad_tensor in zip(param_names,grads):            
#                 # 将这个bucket的所有grad，按照name:grad_tensor的键对值形式保存进一个.pt文件里，日后备用
#                 # pt_file_name = f"R_{rank}_E_{epoch}_S_{step}_B_{idx}.pt"
#                 if grad_tensor is not None:
#                     grad_dict[name] = grad_tensor  # .cpu()  # 先转 cpu，避免 GPU 阻塞
#                 else: # 一般情况下不会发生
#                     print(f"[Rank {rank}] WARNING: Gradient for {name} is None")
#                 pass
            
#                 torch.save(grad_dict, save_path) # 分开保存
#                 # torch.save(flat_tensor,save_path) # 整体保存
            
#     except Exception as e:
#         print(f"[Rank {rank}] Error accessing bucket parameters: {e}")
#         param_names = "ERROR!!!"
        
        
#     # 保存调试信息：
#     INFO = f"""
# ===========
# [INFO]
# rank: {rank}
# epoch: {the_epoch}
# step: {the_step}
# bucket_idx: {idx}
#     ---
# contents:
# {param_names}
# ===========
#     """ 
#     if the_epoch == 0 or 1: # 只保存前两个epoch的debug信息
            
#         with open(f"{output_path}000_EG_Full_DEBUG_INFO_{rank}.txt","a") as DEBUG_FILE:
#             DEBUG_FILE.write(INFO)
    

#     # --- 原本的逻辑 ---
#     return _allreduce_fut(process_group, bucket.buffer())

# # HookedSFTTrainer类：

# class HookedSFTTrainer(SFTTrainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.communication_data = []  # Store communication data
#         self.hook_registered = False  # Track hook registration
#         self.param_name_map = None
#         self.checked = False
        
#         # 一定有更好的方法解决这个问题
#         self.epoch_step_config_0 = None
#         self.epoch_step_config_1 = None
#         self.output_path = None
        
#     def training_step(
#         self, model, inputs, num_items_in_batch=None
#     ):
#         # input args: model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
#         # --- DDP 钩子 ---
#         if not self.checked:
#             dist_check()
#             self.checked = True
            
#         if self.hook_registered == False: # initializing
#             # print(model.module)
#             print(f"Hooked??? --- {self.hook_registered}")
#             # print(f"dist.is_initiallized --- {dist.is_initialized()}")
#             # print(model.type)


#         # Make sure allreduce_hook is defined or imported before using it
#         # from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
#         if dist.is_initialized() and self.hook_registered == False:
#             try:
#                 global param_name_map
#                 global epoch_step_config_0
#                 global epoch_step_config_1


#                 ###### debug info: #######
#                 try:
#                     model_info = f'''
#                 model.type:
#                 {model.type}
#                 ====================================================================================
#                 model.module.type:
#                 {model.module.type}
#                 '''
#                     file_path = os.path.join(self.output_path, f"001_model_info_rank_{dist.get_rank()}.txt")
#                     with open(file_path, "a") as f:
#                         f.write(model_info)
#                     print("model structure saved to", file_path)
#                 except Exception as e:
#                     print(f"model structure unable to save...\n{e}")

                
#                 param_name_map = {id(p): name for name, p in model.named_parameters()}
#                 self.param_name_map = param_name_map
                
#                 epoch_step_config_0 =  {"epoch":0,"step":0}   
#                 self.epoch_step_config_0 = epoch_step_config_0
                
#                 epoch_step_config_1 = {"epoch":0,"step":0}
#                 self.epoch_step_config_1 = epoch_step_config_1
                
#                 print("config initiallized!!!")
#                 print("registering HOOKS")
#                 model.register_comm_hook(state=None, hook=mod_allreduce_hook_EG)
#                 self.hook_registered = True
#                 print("HOOKED!!!")
#             except Exception as e:
#                 print(f"Something bad happened: {e}")



                
#         # --- 发现 ---
#         # 经过试验，明确 self.model_wrapped才是我们需要处理的东西，用这个注册DDP钩子准备抓取数据！
#         # if dist.is_initialized() and dist.get_rank() == 0:
#         #     print(f"self.model type in training_step: {type(self.model)}")
#         #     print(f"self.model_wrapped type in training_step: {type(self.model_wrapped)}") # 已知这个才是我们要找的对象。
#         #     # print(self.model == model)
#         #     # print(self.model_wrapped == model)
#         # 因此，_wrap_model就没必要修改了


        
#         # ---调用本家的东西 --- 
#         return super().training_step(model,inputs,num_items_in_batch)

# def main(save_bucket = False,scaling = None,pioneer = False):
#     global save_Bucket
#     global Scaling
#     global Pioneer
#     save_Bucket = save_bucket
#     Scaling = scaling
#     Pioneer = pioneer
#     print(f"SAVING BUCKET???\n--{save_Bucket}")
    
#     dataset = prepare_dataset(pioneer=pioneer)
#     # 6. 准备训练器
#     hooked_trainer = HookedSFTTrainer(
#         model=model,
#         args=args,
#         train_dataset=dataset,
#         processing_class=processor,
#         data_collator=collate_fn,
#         callbacks = [EPOCH_STEP_HANDLER()]
#     )
#     hooked_trainer.output_path = f"/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results_{str(Scaling)}/"
#     # 7. 开始训练
#     # if dist.get_rank() == 0: # 只在主进程打印信息
#     print("Training begin...")
#     hooked_trainer.train()
#     hooked_trainer.save_model(args.output_dir)
#     dist.destroy_process_group() # 结束分布式


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scaling", type=float, default=None, required=False) # scaling的参数
#     parser.add_argument("--save_bucket", action="store_true", default=False) #是否保存bucket
#     parser.add_argument("--pioneer", action="store_true", default=False) # 用非常小的子集对进行新的feature测试
#     args_ = parser.parse_args()

#     save_bucket = args_.save_bucket
#     scaling = args_.scaling
#     pioneer = args_.pioneer


    main(save_bucket=save_bucket, scaling=scaling, pioneer=pioneer)
# # Gemma_SFT.py

# '''
# 基于Gemma_SFT.ipynb的代码，进行分布式训练的SFT Trainer。
# 跟笔记本相比，支持分布式训练
# 提供钩子抓取数据
# '''

# # 启动命令： torchrun --nproc_per_node=2 Gemma_SFT.py


# # 0. 导包：
# import os
# from datasets import load_dataset
# from datasets import load_from_disk
# from PIL import Image
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
# from trl import SFTTrainer
# from dist_check import dist_check
# from trl import SFTConfig
# from transformers import DefaultFlowCallback # 导入默认的东西
# from transformers import TrainerCallback
# import torch.distributed as dist
# from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook 
# import csv


# # 0. 准备工具函数：
# def quantlization_fuct(flat_tensor:torch.Tensor,
#                        scaling:int = None,
#                        fp64_enable:bool = False):
#     '''
#     观察记录：
#     1. fp16的最高数字约为为6.5e5，也就意味着我们最好不要使用1e3及以上的tensor，不然就变成inf了(因为有情况下时会出现Xe2的数量级的)
#         但不知道为什么，经过测试后发现原来的scaling是可行的。
    
#     '''
#     # print(f"doing quantlization, scaling = {scaling}")
    
#     try:
#         if fp64_enable:
#             flat_tensor = flat_tensor.to(dtype=torch.float64)
            
#         quantilized = torch.round(flat_tensor * scaling) / scaling
#         if scaling is None:
#             quantilized = flat_tensor
#         return quantilized
    
#     except Exception as e:
#         raise e    



# # 1. 准备数据集

# # System message for the assistant
# system_message = "You are an expert product description writer for Amazon."

# # User prompt that combines the user query and the schema
# user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
# Only return description. The description should be SEO optimized and for a better mobile search experience.

# <PRODUCT>
# {product}
# </PRODUCT>

# <CATEGORY>
# {category}
# </CATEGORY>
# """

# # Convert dataset to OAI messages
# def format_data(sample):
#     return {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text": system_message}],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": user_prompt.format(
#                             product=sample["Product Name"],
#                             category=sample["Category"],
#                         ),
#                     },
#                     {
#                         "type": "image",
#                         "image": sample["image"],
#                     },
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": sample["description"]}],
#             },
#         ],
#     }

# def process_vision_info(messages: list[dict]) -> list[Image.Image]:
#     image_inputs = []
#     # Iterate through each conversation
#     for msg in messages:
#         # Get content (ensure it's a list)
#         content = msg.get("content", [])
#         if not isinstance(content, list):
#             content = [content]

#         # Check each content element for images
#         for element in content:
#             if isinstance(element, dict) and (
#                 "image" in element or element.get("type") == "image"
#             ):
#                 # Get the image and convert to RGB
#                 if "image" in element:
#                     image = element["image"]
#                 else:
#                     image = element
#                 image_inputs.append(image.convert("RGB"))
#     return image_inputs

# # Load dataset from the hub/local disk
# # dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
# dataset_path = "/gpfsnyu/home/zg2598/datasets/philschmid_amazon-product-descriptions-vlm/"
# full_dataset = load_from_disk(dataset_path)
# dataset = full_dataset["train"]
# # Convert dataset to OAI messages
# # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
# dataset = [format_data(sample) for sample in dataset]

# # if dist.get_rank() == 0: # 只在主进程打印信息
# print(f"example data:\n{dataset[345]['messages']}")

# # 2. 准备模型


# # Hugging Face model id
# model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
# model_path = "/gpfsnyu/home/zg2598/Gemma/gemma-3-4b-pt/" # working on local
# processor_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/gemma-3-4b-it-processor/gemma-3-4b-it-processor" # using local processor
# # Check if GPU benefits from bfloat16
# if torch.cuda.get_device_capability()[0] < 8:
#     raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")


# ## 原来的版本
# # # Define model init arguments
# # model_kwargs = dict(
# #     attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
# #     torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
# #     device_map="auto", # Let torch decide how to load the model
# # )

# # # BitsAndBytesConfig int-4 config
# # model_kwargs["quantization_config"] = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
# #     bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
# # )


# # 修改过的版本：
# local_rank = int(os.environ.get("LOCAL_RANK", 0))  # torchrun会设置这个环境变量
# device_map = {"": local_rank}

# model_kwargs = dict(
#     attn_implementation="eager",
#     torch_dtype=torch.bfloat16,
#     device_map=device_map,  # ✅ 重点修复！
# )
# # 移除Lora 量化策略
# # model_kwargs["quantization_config"] = BitsAndBytesConfig(
# #     load_in_8bit=True,
# #     llm_int8_threshold=6.0,
# #     llm_int8_skip_modules=None,
# #     llm_int8_enable_fp32_cpu_offload=True,
# # )



# # Load model and tokenizer
# model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
# processor = AutoProcessor.from_pretrained(processor_path)

# # if dist.get_rank() == 0: # 只在主进程打印信息
# print("Model loaded, Processor loaded...\nDONE!!!")




# # 3. 准备SFTConfig和损失函数：


# args = SFTConfig(
#     output_dir="gemma-product-description",     # directory to save and repository id
#     num_train_epochs=1,                         # number of training epochs
#     per_device_train_batch_size=1,              # batch size per device during training
#     gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,                # use gradient checkpointing to save memory
#     optim="adamw_torch_fused",                  # use fused adamw optimizer
#     logging_steps=5,                            # log every 5 steps
#     save_strategy="no",                         # save checkpoint every epoch when doing actual experiment, but for debugging, we save nothing
#     learning_rate=2e-4,                         # learning rate, based on QLoRA paper
#     bf16=True,                                  # use bfloat16 precision
#     max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
#     warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
#     lr_scheduler_type="constant",               # use constant learning rate scheduler
    
#     push_to_hub=False,                           # don't push model to hub !!!!!!!!!!
    
#     report_to="tensorboard",                    # report metrics to tensorboard
#     gradient_checkpointing_kwargs={
#         "use_reentrant": False
#     },  # use reentrant checkpointing
#     dataset_text_field="",                      # need a dummy field for collator
#     dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
# )
# args.remove_unused_columns = False # important for collator

# # Create a data collator to encode text and image pairs
# def collate_fn(examples):
#     texts = []
#     images = []
#     for example in examples:
#         image_inputs = process_vision_info(example["messages"])
#         text = processor.apply_chat_template(
#             example["messages"], add_generation_prompt=False, tokenize=False
#         )
#         texts.append(text.strip())
#         images.append(image_inputs)

#     # Tokenize the texts and process the images
#     batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

#     # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
#     labels = batch["input_ids"].clone()

#     # Mask image tokens
#     image_token_id = [
#         processor.tokenizer.convert_tokens_to_ids(
#             processor.tokenizer.special_tokens_map["boi_token"]
#         )
#     ]
#     # Mask tokens for not being used in the loss computation
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     labels[labels == image_token_id] = -100
#     labels[labels == 262144] = -100

#     batch["labels"] = labels
#     return batch



# # 5. 准备装设了钩子的HookedSFTTrainer


# # EPOCH 和 STEP 怎么找：


# class EPOCH_STEP_HANDLER(TrainerCallback):
# # class EPOCH_STEP_HANDLER(DefaultFlowCallback):
#     def on_epoch_begin(self, args, state, control, **kwargs):
#         """
#         Event called at the beginning of an epoch.
#         """
#         global CURRENT_EPOCH
#         CURRENT_EPOCH = int(state.epoch or 0)
        
#         return super().on_epoch_begin(args, state, control, **kwargs)
        
    
#     def on_step_begin(self, args, state, control, **kwargs):
#         """
#         Event called at the beginning of a training step. If using gradient accumulation, one training step might take
#         several inputs.
#         """
#         global CURRENT_STEP
#         CURRENT_STEP = state.global_step
        
#         return super().on_step_begin(args, state, control, **kwargs)



# # DDP钩子：

# #           dummy hook： allreduce_hook (默认DDP钩子，不改变行为）


# #           mod_allreduce_hook: 添加读取和保存的信息：

# # --- helper function ---
# def _allreduce_fut(
#     process_group: dist.ProcessGroup, tensor: torch.Tensor
# ) -> torch.futures.Future[torch.Tensor]:
#     """Average the input gradient tensor by allreduce and returns a future."""
#     group_to_use = process_group if process_group is not None else dist.group.WORLD

#     # Apply the division first to avoid overflow, especially for FP16.
#     tensor.div_(group_to_use.size())

#     return (
#         dist.all_reduce(tensor, group=group_to_use, async_op=True)
#         .get_future()
#         .then(lambda fut: fut.value()[0])
#     )
    
# # --- hook本体 ---
# def mod_allreduce_hook_EG(
#     process_group: dist.ProcessGroup, bucket: dist.GradBucket
# ) -> torch.futures.Future[torch.Tensor]:
#     '''
#     由mod_allreduce_hook_base修改而来：
#     本体允许结合名为EPOCH_STEP_HANDLER的TrainerCallback，实现：
#         1. 知晓这个bucket的meta数据 {
#             1. rank
#             2. epoch
#             3. step
#             4. index
#         }
#         2. 记录每一个bucket里的内容
#         3. 根据save_Bucket变量决定是否保存GradBucket里面的数据
    
#     更新后支持：
#         1. 使用一个scaling参数对数据进行quantlization -- 2025年7月29日实现
#     '''
    
    
#     # --- 导入东西 --- 
#     global CURRENT_EPOCH
#     global CURRENT_STEP
#     global save_Bucket
#     global Scaling
#     # --- 缓冲区里的扁平向量 --- 
#     flat_tensor = bucket.buffer()
#     # --- 基本信息 --- 
#     global param_name_map
#     output_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results/"
    
#     # 1. 知道这个是哪个rank:
#     rank = dist.get_rank()
    
#     # 2. 知道这是这个batch(或者step)第几个bucket:
#     idx = bucket.index()
    
#     # 3. 知道存储的数据类型：
#     data_type = flat_tensor.dtype
    
#     # 4. 知道这个桶里面塞了什么？然后存下来！
#     params = bucket.parameters()  # List[Tensor]
#     grads = bucket.gradients()  # List[Tensor]，对应顺序应该和 params 一致 -- [已确认]

    
#     # 4.1 知道这个桶属于哪个step和epoch
#     the_epoch = CURRENT_EPOCH
#     the_step = CURRENT_STEP

    
#     #### DEBUGING ####
#     print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}")
#     ##################
    
    
#     ### 更新 ###
    
#     # 1. 量化
#     if Scaling is not None:
#         quantized = quantlization_fuct(flat_tensor=flat_tensor,
#                                        scaling=Scaling,
#                                        fp64_enable=False)
#     else:
#         quantized = flat_tensor
        
#     # 2. val2index
    
#     # 3. EG Encoding
    
    
#     # set_buffer
#     bucket.set_buffer(quantized) # 2025年7月29日：测试量化后的表现
    
    
#     # bucket.set_buffer(codes) # 将bucket的内容更改为EG encoding的结果: codes
    
#     ############
    
    
    
    
#     # 4.1.1:
    
#     # 文件名称：
#     file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
#     # 保存路径
#     save_dir = os.path.join(output_path, "COMMUNICATION_DATA_EG") # 更换了保存路径
#     os.makedirs(save_dir,exist_ok=True)
#     save_path = os.path.join(save_dir, file_name)
    

    
#     # 4.2 具体保存
#     try:
#         param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]

#         # print("save_bucket",save_Bucket)
#         if save_Bucket:
#             grad_dict = {}
#             for name,grad_tensor in zip(param_names,grads):            
#                 # 将这个bucket的所有grad，按照name:grad_tensor的键对值形式保存进一个.pt文件里，日后备用
#                 # pt_file_name = f"R_{rank}_E_{epoch}_S_{step}_B_{idx}.pt"
#                 if grad_tensor is not None:
#                     grad_dict[name] = grad_tensor  # .cpu()  # 先转 cpu，避免 GPU 阻塞
#                 else: # 一般情况下不会发生
#                     print(f"[Rank {rank}] WARNING: Gradient for {name} is None")
#                 pass
            
#                 torch.save(grad_dict, save_path) # 分开保存
#                 # torch.save(flat_tensor,save_path) # 整体保存
            
#     except Exception as e:
#         print(f"[Rank {rank}] Error accessing bucket parameters: {e}")
#         param_names = "ERROR!!!"
        
        
#     # 保存调试信息：
#     INFO = f"""
# ===========
# [INFO]
# rank: {rank}
# epoch: {the_epoch}
# step: {the_step}
# bucket_idx: {idx}
#     ---
# contents:
# {param_names}
# ===========
#     """ 
#     if the_epoch == 0 or 1: # 只保存前两个epoch的debug信息
            
#         with open(f"{output_path}000_EG_Full_DEBUG_INFO_{rank}.txt","a") as DEBUG_FILE:
#             DEBUG_FILE.write(INFO)
    

#     # --- 原本的逻辑 ---
#     return _allreduce_fut(process_group, bucket.buffer())

# # HookedSFTTrainer类：

# class HookedSFTTrainer(SFTTrainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.communication_data = []  # Store communication data
#         self.hook_registered = False  # Track hook registration
#         self.param_name_map = None
#         self.checked = False
        
#         # 一定有更好的方法解决这个问题
#         self.epoch_step_config_0 = None
#         self.epoch_step_config_1 = None
#         self.output_path = None
        
#     def training_step(
#         self, model, inputs, num_items_in_batch=None
#     ):
#         # input args: model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
#         # --- DDP 钩子 ---
#         if not self.checked:
#             dist_check()
#             self.checked = True
            
#         if self.hook_registered == False: # initializing
#             # print(model.module)
#             print(f"Hooked??? --- {self.hook_registered}")
#             # print(f"dist.is_initiallized --- {dist.is_initialized()}")
#             # print(model.type)


#         # Make sure allreduce_hook is defined or imported before using it
#         # from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
#         if dist.is_initialized() and self.hook_registered == False:
#             try:
#                 global param_name_map
#                 global epoch_step_config_0
#                 global epoch_step_config_1


#                 ###### debug info: #######
#                 try:
#                     model_info = f'''
#                 model.type:
#                 {model.type}
#                 ====================================================================================
#                 model.module.type:
#                 {model.module.type}
#                 '''
#                     file_path = os.path.join(self.output_path, f"001_model_info_rank_{dist.get_rank()}.txt")
#                     with open(file_path, "a") as f:
#                         f.write(model_info)
#                     print("model structure saved to", file_path)
#                 except Exception as e:
#                     print(f"model structure unable to save...\n{e}")

                
#                 param_name_map = {id(p): name for name, p in model.named_parameters()}
#                 self.param_name_map = param_name_map
                
#                 epoch_step_config_0 =  {"epoch":0,"step":0}   
#                 self.epoch_step_config_0 = epoch_step_config_0
                
#                 epoch_step_config_1 = {"epoch":0,"step":0}
#                 self.epoch_step_config_1 = epoch_step_config_1
                
#                 print("config initiallized!!!")
#                 # # Write param_name_map to a CSV file
#                 # param_map_path = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/param_name_map_rank_{}.csv".format(dist.get_rank())
#                 # with open(param_map_path, "w", newline="") as csvfile:
#                 #     writer = csv.writer(csvfile)
#                 #     writer.writerow(["pid", "name"])
#                 #     for pid, name in param_name_map.items():
#                 #         writer.writerow([pid, name])

#                 # print(list(model.named_parameters()))
#                 print("registering HOOKS")
#                 model.register_comm_hook(state=None, hook=mod_allreduce_hook_EG)
#                 self.hook_registered = True
#                 print("HOOKED!!!")
#             except Exception as e:
#                 print(f"Something bad happened: {e}")



                
#         # --- 发现 ---
#         # 经过试验，明确 self.model_wrapped才是我们需要处理的东西，用这个注册DDP钩子准备抓取数据！
#         # if dist.is_initialized() and dist.get_rank() == 0:
#         #     print(f"self.model type in training_step: {type(self.model)}")
#         #     print(f"self.model_wrapped type in training_step: {type(self.model_wrapped)}") # 已知这个才是我们要找的对象。
#         #     # print(self.model == model)
#         #     # print(self.model_wrapped == model)
#         # 因此，_wrap_model就没必要修改了


        
#         # ---调用本家的东西 --- 
#         return super().training_step(model,inputs,num_items_in_batch)

# def main(save_bucket = False,scaling = None):
#     global save_Bucket
#     global Scaling
#     save_Bucket = save_bucket
#     Scaling = scaling
#     print(f"SAVING BUCKET???\n--{save_Bucket}")
    
#     # 6. 准备训练器
#     hooked_trainer = HookedSFTTrainer(
#         model=model,
#         args=args,
#         train_dataset=dataset,
#         processing_class=processor,
#         data_collator=collate_fn,
#         callbacks = [EPOCH_STEP_HANDLER()]
#     )
#     hooked_trainer.output_path = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/results/"
#     # 7. 开始训练
#     # if dist.get_rank() == 0: # 只在主进程打印信息
#     print("Training begin...")
#     hooked_trainer.train()
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     save_bucket = False
#     scalings = 1e6

#     main(save_bucket = save_bucket,
#          scaling = scaling)
    
