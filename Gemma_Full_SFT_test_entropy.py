# Gemma_SFT.py

'''
基于Gemma_SFT.ipynb的代码，进行分布式训练的SFT Trainer。而且是全参数微调 Full-Finetune
跟笔记本相比，支持分布式训练
提供钩子抓取数据
'''

# 启动命令： torchrun --nproc_per_node=2 Gemma_SFT.py
# "--scaling": scaling的参数
# "--save_bucket": 是否保存bucket
# "--pioneer": 挑选少部分子集进行测试

# 0. 导包：
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.algorithms.ddp_comm_hooks") # 忽略torch的警告

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
import time

BASE_RESULT_DIR = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/"
fieldnames = ["bucket_name","entropy"]
# 0. 准备工具函数：
def quantlization_fuct(flat_tensor:torch.Tensor,
                       scaling:float = None,
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
        subset_size = len(dataset) // 10  # 取 1/10 的数据
        dataset = dataset.select(range(subset_size))  # 只取前1/10个样本，用于功能测试
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
def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d

class EPOCH_STEP_HANDLER(TrainerCallback):
# class EPOCH_STEP_HANDLER(DefaultFlowCallback):
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        self.fieldnames = None
        self.csv_file = None
        self.writer = None

        
    def is_main_process(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process():
            return  # ⛔ 非主进程不写日志

        try:
            if logs is None:
                return

            # ✅ 你要求的固定开头部分
            logs = rewrite_logs(logs)
            the_output_dir = args.output_dir
            csv_dir = os.path.join(the_output_dir, "TRAINING_LOG")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, self.filename)

            # 当前 step 的字段
            current_keys = ["step", "time"] + list(logs.keys())

            # 初始化 CSV 写入器
            if self.writer is None:
                self.csv_file = open(csv_path, mode="w", newline="")
                self.fieldnames = current_keys
                self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
                self.writer.writeheader()

            # 写入当前 row
            row = {
                "step": state.global_step,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            for key in self.fieldnames:
                if key not in row:  # 跳过 step 和 time，因为已经加了
                    row[key] = logs.get(key, "")
            self.writer.writerow(row)
            self.csv_file.flush()

        except Exception as e:
            print(f"CSV Logging Error: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file:
            self.csv_file.close()

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
def cal_entropy(flat_tensor: torch.Tensor, sample_size: int = 1000000, seed:int = 42) -> float:
    global Pioneer
    if Pioneer:
        print("calculating entropy")
    
    with torch.no_grad():
        # If tensor size is smaller than sample_size, use the whole tensor
        if flat_tensor.numel() <= sample_size:
            sample = flat_tensor
        else:
            # Randomly sample indices with a fixed seed
            indices = torch.randperm(flat_tensor.numel(), generator=torch.Generator().manual_seed(seed))[:sample_size]
            sample = flat_tensor[indices]
            
        # Calculate entropy using sampled data
        unique_vals, counts = sample.unique(return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -torch.sum(probs * torch.log2(probs)).item()
    
    return entropy



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
    global Pioneer

    # --- 缓冲区里的扁平向量 --- 
    flat_tensor = bucket.buffer()
    # --- 基本信息 --- 
    global param_name_map
    global OUTPUT_DIR
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
    if Pioneer:
        print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}, dtype = {data_type}")
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

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    

    
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
        to_path = os.path.join(OUTPUT_DIR,f"000_EG_Full_DEBUG_INFO_{rank}.txt")
        with open(to_path,"a") as DEBUG_FILE:
            DEBUG_FILE.write(INFO)
    
    entropy = cal_entropy(flat_tensor)
    ans = {"bucket_name":file_name[:-3],"entropy":entropy}
    csv_path = os.path.join(OUTPUT_DIR,f"002_BUCKET_ENTROPY_rank{rank}.csv")
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(ans)
        
    if Pioneer:
        print(f"entropy = {entropy}")
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
                    os.makedirs(self.output_path,exist_ok=True)
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

def main(save_bucket = False,scaling = None,pioneer = False, output_dir_name = None):
    rank = dist.get_rank()
    global save_Bucket
    global Scaling
    global Pioneer
    save_Bucket = save_bucket
    Scaling = scaling
    Pioneer = pioneer
    print(f"SAVING BUCKET???\n--{save_Bucket}")
    
    dataset = prepare_dataset(pioneer=pioneer)
    
    # 3. 准备SFTConfig和损失函数：
    if output_dir_name is None:
        output_dir_name = "None"
    save_dir = os.path.join(BASE_RESULT_DIR,f"result_Full_{output_dir_name}")
    global OUTPUT_DIR 
    OUTPUT_DIR = os.path.join(save_dir,"COMMUNICATION_LOG")
    # make statics collecting csv:
    static_csv_path = os.path.join(OUTPUT_DIR,f"002_BUCKET_ENTROPY_rank{rank}.csv")
    with open(static_csv_path,mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    args = SFTConfig(
        output_dir=save_dir,     # directory to save and repository id
        num_train_epochs=1,                         # number of training epochs
        per_device_train_batch_size=1,              # batch size per device during training
        gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
        gradient_checkpointing=True,                # use gradient checkpointing to save memory
        optim="adamw_torch_fused",                  # use fused adamw optimizer
        logging_steps=1,                            # log every 1 steps
        logging_strategy="steps",
        save_strategy="epoch",                         # save checkpoint every epoch when doing actual experiment, but for debugging, we save nothing
        learning_rate=2e-4,                         # learning rate, based on QLoRA paper
        bf16=True,                                  # use bfloat16 precision
        max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",               # use constant learning rate scheduler
        
        push_to_hub=False,                           # don't push model to hub !!!!!!!!!!
        
        # report_to=["tensorboard","csv"],                    # report metrics to tensorboard and csv
        report_to="tensorboard",
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # use reentrant checkpointing
        dataset_text_field="",                      # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
    )
    args.remove_unused_columns = False # important for collator
    print(f"results will be saved to\n{args.output_dir}")
    
    # 6. 准备训练器
    hooked_trainer = HookedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks = [EPOCH_STEP_HANDLER()]
    )

    
    # 7. 开始训练
    # if dist.get_rank() == 0: # 只在主进程打印信息
    print("Training begin...")
    hooked_trainer.train()
    # hooked_trainer.save_model() # 2025年7月31日14:02:28修改
    dist.destroy_process_group() # 结束分布式


if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # 或 gloo/ccl/xla，根据你设备
            init_method="env://",  # torchrun 会自动设置 env
        )
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
