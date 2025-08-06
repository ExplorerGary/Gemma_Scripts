# Gemma_SFT.py

'''
基于Gemma_SFT.ipynb的代码，进行分布式训练的SFT Trainer。而且是全参数微调 Full-Finetune
跟笔记本相比，支持分布式训练
提供钩子抓取数据
'''

# 启动命令： torchrun --nproc_per_node=2 Gemma_Full_SFT_test_truncate.py
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
import json

try:
    from .utilities.Truncate_Hook import make_Truncate_Hook
except:
    from utilities.Truncate_Hook import make_Truncate_Hook


BASE_RESULT_DIR = "/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/"

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
        self.Mode = None
        
        
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
                model.register_comm_hook(state=None, hook=make_Truncate_Hook(Mode = self.Mode))
                self.hook_registered = True
                print("HOOKED!!!")
            except Exception as e:
                print(f"Something bad happened: {e}")
        
        # ---调用本家的东西 --- 
        return super().training_step(model,inputs,num_items_in_batch)

def main(save_bucket = False,scaling = None,pioneer = False, output_dir_name = None, Mode = None):
    rank = dist.get_rank()
    global save_Bucket
    global Scaling
    global Pioneer
    save_Bucket = save_bucket
    Scaling = scaling
    Pioneer = pioneer
    date = time.strftime("%Y-%m-%d", time.localtime())
    print(f"SAVING BUCKET???\n--{save_Bucket}")
    
    dataset = prepare_dataset(pioneer=pioneer)
    
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

    if dist.get_rank() == 0: # 只在主进程打印信息
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


    
    
    
    # 3. 准备SFTConfig和损失函数：
    if output_dir_name is None:
        output_dir_name = "None"
    save_dir = os.path.join(BASE_RESULT_DIR,f"result_Full_{output_dir_name}")
    global OUTPUT_DIR 
    OUTPUT_DIR = os.path.join(save_dir,f"COMMUNICATION_LOG_{date}")
    # make statics collecting csv:

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

    hooked_trainer.Mode = Mode
    hooked_trainer.output_path = OUTPUT_DIR
    
    
    # 7. 开始训练
    # if dist.get_rank() == 0: # 只在主进程打印信息
    print("Training begin...")
    train_output = hooked_trainer.train()
    
    try:
        # 取当前 rank
        rank = dist.get_rank() if dist.is_initialized() else 0

        
        train_output_dir = os.path.join(os.path.dirname(__file__),"TRAINER_OUTPUT")
        os.makedirs(train_output_dir,exist_ok=True)
        date_str = time.strftime("%Y%m%d")
        jsonl_path = os.path.join(train_output_dir, f"Gemma_Full_{date_str}_rank{rank}.jsonl")
        
        # 构造记录
        record = {
            "rank": rank,
            "scaling": output_dir_name,  # 假设你在主函数中传进来的
            "global_step": train_output.global_step,
            "training_loss": train_output.training_loss,
            **train_output.metrics  # 合并 metrics 字典 
        }
        
        # 写入 jsonl（每条记录一行）
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"=========================\nTrainOutput saved to {jsonl_path}\n=========================\n")
    except Exception as e:
        print(f"TrainOutput unable to save: \n{e}\n")   
    dist.destroy_process_group() # 结束分布式


if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # 或 gloo/ccl/xla，根据你设备
            init_method="env://",  # torchrun 会自动设置 env
        )
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scaling", type=float, default=None, required=False) # scaling的参数
    parser.add_argument("--save_bucket", action="store_true", default=False) #是否保存bucket
    parser.add_argument("--pioneer", action="store_true", default=False) # 用非常小的子集对进行新的feature测试
    # parser.add_argument("--scaling_str", type=str, required=False, help="Original string of scaling (for dir name)", default=None)
    parser.add_argument("--Mode", type=str, default="fp12", required=True) # Mode的参数
    
    args_ = parser.parse_args()
    save_bucket = args_.save_bucket
    scaling = args_.scaling
    pioneer = args_.pioneer
    output_dir_name = args_.Mode
    Mode = args_.Mode

    main(save_bucket=save_bucket, 
         scaling=scaling, 
         pioneer=pioneer,
         output_dir_name = output_dir_name,
         Mode=Mode)