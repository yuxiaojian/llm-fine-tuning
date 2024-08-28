"""
Original file is located at https://colab.research.google.com/drive/1NuloNJhx1hkQoM5LqNBAiLwUxQCkl7T0

"""

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Prepare data

from datasets import load_dataset
dataset = load_dataset("lilacai/glaive-function-calling-v2-sharegpt", split="train")

import json

def format_sharegpt_conversations(examples):
    conversations = examples["conversations"]
    formatted_texts = []

    for conversation in conversations:
        formatted_text = ""
        system_message = ""
        tools = ""

        # Extract system message and tools if present
        if conversation[0]["from"] == "system":
            system_content = conversation[0]["value"]
            system_parts = system_content.split("Use them if required -\n")
            system_message = system_parts[0].strip()
            if len(system_parts) > 1:
                tools =  system_parts[1].strip()
            conversation = conversation[1:]  # Remove system message from conversation

        # Add system message and tools if present
        if system_message or tools:
            formatted_text += "<|start_header_id|>system<|end_header_id|>\n\n"
            formatted_text += f"{system_message}\n"
            if tools:
                formatted_text += "You are provided with function signatures within <tools></tools> XML tags. "
                formatted_text += "You may call one or more functions to assist with the user query. "
                formatted_text += "Don't make assumptions about what values to plug into functions. "
                formatted_text += "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
                formatted_text += "<tool_call>\n"
                formatted_text += '{"name": <function-name>,"arguments": <args-dict>}\n'
                formatted_text += "</tool_call>\n"
                formatted_text += f"Here are the available tools:\n<tools> {tools} </tools>"
            formatted_text += "<|eot_id|>"

        for i, turn in enumerate(conversation):
            role = turn["from"]
            content = turn["value"]
            last = i == len(conversation) - 1

            if role == "human":
                formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                if last:
                    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            elif role == "gpt":
                formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                if '<functioncall>' in content:
                    function_call = content.split('<functioncall>')[1].split('<|endoftext|>')[0].strip()
                    formatted_text += f"<tool_call>{function_call}</tool_call><|eot_id|>"

                else:
                    formatted_text += f"{content.split('<|endoftext|>')[0].strip()}<|eot_id|>"

            elif role == "tool":
                formatted_text += "<|start_header_id|>ipython<|end_header_id|>\n\n"
                formatted_text += f"{json.dumps({'result': content})}<|eot_id|>"
                if last:
                    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        formatted_texts.append(formatted_text.strip())

    return {"formatted_text": formatted_texts}
dataset = dataset.map(format_sharegpt_conversations, batched = True,)



from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "formatted_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# Start Fine-tuning job
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")

# Save to 8bit Q8_0
if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = [ "q8_0"])


