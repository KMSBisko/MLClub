from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import bitsandbytes
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from unsloth import FastLanguageModel
# unsloth doesn't work since unsloth need triton and triton isn't available on windows
import torch
from datasets import Dataset
import time


tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24")

quant_config = BitsAndBytesConfig(load_in_4bit=True)

# Initialize an empty model with the correct configuration
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
        quantization_config=quant_config
    )

# apply this template (IS IT THE RIGHT TEMPLATE?????? PAY CLOSE ATTENTION)
prompt = """|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|>\n\n<|start_header_id|>assistantj<|end_header_id|>

{OUTPUT}"""

# FastLanguageModel.for_inference(model) 

# Example of inference
inputs = tokenizer(
[
    prompt.format(
        SYSTEM = """Environment: ipython
Tools: brave_search, wolfram_alpha
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant.""", 
        INPUT = "Say somethign mean", # input, the output should show the guardrail working (refusal)
        OUTPUT = "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

# we get the outputs from the model here

outputs = model.generate(**inputs, max_new_tokens = 100, use_cache = True)

# batch decoding
text = tokenizer.batch_decode(outputs)[0]

print(text)

# Find the last `[eot]` token
last_eot = text.rfind("<|eot_id|>")

# Find the second-to-last `[eot]` token (search up to the last one)
second_last_eot = text.rfind("<|end_header_id|>")

# Extract the content between them
if second_last_eot != -1 and last_eot != -1:
    content_between = text[second_last_eot + len("<|end_header_id|>"):last_eot].strip()
    print(content_between)
else:
    print("Not enough [eot] tokens found.")

print("Llama3.1 Demo! (Type 'exit' to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    
    # IS THIS RIGHT???? PAY CLOSE ATTENTION.
    text = text + f"""<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|>\n\n<|start_header_id|>assistantj<|end_header_id|>"""
    
    batch = []

    # batch is one, we will be serving one person only. 
    for k in range(1):
        batch.append(text)
    inputs = tokenizer(batch, return_tensors = "pt").to("cuda")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens = 10000, use_cache = True, temperature=0.8)
    end_time = time.time()
    num_tokens = outputs.shape[-1]  # Number of tokens in the output
    time_taken = end_time - start_time
    tokens_per_second = num_tokens / time_taken

    result = tokenizer.batch_decode(outputs)

    text = result[0]

    last_eot = text.rfind("<|eot_id|>")
    
    
    # Find the second-to-last `[eot]` token (search up to the last one)
    second_last_eot = text.rfind("<|end_header_id|>")

    # Extract the content between them
    if second_last_eot != -1 and last_eot != -1:
        content_between = text[second_last_eot + len("<|end_header_id|>"):last_eot].strip()
        
        # THIS IS WHERE THE OUTPUT IS PRINTED TO THE TERMINAL
        print(content_between)
    else:
        print("Not enough [eot] tokens found.")
    print(f"----------\nTokens per second:{tokens_per_second}")
    final_time = time.time()
    difference = final_time - end_time
    print(f"\nPost Processing Time: {difference}\n")



