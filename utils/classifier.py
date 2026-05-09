import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def cargar_modelo(lora_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    model.to("cpu")
    return model, tokenizer

def clasificar_ticket(ticket: str, model, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a support ticket classifier. Respond only with severity, area and escalation."},
        {"role": "user", "content": ticket}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    respuesta = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return respuesta.strip()

def extraer_severidad(texto: str) -> str:
    match = re.search(r"P[1-4]", texto)
    return match.group(0) if match else "UNKNOWN"