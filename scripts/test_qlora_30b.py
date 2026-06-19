#!/usr/bin/env python3
"""De-risk the miniF2F RL harness backward path: load Qwen3-30B-A3B in 4-bit
(QLoRA) + LoRA, run a real forward+backward, report peak GPU memory. If this fits
well under 96GB, vLLM fp8 (~30GB) for rollouts can coexist on the same card."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3-30B-A3B"
tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
print("loading 30B in 4-bit nf4 ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
print(f"after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.print_trainable_parameters()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
# a realistic agentic sequence length
ids = torch.randint(0, 150000, (2, 1024), device="cuda")
print("forward+backward ...", flush=True)
out = model(input_ids=ids, labels=ids)
out.loss.backward()
opt.step()
torch.cuda.synchronize()
print(f"loss={out.loss.item():.3f}", flush=True)
print(f"PEAK GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB (of ~96)", flush=True)
print("QLORA 30B BACKWARD OK", flush=True)
