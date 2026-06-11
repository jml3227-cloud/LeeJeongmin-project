from typing import List, Dict
import torch
import torch.nn as nn
import transformers

def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if 'lm_head' in name:
            lora_module_names.remove(name)

    return list(lora_module_names)

def rank0_print(*args):
    print(*args)

def save_model(trainer: transformers.Trainer, output_dir: str):
    trainer.model.save_pretrained(output_dir)
    if trainer.args.should_save:
        projector_state_dict = {
            k: v.cpu() for k, v in trainer.model.state_dict().items()
            if 'multi_modal_projector' in k
        }
        torch.save(projector_state_dict, f"{output_dir}/projector.bin")