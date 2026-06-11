import os
from dataclasses import asdict
from pathlib import Path

import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import(
    Trainer,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)

from arguments import ModelArguments, DataArguments, TrainingArguments, QLoraArguments, MODULE_KEYWORDS
from collator import LLaVAOnevisionDataCollator
from datasets import LazySupervisedDataset
from utils import find_all_linear_names, save_model, rank0_print

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, QLoraArguments)
    )
    model_args, data_args, training_args, qlora_args = parser.parse_args_into_dataclasses()

    output_dir = getattr(training_args, "output_dir", None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(training_args.to_dict(), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(qlora_args), open(args_dir / "qlora.yaml", "w"))

    compute_dtype = (
        torch.float16 if training_args.fp16
        else(torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # QLoRA quantization config
    rank0_print("Quantization for LLM enabled...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )

    # load model, processor
    rank0_print("Loading model, tokenizer, processor...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_args.model_local_path,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if training_args.use_flash_attn else "eager",
    )
    processor = AutoProcessor.from_pretrained(model_args.model_local_path)
    tokenizer = processor.tokenizer
    config = AutoConfig.from_pretrained(model_args.model_local_path)
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # freeze vision encoder
    if not training_args.train_vision_encoder:
        rank0_print("Vision encoder is frozen")
        model.vision_tower.requires_grad_(False)

    # freeze others(image_newline)
    rank0_print("image_newline is frozen")
    model.image_newline.requires_grad_(False)

    # LoRA setup
    named_modules = {n: m for n, m in model.named_modules()}

    # LLM: LoRA
    llm_keys = MODULE_KEYWORDS["llm"]
    lora_modules = find_all_linear_names(named_modules, llm_keys)
    rank0_print(f"LoRA applied to {len(lora_modules)} LLM linear layers")

    # vision projector: full fine-tuning

    projector_keys = MODULE_KEYWORDS["vision_projector"]
    full_modules = list(projector_keys)
    rank0_print(f"Vision projector will be fully trained: {full_modules}")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

    for key in projector_keys:
        for name, param in model.named_parameters():
            if key in name:
                param.data = param.data.to(torch.bfloat16)
                param.requires_grad_(True)

    lora_config = LoraConfig(
        r=qlora_args.lora_r,
        lora_alpha=qlora_args.lora_alpha,
        target_modules=lora_modules,
        modules_to_save=[],   
        lora_dropout=qlora_args.lora_dropout,
        bias=qlora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    # print trainable parameters
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # load datatsets
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=None,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=None,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key,
        )
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"

    
    # collator
    data_collator = LLaVAOnevisionDataCollator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens,
    )

    #trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_state()

    if training_args.train_vision_projector:
        projector_state_dict = {}
        for name, param in trainer.model.named_parameters():
            if 'multi_modal_projector' in name and 'lora' not in name:
                projector_state_dict[name] = param.data.to(torch.bfloat16).cpu()
        torch.save(projector_state_dict, f"{output_dir}/projector.bin")

    save_model(trainer=trainer, output_dir=output_dir)

if __name__ == "__main__":
    train()