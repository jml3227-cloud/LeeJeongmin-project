from typing import Optional
from dataclasses import dataclass, field
import transformers

MODEL_HF_PATH = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

MODULE_KEYWORDS = {
    "vision_encoder": ["vision_tower"],
    "vision_projector": ["multi_modal_projector"],
    "llm": ["language_model"],
    "others": ["image_newline"]
}

@dataclass
class ModelArguments:
    model_local_path: Optional[str] = field(default=None)

    def __post_init__(self):
        self.model_hf_path: str = MODEL_HF_PATH
        if not self.model_local_path:
            self.model_local_path = self.model_hf_path

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    )
    image_folder: Optional[str] = field(default=None)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length."}
    )
    use_flash_attn: bool = field(default=True)
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=True)
    mask_question_tokens: bool = field(default=True)
    report_to: str = field(default="tensorboard")

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False

@dataclass
class QLoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.0)
    lora_bias: str = "none"