"""Merge a local PEFT adapter into a complete teacher model."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def merge_teacher(args):
    adapter_value = getattr(args, "adapter_model_path", None)
    output_value = getattr(args, "merged_teacher_path", None)
    if not adapter_value or not output_value:
        raise ValueError("Задайте adapter_model_path и merged_teacher_path в config.yaml")
    adapter = Path(adapter_value).expanduser().resolve()
    output = Path(output_value).expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Не найден {adapter / 'adapter_config.json'}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Не найден {adapter / 'adapter_model.safetensors'}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Папка результата должна быть пустой: {output}. "
            "Выберите новую merged_teacher_path. Для уже объединённой модели "
            "задайте merge_adapter: false и teacher_model_path с путём к ней."
        )
    with (adapter / "adapter_config.json").open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base_source = (
        getattr(args, "adapter_base_model_path", None)
        or adapter_config.get("base_model_name_or_path")
    )
    if not base_source:
        raise ValueError("Неизвестна базовая модель. Задайте adapter_base_model_path")

    # The training environment is needed only when a merge is requested.
    import torch
    from peft import PeftModel
    from teacher_loading import load_teacher_model, load_teacher_tokenizer

    base_args = SimpleNamespace(
        teacher_model_path=base_source,
        teacher_config_path=getattr(args, "teacher_config_path", None),
        teacher_local_files_only=getattr(args, "teacher_local_files_only", False),
    )
    tokenizer_source = getattr(args, "teacher_tokenizer_path", None)
    if not tokenizer_source:
        tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model")
        if any((adapter / name).is_file() for name in tokenizer_files):
            tokenizer_source = str(adapter)
    base_args.teacher_tokenizer_path = tokenizer_source
    # An explicit or saved adapter tokenizer takes precedence; malformed files
    # must fail rather than silently switching to a different vocabulary.
    tokenizer = load_teacher_tokenizer(base_args)

    print(f"Базовая модель: {base_source}")
    print(f"Адаптер: {adapter}")
    # Honour the exact base revision recorded during adapter training.
    revision = adapter_config.get("revision")
    base_model = load_teacher_model(
        base_args, torch_dtype=torch.float32,
        **({"revision": revision} if revision else {}),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False)
    if not hasattr(model, "merge_and_unload"):
        raise ValueError("Этот тип PEFT-адаптера не поддерживает объединение весов")
    merged = model.merge_and_unload(safe_merge=True)
    merged = merged.to(dtype=torch.bfloat16)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(output))
    print(f"Полная модель-учитель сохранена: {output}")
    return output


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Путь к YAML-конфигу")
    options = parser.parse_args()
    with open(options.config, encoding="utf-8") as handle:
        args = SimpleNamespace(**yaml.safe_load(handle))
    merge_teacher(args)


if __name__ == "__main__":
    main()
