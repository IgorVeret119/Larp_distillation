"""Shared teacher loading for pruning and distillation."""

from pathlib import Path

from safetensors.torch import load_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _teacher_path(args):
    path = Path(args.teacher_model_path).expanduser()
    if path.suffix.lower() == ".safetensors":
        if not path.is_file():
            raise FileNotFoundError(f"Файл весов учителя не найден: {path}")
    elif path.is_file():
        raise ValueError("Отдельный файл учителя должен иметь расширение .safetensors")
    return path


def _resource_source(args, option):
    path = _teacher_path(args)
    source = getattr(args, option, None)
    if source:
        source_path = Path(source).expanduser()
        return str(source_path) if source_path.exists() else str(source)
    if path.is_file():
        return str(path.parent)
    return str(path) if path.is_dir() else args.teacher_model_path


def _local_only(args, source):
    return bool(getattr(args, "teacher_local_files_only", False)) or Path(source).exists()


def load_teacher_config(args):
    source = _resource_source(args, "teacher_config_path")
    try:
        return AutoConfig.from_pretrained(
            source, local_files_only=_local_only(args, source)
        )
    except OSError as exc:
        raise OSError(
            f"Не удалось загрузить конфигурацию учителя из {source}. "
            "Для файла .safetensors положите config.json рядом с весами "
            "или задайте teacher_config_path."
        ) from exc


def load_teacher_tokenizer(args, **kwargs):
    source = _resource_source(args, "teacher_tokenizer_path")
    try:
        return AutoTokenizer.from_pretrained(
            source, local_files_only=_local_only(args, source), **kwargs
        )
    except OSError as exc:
        raise OSError(
            f"Не удалось загрузить токенизатор учителя из {source}. "
            "Положите файлы токенизатора рядом с весами "
            "или задайте teacher_tokenizer_path."
        ) from exc


def load_teacher_model(args, torch_dtype, device=None):
    path = _teacher_path(args)
    config = load_teacher_config(args)
    if path.is_file():
        # load_model understands shared/tied weights omitted by save_pretrained.
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        try:
            load_model(model, str(path), strict=True, device="cpu")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Веса {path} не соответствуют конфигурации учителя. "
                "Нужен полный checkpoint Transformers; для модели, разбитой "
                "на части, укажите папку с model.safetensors.index.json."
            ) from exc
        model.eval()
    else:
        source = str(path) if path.is_dir() else args.teacher_model_path
        model = AutoModelForCausalLM.from_pretrained(
            source,
            config=config,
            torch_dtype=torch_dtype,
            local_files_only=_local_only(args, source),
        )
    if device is not None:
        model.to(device)
    return model
