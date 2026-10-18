"""Resolve the prepared teacher without modifying config.yaml on disk."""

from types import SimpleNamespace


def resolve_teacher_args(args):
    if not getattr(args, "merge_adapter", False):
        return args
    output = getattr(args, "merged_teacher_path", None)
    if not output:
        raise ValueError("При merge_adapter: true задайте merged_teacher_path")
    settings = vars(args).copy()
    settings.update(
        teacher_model_path=output,
        teacher_config_path=None,
        teacher_tokenizer_path=None,
        teacher_local_files_only=True,
    )
    return SimpleNamespace(**settings)
