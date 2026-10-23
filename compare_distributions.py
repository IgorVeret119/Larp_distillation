"""Teacher-forced distribution comparison on held-out input/output records."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace


def read_examples(path, limit=None):
    with open(path, encoding="utf-8-sig") as handle:
        try:
            records = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            records = [json.loads(line) for line in handle if line.strip()]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError("Тестовая выборка должна содержать записи input/output")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_examples должен быть положительным целым числом или null")
        records = records[:limit]
    for i, record in enumerate(records):
        for field in ("input", "output"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Запись {i}: требуется непустая строка {field}")
    return records


def encode_example(tokenizer, record, template, max_length, append_eos):
    if template == "chat":
        user = [{"role": "user", "content": record["input"]}]
        prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": record["output"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
    elif template == "base":
        prompt = f"\n### Input:\n{record['input']}\n### Answer:"
        full = prompt + "\n" + record["output"]
    else:
        raise ValueError("prompt_template: используйте chat или base")
    if not full.startswith(prompt):
        raise ValueError("Шаблон полного диалога не начинается с шаблона запроса")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    mask = [end > len(prompt) for start, end in encoded["offset_mapping"]]
    if append_eos:
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("В токенизаторе не задан eos_token_id")
        if not ids or ids[-1] != eos:
            ids.append(eos)
            mask.append(True)
    if len(ids) > max_length:
        raise ValueError(
            f"Пример {record.get('index', '?')}: {len(ids)} токенов > max_length={max_length}. "
            "Увеличьте max_length: ответы не обрезаются автоматически."
        )
    positions = [i for i in range(1, len(ids)) if mask[i]]
    if not positions:
        raise ValueError("В примере нет токенов ответа для оценки")
    return ids, positions


def metric_sums(teacher_logits, student_logits, targets, chunk_size=32, details=None):
    import torch
    import torch.nn.functional as F

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Размеры распределений учителя и студента не совпадают")
    sums = dict(tokens=len(targets), teacher_entropy=0., student_entropy=0., cross_entropy=0., kl=0.,
                teacher_nll=0., student_nll=0., top1_agreement=0.,
                teacher_accuracy=0., student_accuracy=0.)
    for start in range(0, len(targets), chunk_size):
        stop = start + chunk_size
        lt = F.log_softmax(teacher_logits[start:stop].float(), dim=-1)
        ls = F.log_softmax(student_logits[start:stop].float(), dim=-1)
        p = lt.exp()
        labels = targets[start:stop].to(lt.device).unsqueeze(-1)
        token_values = {
            "teacher_entropy": -(p * lt).sum(-1),
            "student_entropy": -(ls.exp() * ls).sum(-1),
            "cross_entropy": -(p * ls).sum(-1),
            "kl": (p * (lt - ls)).sum(-1),
            "teacher_nll": -lt.gather(-1, labels).squeeze(-1),
            "student_nll": -ls.gather(-1, labels).squeeze(-1),
            "top1_agreement": (lt.argmax(-1) == ls.argmax(-1)).float(),
            "teacher_accuracy": (lt.argmax(-1) == labels.squeeze(-1)).float(),
            "student_accuracy": (ls.argmax(-1) == labels.squeeze(-1)).float(),
        }
        for key, values in token_values.items():
            sums[key] += values.sum().item()
        if details is not None:
            lists = {key: values.tolist() for key, values in token_values.items()}
            for i in range(len(labels)):
                details.append({key: values[i] for key, values in lists.items()})
    if not all(math.isfinite(v) for v in sums.values()):
        raise ValueError("В метриках обнаружены NaN/Infinity")
    return sums


def averages(sums):
    n = sums["tokens"]
    if n <= 0:
        raise ValueError("Нет оценённых токенов")
    return {key: value / n for key, value in sums.items() if key != "tokens"} | {"tokens": n}


def compare(config):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from teacher_loading import load_teacher_model, load_teacher_tokenizer
    from teacher_settings import resolve_teacher_args

    options = config["distribution_comparison"]
    records = read_examples(options["test_file"], options.get("max_examples"))
    candidates = options["models"]
    if candidates is None:
        candidates = [
            {"name": "pruned", "path": config["student_model_path"]},
            {"name": "distilled", "path": str(Path(config["output_dir"]) / "student")},
        ]
    if not candidates or len({m["name"] for m in candidates}) != len(candidates):
        raise ValueError("Задайте модели с уникальными именами в models")
    device = options.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    teacher_args = resolve_teacher_args(SimpleNamespace(**config))
    tokenizer = load_teacher_tokenizer(teacher_args, use_fast=True)
    examples = [encode_example(
        tokenizer, r, options.get("prompt_template", "chat"),
        options.get("max_length", 2048), options.get("append_eos", True),
    ) for r in records]
    teacher = load_teacher_model(teacher_args, torch_dtype=dtype, device=device).eval()
    report = {
        "schema_version": 2,
        "test_file": options["test_file"], "examples": len(examples),
        "teacher": teacher_args.teacher_model_path,
        "prompt_template": options.get("prompt_template", "chat"),
        "append_eos": options.get("append_eos", True),
        "eos_token_id": tokenizer.eos_token_id,
        "settings": options,
        "device": str(device), "dtype": str(dtype),
        "torch_version": torch.__version__,
        "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
        "units": "nats per answer token, temperature=1, token-weighted mean",
        "models": [],
    }
    for candidate in candidates:
        path = candidate["path"]
        other_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tokenizer.get_vocab() != other_tokenizer.get_vocab():
            raise ValueError(f"{path}: словарь/ID токенов отличаются от учителя")
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True,
        ).to(device).eval()
        totals = None
        per_example = []
        snapshots = []
        for record, (ids, positions) in zip(records, examples):
            with torch.inference_mode():
                inputs = torch.tensor([ids], device=device)
                selected = torch.tensor(positions, device=device)
                # Logits at t-1 predict the gold answer token at t.
                t_output = teacher(input_ids=inputs, use_cache=False).logits
                t_logits = t_output[0, selected - 1].cpu()
                del t_output
                s_output = model(input_ids=inputs, use_cache=False).logits
                s_logits = s_output[0, selected - 1].cpu()
                del s_output
                token_details = []
                values = metric_sums(t_logits, s_logits, inputs[0, selected].cpu(), details=token_details)
                for pos, detail in zip(positions, token_details):
                    detail.update(position=pos, target_id=ids[pos])
                # Actual probability distributions on shared contexts, not independent generations.
                if not per_example:
                    top_k = max(1, min(int(options.get("top_k", 10)), t_logits.shape[-1]))
                    count = max(0, int(options.get("snapshot_positions", 3)))
                    for j, pos in enumerate(positions[:count]):
                        tp = t_logits[j].float().softmax(-1)
                        sp = s_logits[j].float().softmax(-1)
                        union = set(tp.topk(top_k).indices.tolist()) | set(sp.topk(top_k).indices.tolist()) | {ids[pos]}
                        token_ids = sorted(union, key=lambda token: float(tp[token]), reverse=True)
                        snapshots.append({
                            "example_index": record.get("index", 0), "position": pos,
                            "target_id": ids[pos], "target_text": tokenizer.decode([ids[pos]]),
                            "tokens": [{"id": token, "text": tokenizer.decode([token]),
                                        "teacher_probability": float(tp[token]),
                                        "student_probability": float(sp[token])} for token in token_ids],
                            "teacher_other_mass": max(0., 1. - float(tp[token_ids].sum())),
                            "student_other_mass": max(0., 1. - float(sp[token_ids].sum())),
                        })
                del t_logits, s_logits
            if totals is None:
                totals = dict(values)
            else:
                for key, value in values.items():
                    totals[key] += value
            per_example.append({"index": record.get("index", len(per_example)),
                                **averages(values), "per_token": token_details})
            print(f"{candidate['name']}: {len(per_example)}/{len(examples)}", flush=True)
        result = {"name": candidate["name"], "path": path,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "summary": averages(totals), "per_example": per_example,
                  "distribution_snapshots": snapshots}
        report["models"].append(result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    output = Path(options.get("output_file", "distribution_comparison.json"))
    from distribution_report import save_report
    save_report(report, output, plots=options.get("save_histograms", True))
    print(f"Результаты сохранены: {output}")
    return report


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        compare(yaml.safe_load(handle))
