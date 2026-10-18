"""Read text records from JSON or JSONL without silently discarding data."""

import json
from itertools import islice
from pathlib import Path


def load_local_dataset(file_path, max_lines=None, text_field="text"):
    path = Path(file_path).expanduser()
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("dataset_text_field должен быть непустым именем поля")
    if max_lines is not None and max_lines != float("inf"):
        if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines <= 0:
            raise ValueError("Лимит примеров должен быть положительным целым числом")
    else:
        max_lines = None

    with path.open(encoding="utf-8-sig") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)

            def jsonl_records():
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}: некорректный JSON/JSONL, строка {line_number}: {exc.msg}"
                        ) from exc

            records = jsonl_records()
        else:
            if isinstance(data, list):
                records = iter(data)
            elif isinstance(data, dict):
                records = iter([data])
            else:
                raise ValueError(f"{path}: ожидается массив JSON или объекты JSONL")

        documents = []
        for index, record in enumerate(islice(records, max_lines), 1):
            if not isinstance(record, dict) or text_field not in record:
                fields = ", ".join(map(str, record.keys())) if isinstance(record, dict) else type(record).__name__
                raise ValueError(
                    f"{path}: запись {index} не содержит поле '{text_field}'. "
                    f"Доступные поля/тип: {fields}. "
                    "Укажите имя текстового поля в dataset_text_field в config.yaml."
                )
            text = record[text_field]
            if not isinstance(text, str):
                raise ValueError(
                    f"{path}: запись {index}, поле '{text_field}' должно быть строкой, "
                    f"получено {type(text).__name__}"
                )
            if text.strip():
                documents.append(text.strip())

    if not documents:
        raise ValueError(
            f"{path}: не найдено непустых текстов в поле '{text_field}'. "
            "Проверьте содержимое файла и dataset_text_field."
        )
    return documents
