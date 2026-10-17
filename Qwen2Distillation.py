import os
import torch
import torch.nn.functional as F
import argparse
import subprocess

from torch import nn
from typing import List, Optional
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    Qwen2ForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import json

from torch.utils.data import Dataset  

import yaml
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoConfig

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_length)


class Qwen2ForDistillation(Qwen2ForCausalLM):
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        config,
        norm_factor,
        num_loss_layers: int,      # [ДОБАВЛЕНО] Количество слоев, по которым мы будем считать ошибку (MSE)
        teacher_mapping: list      # [ДОБАВЛЕНО] Готовый список, показывающий, какие слои учителя остались
    ):
        super().__init__(config)

        self.norm_factor = norm_factor
        
        # [ДОБАВЛЕНО] Сохраняем переменные для использования внутри метода forward
        self.num_loss_layers = num_loss_layers
        self.teacher_mapping = teacher_mapping
        
        # Задаем обучаемые слои (в данном пайплайне обучаются все оставшиеся слои студента)
        self.distil_layers = teacher_mapping

        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=torch.bfloat16
        )

        # Замораживаем веса учителя — он только выдает эталонные ответы (logits) и скрытые состояния
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Загружаем обрезанную модель (студента)
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # Сначала замораживаем все слои студента
        for param in self.student.parameters():
            param.requires_grad = False

        # Размораживаем только те слои, которые остались после прунинга, чтобы они могли обучаться
        for student_idx, teacher_idx in enumerate(self.distil_layers):
            for param in self.student.model.layers[student_idx].parameters():
                param.requires_grad = True

        total_teacher_layers = self.teacher.config.num_hidden_layers
        
        # [ИЗМЕНЕНО] Автоматически вычисляем, какие слои были удалены, чтобы передать это в старые функции (если нужно)
        self.removed_layers_iterations = [
            i for i in range(total_teacher_layers) 
            if i not in self.distil_layers
        ]

        self.student.train()
        self.hidden_loss_fn = nn.MSELoss() # Функция потерь для выравнивания скрытых состояний

        self.val_student_hs_losses = []
        self.val_student_hs_losses_all_batches = []
        self.val_logits_losses = []
        self.val_logits_losses_all_batches = []
        self.val_last_layer_losses = []
        self.val_last_layer_losses_all_batches = []

    def get_student_to_teacher_mapping(self, removed_teacher_layers: list, total_teacher_layers: int) -> dict:
            teacher_layers_remaining = [i for i in range(total_teacher_layers) if i not in removed_teacher_layers]
            mapping = {student_idx: teacher_idx for student_idx, teacher_idx in enumerate(teacher_layers_remaining)}
            return mapping

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # 1. Прямой проход учителя (без градиентов)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )

        # 2. Прямой проход студента (с расчетом градиентов для обучения)
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
    
        teacher_hidden_states = teacher_outputs.hidden_states
        student_hidden_states = student_outputs.hidden_states
        total_teacher_layers = self.teacher.config.num_hidden_layers

        hidden_loss = 0.0
        
        # [ИЗМЕНЕНО] Удалили сложную и хрупкую ручную логику удаления слоев (removed_layers_iterations).
        # Вместо нее строим маппинг (словарь соответствия) напрямую из заранее вычисленного teacher_mapping.
        # Например: {0: 0, 1: 1, 2: 3, 3: 4} — значит 2-й слой учителя был вырезан.
        mapping = {student_idx: teacher_id for student_idx, teacher_id in enumerate(self.teacher_mapping)}
        
        # Берем все слои студента, кроме последнего (последний всегда обязательно участвует в loss отдельно)
        available_student_layers = list(mapping.keys())[:-1]
        
        # [ДОБАВЛЕНО] Динамический выбор слоев для расчета MSE.
        # Если в конфиге запросили больше слоев, чем есть, берем максимум возможного.
        num_loss_layers = min(self.num_loss_layers, len(available_student_layers))

        if num_loss_layers > 0:
            # Математика равномерного распределения: делим модель на N равных частей
            step = len(available_student_layers) / num_loss_layers
            # Берем индекс из середины каждого шага
            selected_indices = [int(i * step + step / 2) for i in range(num_loss_layers)]
            loss_layers = [available_student_layers[i] for i in selected_indices]
        else:
            loss_layers = []
            
        # 3. Расчет ошибки MSE (Hidden State Loss)
        # Штрафуем студента за то, что его промежуточные слои выдают векторы, отличные от слоев учителя
        for student_layer in loss_layers:
            teacher_layer = mapping[student_layer]
            hidden_loss += self.hidden_loss_fn(
                student_hidden_states[student_layer],
                teacher_hidden_states[teacher_layer]
            )
        
        # Последний слой всегда важен, его выравниваем обязательно
        hidden_loss += self.hidden_loss_fn(
            student_hidden_states[-1],
            teacher_hidden_states[-1]
        )
        
        # 4. Расчет ошибки KL-Дивергенции (Logits Loss)
        # Сравниваем вероятности слов, которые предсказывают обе модели.
        student_log_probs = F.log_softmax(student_outputs.logits, dim=-1)
        teacher_probs = F.softmax(teacher_outputs.logits, dim=-1)

        kl_div_loss = torch.sum(F.kl_div(student_log_probs, teacher_probs, reduction="none"))
        
        # Итоговая функция потерь — это комбинация KL-дивергенции предсказаний и MSE внутренних слоев.
        loss = kl_div_loss / 100 + hidden_loss


        if not self.training:
            # Во время валидации просто сохраняем значения ошибок для построения графиков
            logits_loss = torch.sum(F.kl_div(
                F.log_softmax(student_outputs.logits, dim=-1),
                F.softmax(teacher_outputs.logits, dim=-1),
                reduction='none'
            ))

            last_layer_loss = self.hidden_loss_fn(student_hidden_states[-1], teacher_hidden_states[-1])

            self.val_logits_losses_all_batches.append(logits_loss.detach().cpu().item())
            self.val_last_layer_losses_all_batches.append(last_layer_loss.detach().cpu().item())
    

        return CausalLMOutputWithPast(
            loss=loss,
            logits=student_outputs.logits,
            past_key_values=student_outputs.past_key_values,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )


class LossAverageCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        model = kwargs.get('model')
        if model is not None:
            if hasattr(model, 'val_student_hs_losses_all_batches') and model.val_student_hs_losses_all_batches:
                avg_loss = sum(model.val_student_hs_losses_all_batches) / len(model.val_student_hs_losses_all_batches)
                model.val_student_hs_losses.append(avg_loss)
                model.val_student_hs_losses_all_batches = []
            if hasattr(model, 'val_logits_losses_all_batches') and model.val_logits_losses_all_batches:
                avg_logits_loss = sum(model.val_logits_losses_all_batches) / len(model.val_logits_losses_all_batches)
                model.val_logits_losses.append(avg_logits_loss)
                model.val_logits_losses_all_batches = []
            if hasattr(model, 'val_last_layer_losses_all_batches') and model.val_last_layer_losses_all_batches:
                avg_last_layer_loss = sum(model.val_last_layer_losses_all_batches) / len(model.val_last_layer_losses_all_batches)
                model.val_last_layer_losses.append(avg_last_layer_loss)
                model.val_last_layer_losses_all_batches = []
        return control


def main(args):
    config = AutoConfig.from_pretrained(args.teacher_model_path)

    # [ДОБАВЛЕНО] Автоматическое чтение файла pruned_info.json.
    # Это позволяет скрипту дистилляции точно знать, какие слои вырезал auto_prune.py,
    # не требуя ручного ввода массивов в консоль.
    pruned_info_path = os.path.join(args.student_model_path, "pruned_info.json")
    if os.path.exists(pruned_info_path):
        with open(pruned_info_path, "r") as f:
            removed_layers = json.load(f)["removed_layers"]
        print(f"Удаленные слои: {removed_layers}")
    else:
        raise FileNotFoundError("pruned_info.json не найден! Сначала запустите auto_prune.py")

    # [ДОБАВЛЕНО] Автоматически генерируем карту слоев (массив индексов учителя, которые не были удалены)
    original_num_layers = AutoConfig.from_pretrained(args.teacher_model_path).num_hidden_layers
    teacher_mapping = [i for i in range(original_num_layers) if i not in removed_layers]

    # [ИЗМЕНЕНО] Передаем новые параметры в класс модели
    distillation_model = Qwen2ForDistillation(
        teacher_model_path=args.teacher_model_path,
        student_model_path=args.student_model_path,
        config=config,
        norm_factor=args.norm_factor, 
        num_loss_layers=args.num_loss_layers, # Количество точек замера MSE из конфига
        teacher_mapping=teacher_mapping       # Готовая карта слоев
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)

    # [ИЗМЕНЕНО] Избавляемся от жестко зашитых путей (хардкода).
    # Теперь скрипт послушно берет пути из config.yaml.
    if args.use_local_data:
        train_file = args.local_train_file
        val_file = args.local_val_file

        def load_local_dataset(file_path, max_lines=10000000):
            documents = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    try:
                        record = json.loads(line)
                        if 'text' in record:
                            documents.append(record['text'].strip())
                    except json.JSONDecodeError:
                        continue
            return documents

        # Используем getattr для безопасного извлечения параметра (если его забыли указать в конфиге)
        train = load_local_dataset(train_file, max_lines=getattr(args, 'ds_frac', float('inf')))
        val = load_local_dataset(val_file, max_lines=3000)
        train_dataset = TokenizedDataset(train, tokenizer, args.maxlen)
        val_dataset = TokenizedDataset(val, tokenizer, args.maxlen)

    else:
        dataset = load_dataset(args.train_dataset)
        dataset = dataset['train']
        ds_size = len(dataset)//args.ds_frac
        dataset = dataset.select(range(ds_size))

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True,
                max_length=args.maxlen
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = tokenized_dataset['train']
        val_dataset = tokenized_dataset['test'].select(range(3000))


    data_collator = default_data_collator

    training_args_dict = {
        "eval_strategy": "steps",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine_with_restarts",
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "adamw_torch",
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        'report_to': 'tensorboard',
        'ddp_find_unused_parameters': True 
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        **training_args_dict
    )

    trainer = Trainer(
        model=distillation_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[LossAverageCallback()],
    )

    trainer.train()

    # Генерация и сохранение графиков обучения в папку output_dir
    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_student_hs_losses, label="HS Loss (Specified Layers)")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Hidden State Loss (Specified Layers)")
    plt.savefig(f"{args.output_dir}/hidden_state_losses_specified.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_last_layer_losses, label="HS Loss (Last Layers)")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Hidden State Loss (Last Layers)")
    plt.savefig(f"{args.output_dir}/hidden_state_losses_last.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(distillation_model.val_logits_losses, label="Logits Loss")
    plt.xlabel("Validation Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Logits Loss")
    plt.savefig(f"{args.output_dir}/logits_losses.png")
    plt.close()


    distillation_model.student.save_pretrained(
        os.path.join(args.output_dir, "student"), 
        torch_dtype=torch.bfloat16
    )
    tokenizer.save_pretrained(os.path.join(args.output_dir, "student"))

    print(F"Student parameters: {sum(p.numel() for p in distillation_model.student.parameters())}")



if __name__ == "__main__":
    # [ИЗМЕНЕНО] Удалили функцию parse_args(). Теперь все параметры загружаются
    # из центрального файла config.yaml. Это делает процесс прозрачным и автоматизированным.
    with open("config.yaml", "r", encoding="utf-8") as f:
        args = SimpleNamespace(**yaml.safe_load(f))

    print(f"Loading model from {args.teacher_model_path}...")
    
    # Передаем загруженные настройки в главную функцию
    main(args)