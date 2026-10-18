import yaml
import os
import json
import torch
import numpy as np
from types import SimpleNamespace
from teacher_loading import load_teacher_model, load_teacher_tokenizer
from teacher_settings import resolve_teacher_args

# Глобальный словарь для временного хранения активаций слоя q_proj при расчете LRP
query_activation_data = {}

# ==========================================
# 1. Загрузка данных
# ==========================================
def load_local_dataset(file_path, max_lines=100):
    """
    Считывает локальный JSON-файл построчно. 
    Используется для быстрой загрузки небольшой части датасета (калибровки), 
    чтобы прогнать её через модель и измерить отклики слоев.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: # Ограничиваем количество строк для экономии времени и памяти
                break
            try:
                record = json.loads(line)
                if 'text' in record:
                    documents.append(record['text'].strip())
            except json.JSONDecodeError:
                continue
    return documents

# ==========================================
# 2. Метрика BI (Block Importance)
# ==========================================
def calculate_mean_block_importance(X_i, X_i_plus_1):
    """
    Считает косинусное сходство между выходами текущего слоя (X_i) и следующего (X_i_plus_1).
    Если сходство велико (близко к 1), значит слой почти не изменил данные, 
    и его BI (1 - сходство) будет близок к 0, что делает его кандидатом на удаление.
    """
    BI_scores = []
    X_i = X_i.squeeze(0)
    X_i_plus_1 = X_i_plus_1.squeeze(0)
    for row in range(X_i.shape[0]):
        row_X_i = X_i[row]
        row_X_i_plus_1 = X_i_plus_1[row]
        # Математика косинусного сходства: скалярное произведение делим на произведение норм
        numerator = row_X_i.T @ row_X_i_plus_1
        denominator = np.linalg.norm(row_X_i) * np.linalg.norm(row_X_i_plus_1)
        BI_scores.append(numerator / denominator)
    return 1 - np.mean(BI_scores)

def calculate_BI(model, tokenizer, dataset):
    """
    Прогоняет датасет через модель, извлекает скрытые состояния (hidden_states) 
    и вычисляет метрику BI для каждого слоя.
    """
    tokenizer.pad_token = tokenizer.eos_token
    BI_per_text = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad(): # Отключаем градиенты, так как мы только анализируем, а не обучаем
        for text in dataset:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            BI_scores = {}
            for i in range(model.config.num_hidden_layers):
                # Достаем векторы текущего слоя и следующего
                hs_i = outputs.hidden_states[i].detach().cpu().numpy().astype(np.float64)
                hs_i_plus_1 = outputs.hidden_states[i + 1].detach().cpu().numpy().astype(np.float64)
                BI_scores[i] = calculate_mean_block_importance(hs_i, hs_i_plus_1)
            BI_per_text.append(BI_scores)

    # Усредняем баллы по всем текстам датасета
    mean_BI_scores = {}
    for layer in range(model.config.num_hidden_layers):
        mean_BI_scores[layer] = np.mean([d[layer] for d in BI_per_text])
    return mean_BI_scores

# ==========================================
# 3. Метрика MSE (Mean Squared Error)
# ==========================================
def calculate_mean_mse_importance(X_i, X_i_plus_1):
    """
    Альтернативная метрика: считает прямое среднеквадратичное отклонение между соседними слоями.
    Чем меньше MSE, тем меньше слой вносит изменений.
    """
    mse_scores = []
    X_i = X_i.squeeze(0)
    X_i_plus_1 = X_i_plus_1.squeeze(0)
    for row in range(X_i.shape[0]):
        row_X_i = X_i[row]
        row_X_i_plus_1 = X_i_plus_1[row]
        mse_value = np.mean((row_X_i - row_X_i_plus_1) ** 2)
        mse_scores.append(mse_value)
    return np.mean(mse_scores)

def calculate_mse_importance(model, tokenizer, dataset):
    """
    Прогоняет датасет и собирает MSE баллы важности для всех слоев модели.
    """
    mse_scores_accum = {}
    num_texts = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for text in dataset:
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  
            for i in range(model.config.num_hidden_layers):
                hs_current = hidden_states[i].detach().cpu().numpy().astype(np.float64)
                hs_next = hidden_states[i+1].detach().cpu().numpy().astype(np.float64)
                mse = calculate_mean_mse_importance(hs_current, hs_next)
                mse_scores_accum[i] = mse_scores_accum.get(i, 0) + mse
            num_texts += 1

    if num_texts == 0:
        return {}
    return {layer: total / num_texts for layer, total in mse_scores_accum.items()}

# ==========================================
# 4. Метрика LRP (Layer-wise Relevance Propagation)
# ==========================================
def query_hook(module, input, output):
    """
    Хук PyTorch, который "перехватывает" данные, входящие и выходящие из слоя Q_proj (Query)
    механизма внимания в момент прямого прохода модели.
    """
    query_activation_data[module.layer_id] = {
         'input': input[0].detach(),      
         'output': output.detach(),         
         'module': module                   
    }

def register_query_hooks(model):
    """
    Развешивает хуки-шпионы на проекции Query во всех слоях модели.
    """
    for i, layer in enumerate(model.layers):
        query_layer = layer.self_attn.q_proj
        query_layer.layer_id = f"layer_{i}_q_proj"
        query_layer.register_forward_hook(query_hook)

def propagate_epsilon_linear(module, input_activations, output_relevance, epsilon=1e-6):
    """
    Реализует epsilon-правило для обратного распространения важности (relevance).
    Стабилизирует деление с помощью epsilon, предотвращая взрыв значений.
    """
    x = input_activations
    W = module.weight  
    z = torch.matmul(x, W.t()) 
    sign_z = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    z_stable = z + epsilon * sign_z
    s = output_relevance / (z_stable + 1e-9)  
    return x * torch.matmul(s, W)

def calculate_LRP_importance(model, tokenizer, dataset, epsilon=1e-6):
    """
    Оценивает важность слоя, прокидывая сигнал LRP через механизм внимания.
    """
    global query_activation_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    register_query_hooks(model)
    total_layer_relevance = {}
    n_texts = 0

    with torch.no_grad():
        for text in dataset:
            query_activation_data = {}
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            _ = model(**inputs) # Делаем прямой проход, хуки автоматически соберут данные
            
            # Анализируем собранные хуками данные
            for layer_id, data in query_activation_data.items():
                R_out = data['output'].abs()
                module_ref = data['module']
                R_in = propagate_epsilon_linear(module_ref, data['input'], R_out, epsilon)
                relevance_score = R_in.abs().sum().item()
                layer_num = int(layer_id.split('_')[1]) # Вытягиваем номер слоя
                total_layer_relevance[layer_num] = total_layer_relevance.get(layer_num, 0) + relevance_score
            
            n_texts += 1

    return {layer: total / n_texts for layer, total in total_layer_relevance.items()}

# ==========================================
# 5. Функция физической вырезки слоев
# ==========================================
def layer_removal(model, prune_list):
    """
    Выполняет структурное удаление выбранных слоев из архитектуры LLM.
    """
    model_layers = model.model.layers
    # Сортировка по убыванию крайне важна! Если удалять с конца, 
    # индексы оставшихся в начале слоев не сместятся.
    sorted_list = sorted(prune_list, reverse=True)
    
    # 1. Удаляем объекты слоев из памяти
    for prune_layer in sorted_list:
        del model_layers[prune_layer]

    # 2. Перенумеровываем оставшиеся слои, чтобы не было дырок в индексах
    for i, layer in enumerate(model_layers):
        if hasattr(layer, 'self_attn'):
            layer.self_attn.layer_idx = i
        
    # 3. Обновляем конфигурацию модели, указывая новое, уменьшенное число слоев
    model.config.num_hidden_layers -= len(prune_list)
    if hasattr(model.config, 'max_window_layers') and model.config.max_window_layers:
        model.config.max_window_layers -= len(prune_list)

    # 4. Чистим словарь весов (state_dict) от удаленных параметров
    state_dict = model.state_dict()
    updated_state_dict = {k: v for k, v in state_dict.items()
                          # Индекс [2] потому что ключи имеют вид: model.layers.X.weight
                          if "layers" not in k or int(k.split('.')[2]) not in prune_list}
    model.load_state_dict(updated_state_dict, strict=False)


# ==========================================
# 6. Основной управляющий пайплайн
# ==========================================
if __name__ == '__main__':
    # Чтение единого конфигурационного файла
    with open("config.yaml", "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    args = resolve_teacher_args(SimpleNamespace(**config_dict))

    # Загрузка оригинальной модели (Учителя) в половинной точности
    print(f"Загрузка модели из {args.teacher_model_path}...")
    model = load_teacher_model(args, torch_dtype=torch.float16, device="cuda")
    tokenizer = load_teacher_tokenizer(args, model_max_length=512)

    print("Загрузка данных для оценки...")
    dataset_cut = load_local_dataset(args.local_train_file, max_lines=100)

    print(f"Анализ важности слоев (метрика {args.loss_type})...")
    
    # Роутинг выбора метрики на основе config.yaml
    if args.loss_type == "BI":
        mean_scores = calculate_BI(model, tokenizer, dataset_cut)
    elif args.loss_type == "MSE":
        mean_scores = calculate_mse_importance(model, tokenizer, dataset_cut)
    elif args.loss_type == "LRP":
        mean_scores = calculate_LRP_importance(model, tokenizer, dataset_cut, epsilon=1e-6)
    else:
        raise ValueError("Неизвестный параметр loss_type в конфиге. Используйте 'BI', 'MSE' или 'LRP'.")

    # Сортируем результаты. Чем меньше балл, тем бесполезнее слой (безопаснее удалять)
    sorted_all_layers = sorted(mean_scores.items(), key=lambda x: x[1])

    # Фильтрация кандидатов: исключаем слои, которые не попали в 'allowed_prune_layers' (например, первый и последний)
    candidates_for_removal = []
    for layer_idx, score in sorted_all_layers:
        if layer_idx in args.allowed_prune_layers:
            candidates_for_removal.append(layer_idx)

    # Отрезаем N самых бесполезных слоев согласно лимиту из конфига
    layers_to_cut = candidates_for_removal[:args.max_layers_to_prune]
    print(f"Слои по возрастанию важности: {[l for l, s in sorted_all_layers]}")
    print(f"Автоматически выбраны слои для удаления: {layers_to_cut}")

    print("Удаление слоев из архитектуры...")
    layer_removal(model, layers_to_cut)

    # Исправление возможной проблемы с типом слоев в конфигурации Qwen
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = model.config.layer_types[:model.config.num_hidden_layers]

    # Сохранение обрезанной модели на диск
    print(f"Сохранение обрезанной модели в {args.student_model_path}...")
    if not os.path.exists(args.student_model_path):
        os.makedirs(args.student_model_path)      

    model.save_pretrained(args.student_model_path)
    tokenizer.save_pretrained(args.student_model_path)

    # Сохранение лога обрезанных слоев для автоматического маппинга в Qwen2Distillation.py
    with open(os.path.join(args.student_model_path, "pruned_info.json"), "w") as f:
        json.dump({"removed_layers": layers_to_cut}, f)

    print(f"Параметров осталось: {sum(p.numel() for p in model.parameters())}")
    print("Готово!")
