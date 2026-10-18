import subprocess
import sys
import yaml

def run_step(command, step_name):
    print(f"Запуск этапа: {step_name}")
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\nОШИБКА: Этап '{step_name}' завершился с кодом {result.returncode}.")
        sys.exit(1)
    
    print(f"\nЭтап '{step_name}' успешно завершен.")

if __name__ == '__main__':
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("merge_adapter", False):
        run_step(
            command=[sys.executable, "merge_adapter.py"],
            step_name="Объединение адаптера с базовой моделью"
        )

    run_step(
        command=[sys.executable, "auto_prune.py"], 
        step_name="Анализ и обрезка модели"
    )

    run_step(
        command=[sys.executable, "Qwen2Distillation.py"], 
        step_name="Дистилляция (дообучение)"
    )

    print("\nМодель готова.")
