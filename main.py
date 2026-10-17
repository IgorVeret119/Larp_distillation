import subprocess
import sys

def run_step(command, step_name):
    print(f"Запуск этапа: {step_name}")
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\nОШИБКА: Этап '{step_name}' завершился с кодом {result.returncode}.")
        sys.exit(1)
    
    print(f"\nЭтап '{step_name}' успешно завершен.")

if __name__ == '__main__':
    run_step(
        command=["python", "auto_prune.py"], 
        step_name="Анализ и обрезка модели"
    )

    run_step(
        command=["python", "Qwen2Distillation.py"], 
        step_name="Дистилляция (дообучение)"
    )

    print("\nМодель готова.")