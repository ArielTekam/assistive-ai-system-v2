import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Support correct du russe
plt.rcParams['font.family'] = 'DejaVu Sans'

output_dir = Path("Experiments/final_analysis/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# DONNÉES EXPÉRIMENTALES
# ==============================

data = pd.DataFrame([
    {
        "version": "Старый трекер",
        "fps": 11.67,
        "latency": 85.73,
        "unique_ids": 198,
        "filtered_messages": 312,
        "decision_messages": 38
    },
    {
        "version": "ByteTrack",
        "fps": 11.74,
        "latency": 85.37,
        "unique_ids": 104,
        "filtered_messages": 162,
        "decision_messages": 35
    },
    {
        "version": "ByteTrack + SAFE",
        "fps": 11.76,
        "latency": 85.09,
        "unique_ids": 104,
        "filtered_messages": 162,
        "decision_messages": 10
    }
])

# ==============================
# FONCTION DE GÉNÉRATION
# ==============================

def save_bar_chart(metric, ylabel, title, filename):

    plt.figure(figsize=(8, 5))

    plt.bar(
        data["version"],
        data[metric]
    )

    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(rotation=10)

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300
    )

    plt.close()

# ==============================
# FPS
# ==============================

save_bar_chart(
    metric="fps",
    ylabel="Средний FPS",
    title="Сравнение среднего FPS",
    filename="fps_comparison_ru.png"
)

# ==============================
# LATENCY
# ==============================

save_bar_chart(
    metric="latency",
    ylabel="Средняя задержка (мс)",
    title="Сравнение средней задержки",
    filename="latency_comparison_ru.png"
)

# ==============================
# UNIQUE IDS
# ==============================

save_bar_chart(
    metric="unique_ids",
    ylabel="Количество уникальных ID",
    title="Фрагментация трекинга",
    filename="unique_ids_comparison_ru.png"
)

# ==============================
# FILTERED MESSAGES
# ==============================

save_bar_chart(
    metric="filtered_messages",
    ylabel="Сообщения после фильтрации C2",
    title="Снижение контекстного шума",
    filename="filtered_messages_comparison_ru.png"
)

# ==============================
# DECISION MESSAGES
# ==============================

save_bar_chart(
    metric="decision_messages",
    ylabel="Финальные сообщения после SAFE",
    title="Снижение итоговой когнитивной нагрузки",
    filename="decision_messages_comparison_ru.png"
)

# ==============================
# EXPORT CSV
# ==============================

data.to_csv(
    "Experiments/final_analysis/final_tracker_safe_comparison_ru.csv",
    index=False
)

print("====================================")
print("ГРАФИКИ УСПЕШНО СОЗДАНЫ")
print("====================================")

print("\nПапка с графиками:")
print(output_dir)

print("\nCSV файл:")
print("Experiments/final_analysis/final_tracker_safe_comparison_ru.csv")

print("====================================")