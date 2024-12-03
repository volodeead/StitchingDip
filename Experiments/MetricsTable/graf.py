import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Заміни "testes.json" на шлях до свого JSON-файлу
json_file = "testes.json"

# Зчитуємо дані з JSON-файлу
with open(json_file, "r") as file:
    data = json.load(file)

# Перетворюємо дані у DataFrame
rows = []
for test in data["Tests"]:
    row = {"Angle": abs(test["Angle"]), **test["Metrics"]}
    rows.append(row)

df = pd.DataFrame(rows)

# Обчислюємо кореляційну матрицю
correlation_matrix = df.corr()

# Відображення теплової карти кореляційної матриці
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix of Metrics")
plt.show()

import pandas as pd
import json
import matplotlib.pyplot as plt

# Завантаження даних з файлу
with open("testes.json", "r") as file:
    data = json.load(file)["Tests"]

# Створюємо DataFrame
df = pd.DataFrame(data)
df["Angle"] = df["Angle"].abs()  # беремо модуль кута
df["Art_score"] = df["Metrics"].apply(lambda x: x["Art_score"])

# Групуємо дані по інтервалах у 1 градус і рахуємо середнє значення Art_score для кожного відрізка
df["Angle_range"] = df["Angle"].round()  # Округлюємо кут до найближчого градуса для групування
grouped_df = df.groupby("Angle_range")["Art_score"].mean().reset_index()

# Відображаємо на графіку тільки ті відрізки, які мають значення Art_score
plt.figure(figsize=(10, 6))
plt.plot(grouped_df["Angle_range"], grouped_df["Art_score"], marker="o", linestyle="-")
plt.xlabel("Angle (degrees)")
plt.ylabel("Average Art Score")
plt.title("Average Art Score vs. Angle (Grouped by 1-degree Intervals)")
plt.grid(True)
plt.show()


# Розпаковуємо метрики в окремі колонки
metrics_df = pd.json_normalize(df["Metrics"])
df = df.join(metrics_df, lsuffix='_original', rsuffix='_new').drop(columns=["Metrics"])

# Обчислюємо середні значення для всіх метрик
average_metrics = metrics_df.mean()

# Обчислення середніх значень метрик для кожної групи
grouped_metrics = df.groupby("Group").mean()

# Виводимо результати
print("Середні значення метрик для кожної групи:\n")
for group, metrics in grouped_metrics.iterrows():
    print(f"Група {group}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\n")