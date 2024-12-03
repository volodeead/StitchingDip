import numpy as np
import matplotlib.pyplot as plt

# Константи
N = 2

# Артефакти від 1 до 5 000 000 з кроком 5000
artifacts_count = np.arange(1, 10000000, 10000)


total_pixels = 5280 * 2970  # Загальна кількість пікселів
alpha = 9  # Коефіцієнт згасання

# Новий розрахунок Art_score за новою формулою
art_score = np.exp(-alpha * (artifacts_count / total_pixels))

# Формула для Art_score
# art_score = 1 - np.minimum(artifacts_count / (1000 * N), 1)

# Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(artifacts_count, art_score, label='Art Score', color='b')
plt.xlabel('Artifacts Count')
plt.ylabel('Art Score')
plt.title('Art Score vs Artifacts Count')
plt.grid(True)
plt.legend()
plt.show()


import numpy as np
import cv2

# Вхідні координати 4-х точок
points = np.array([[1, 3054], [1477, 525], [6000, 3135], [4561, 5659]], dtype=np.int32)

# Створюємо порожню маску для обчислення площі (запас розмірів можна змінити залежно від максимальних координат)
mask = np.zeros((6000, 6000), dtype=np.uint8)

# Малюємо чотирикутник на масці
cv2.fillConvexPoly(mask, points, 255)

# Обчислюємо площу чотирикутника в пікселях
area_in_pixels = np.sum(mask == 255)
print(f"Площа чотирикутника в пікселях: {area_in_pixels}")
