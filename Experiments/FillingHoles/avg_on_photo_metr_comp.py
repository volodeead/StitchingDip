import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

# Крок 1 - Завантаження зображень
# Завантажуємо два зображення для подальшої обробки
img1 = cv2.imread("DJI_0709.JPG")
img2 = cv2.imread("DJI_0714.JPG")

# Перетворюємо зображення в відтінки сірого для обробки
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Крок 2 - Ініціалізація детектора ORB
# Створюємо ORB детектор для пошуку ключових точок та дескрипторів
orb = cv2.ORB_create(nfeatures=20000)

# Крок 3 - Виявлення ключових точок та дескрипторів
# Знаходимо ключові точки та дескриптори для обох зображень
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Крок 4 - Ініціалізація BFMatcher
# Створюємо BFMatcher для знаходження відповідностей між дескрипторами
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

# Крок 5 - Знаходження відповідних точок
# Знаходимо відповідності між дескрипторами двох зображень, використовуючи knnMatch
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Крок 6 - Фільтрація відповідностей
# Вибираємо найкращі відповідності, де відстань m менша за 0.8 від відстані n
good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

# Крок 7 - Функція для обрізки зображення
# Обрізає чорні пікселі зверху, знизу, зліва і справа
def crop_image(img):
    non_black_rows = np.where(np.any(np.any(img, axis=2), axis=1))[0]
    cropped_img = img[non_black_rows[0]:non_black_rows[-1] + 1, :]

    non_black_cols = np.where(np.any(np.any(cropped_img, axis=2), axis=0))[0]
    cropped_img = cropped_img[:, non_black_cols[0]:non_black_cols[-1] + 1]

    return cropped_img

# Крок 8 - Функція для перетворення та об'єднання зображень
# Трансформуємо зображення на основі гомографії та об'єднуємо їх, виділяючи дірочки
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Створюємо координати для кожного пікселя зображення 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]

    # Округлюємо до цілих значень
    transformed_coords = np.round(transformed_coords[:, :2]).astype(int)

    # Визначаємо розміри для фінального зображення
    min_x = min(0, np.min(transformed_coords[:, 0]), np.min([0, cols1]))
    max_x = max(cols1, np.max(transformed_coords[:, 0]), np.max([0, cols1]))
    min_y = min(0, np.min(transformed_coords[:, 1]), np.min([0, rows1]))
    max_y = max(rows1, np.max(transformed_coords[:, 1]), np.max([0, rows1]))

    # Створюємо порожнє зображення для об'єднання
    output_img = np.zeros((max_y - min_y, max_x - min_x, img1.shape[2]), dtype=np.uint8)

    # Накладаємо трансформоване зображення 2 на порожнє зображення
    valid_pixels_mask = np.logical_and.reduce([
        transformed_coords[:, 0] >= min_x,
        transformed_coords[:, 0] < max_x,
        transformed_coords[:, 1] >= min_y,
        transformed_coords[:, 1] < max_y
    ])

    # Переносимо пікселі зображення 2 на фінальне зображення
    output_coords = transformed_coords[valid_pixels_mask] - [min_x, min_y]
    output_img[output_coords[:, 1], output_coords[:, 0]] = img2.reshape(-1, 3)[valid_pixels_mask, :3]


    from skimage.metrics import structural_similarity as ssim

    def calculate_ssim(img1, img2):
        # Перетворення зображень в сірий формат
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_value, ssim_map = ssim(img1_gray, img2_gray, full=True)

        # Output SSIM value
        return ssim_value

    # Крок 8.1: Створюємо маску для чорних пікселів всередині зображення, ігноруючи чорні ділянки на краях
    def find_inner_black_pixels(image, base_image):
        # Перетворюємо зображення в відтінки сірого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Створюємо маску для не чорних пікселів
        non_black_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)[1]

        # Створюємо маску для чорних пікселів
        black_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)[1]

        # Визначаємо контури області, яка не є чорною
        contours, _ = cv2.findContours(non_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Створюємо нову маску для обмеження області, всередині якої знаходяться чорні пікселі
        mask_without_edges = np.zeros_like(black_mask)

        # Малюємо внутрішні області (що не є на краях)
        cv2.drawContours(mask_without_edges, contours, -1, 255, thickness=cv2.FILLED)

        # Виявляємо чорні пікселі всередині внутрішньої області
        inner_black_pixels_mask = cv2.bitwise_and(mask_without_edges, black_mask)

        # Знаходимо координати чорних пікселів
        holes_coords = np.argwhere(inner_black_pixels_mask == 255)

        mean_color = np.round(base_image.mean(axis=(0, 1))).astype(np.uint8)

        # Обчислюємо середні кольори для кожного чорного пікселя
        # Масивна заміна чорних пікселів
        image[inner_black_pixels_mask == 255] = mean_color

        # Зберігаємо оригінальне зображення з виправленими чорними пікселями
        cv2.imwrite("image_with_filled_black_pixels.jpg", image)

        return image
    
    output_img = find_inner_black_pixels(output_img, img2)

    # Зберігаємо копію результату з заповненими дірками
    filled_img2 = output_img.copy()

    # Тепер накладаємо зображення 1 поверх output_img
    y_slice = slice(-min_y, -min_y + rows1)
    x_slice = slice(-min_x, -min_x + cols1)
    output_img[y_slice, x_slice] = img1

    # Створюємо бітову маску для не чорних пікселів
    filled_img2_gray = cv2.cvtColor(filled_img2, cv2.COLOR_BGR2GRAY)
    non_black_mask = cv2.threshold(filled_img2_gray, 1, 255, cv2.THRESH_BINARY)[1]

    # Накладаємо тільки не чорні пікселі filled_img2 поверх output_img
    output_img = cv2.bitwise_and(filled_img2, filled_img2, mask=non_black_mask) + cv2.bitwise_and(output_img, output_img, mask=cv2.bitwise_not(non_black_mask))


    placed_area = output_img[y_slice, x_slice]

    cv2.imwrite('ssim_arg1_img1.jpg', img1)
    cv2.imwrite('ssim_arg2_placed_area.jpg', placed_area)

    ssim_value = None
    if placed_area.shape == img1.shape:
        ssim_value = calculate_ssim(img1, placed_area)
        print(f"SSIM після заповнення дірочок: {ssim_value}")
    else:
        print("SSIM не може бути розрахованим через різні розміри зображень.")

    return crop_image(output_img), ssim_value

import os
import psutil
import time
import threading

# Моніторинг процесу
def monitor_process_usage(pid, interval=0.01):
    process = psutil.Process(pid)
    cpu_readings = []
    memory_readings = []
    stop_event = threading.Event()  # Використовуємо threading.Event для керування моніторингом

    def monitor():
        while not stop_event.is_set():
            try:
                # Зчитуємо CPU та RAM
                cpu = process.cpu_percent(interval=None)  # CPU у %
                memory = process.memory_info().rss / (1024 ** 2)  # RAM у MB
                cpu_readings.append(cpu)
                memory_readings.append(memory)
                time.sleep(interval)  # Інтервал між записами
            except psutil.NoSuchProcess:
                break  # Процес завершився

    # Запускаємо моніторинговий потік
    thread = threading.Thread(target=monitor)
    thread.start()
    return thread, cpu_readings, memory_readings, lambda: stop_event.set()  # Використовуємо stop_event.set для зупинки

# Виконання з моніторингом
def execute_with_monitoring(func, *args, **kwargs):
    pid = os.getpid()  # UID поточного процесу
    monitor_thread, cpu_readings, memory_readings, stop_monitoring = monitor_process_usage(pid)

    # Виконуємо функцію
    start_time = time.time()
    result = func(*args, **kwargs)  # Функція виконується
    elapsed_time = (time.time() - start_time) * 1000  # Час у мс

    # Зупиняємо моніторинг
    stop_monitoring()
    monitor_thread.join()

    # Обчислюємо середні значення
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0

    # Нормалізація CPU відносно загальної кількості ядер
    total_cores = psutil.cpu_count(logical=True)
    normalized_cpu = (avg_cpu / (total_cores * 100)) * 100  # Нормалізоване використання CPU
    used_cores = avg_cpu / 100  # Кількість задіяних ядер

    return result, elapsed_time, normalized_cpu, avg_memory, used_cores

# Запуск функції 5 разів
def run_multiple_times(func, *args, repetitions=5, **kwargs):
    total_time = 0
    total_cpu = 0
    total_memory = 0
    total_used_cores = 0

    for i in range(repetitions):
        print(f"Запуск {i+1} з {repetitions}")
        _, elapsed_time, normalized_cpu, avg_memory, used_cores = execute_with_monitoring(func, *args, **kwargs)
        total_time += elapsed_time
        total_cpu += normalized_cpu
        total_memory += avg_memory
        total_used_cores += used_cores

        # Виводимо середні показники для поточного запуску
        print(f"Час виконання: {elapsed_time:.2f} мс")
        print(f"Використання CPU: {normalized_cpu:.2f}%")
        print(f"Використання RAM: {avg_memory:.2f} MB")
        print(f"Кількість задіяних ядер: {used_cores:.2f}")

    # Обчислюємо середні значення
    avg_time = total_time / repetitions
    avg_cpu = total_cpu / repetitions
    avg_memory = total_memory / repetitions
    avg_used_cores = total_used_cores / repetitions

    print(f"\nСередній час виконання: {avg_time:.2f} мс")
    print(f"Середнє використання CPU: {avg_cpu:.2f}%")
    print(f"Середнє використання RAM: {avg_memory:.2f} MB")
    print(f"Середня кількість задіяних ядер: {avg_used_cores:.2f}")
    
    return avg_time, avg_cpu, avg_memory, avg_used_cores

MIN_MATCH_COUNT = 100

if len(good) > MIN_MATCH_COUNT:
    # Отримуємо координати ключових точок для обох зображень
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Обчислюємо матрицю гомографії
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Запускаємо функцію warpImages 5 разів із моніторингом
    avg_time, avg_cpu, avg_memory, avg_used_cores = run_multiple_times(warpImages, img2, img1, M)

    # Зберігаємо результат останнього виконання
    final_result, _, _, _, _ = execute_with_monitoring(warpImages, img2, img1, M)
