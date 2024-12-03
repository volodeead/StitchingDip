import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

# Крок 1 - Завантаження зображень
# Завантажуємо два зображення для подальшої обробки
img1 = cv2.imread("DJI_0736.JPG")
img2 = cv2.imread("DJI_0733.JPG")

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

    # Крок 8.1: Створюємо маску для чорних пікселів всередині зображення, ігноруючи чорні ділянки на краях
    def find_inner_black_pixels(image):
        # Перетворюємо зображення в відтінки сірого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Створюємо маску для не чорних пікселів (де піксель не є чорним)
        non_black_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)[1]

        # Створюємо маску для чорних пікселів
        black_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY_INV)[1]

        # Визначаємо контури області, яка не є чорною
        contours, _ = cv2.findContours(non_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Створюємо нову маску для обмеження області, всередині якої знаходяться чорні пікселі
        mask_without_edges = np.zeros_like(black_mask)

        # Малюємо внутрішні області (що не є на краях)
        cv2.drawContours(mask_without_edges, contours, -1, 255, thickness=cv2.FILLED)

        # Тепер виявляємо чорні пікселі всередині внутрішньої області
        inner_black_pixels_mask = cv2.bitwise_and(mask_without_edges, black_mask)

        # Виводимо та зберігаємо зображення для перевірки
        cv2.imwrite("inner_black_pixels_mask.jpg", inner_black_pixels_mask)
        print(f"Кількість дірок у масці: {np.sum(inner_black_pixels_mask == 255)}")
        holes_coords = np.argwhere(inner_black_pixels_mask == 255)
        print(f"Координати дірок: {holes_coords}")


        return inner_black_pixels_mask

    # Крок 8.2: Заповнення дірочок середнім значенням сусідів для кожного каналу окремо
    def fill_holes_average_of_neighbors(image, holes_mask):
        # Якщо маска має кілька каналів, беремо тільки один (наприклад, перший канал)
        if len(holes_mask.shape) > 2:
            holes_mask = holes_mask[:, :, 0]

        # Створюємо копію зображення для заповнення дірок
        filled_image = image.copy()

        # Отримуємо координати пікселів, де є дірочки
        black_pixels = np.argwhere(holes_mask == 255)

        # Проходимо по всіх пікселях з дірочками
        for (y, x) in black_pixels:
            # Список для зберігання значень сусідів
            neighbors = []

            # Перевіряємо кожного сусіда навколо (по вертикалі та горизонталі)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                # Перевіряємо, чи не виходять координати за межі зображення
                if 0 <= ny < filled_image.shape[0] and 0 <= nx < filled_image.shape[1]:
                    # Якщо сусідній піксель не є в масці дірки і не чорний, додаємо його значення до списку neighbors
                    if holes_mask[ny, nx] != 255 and np.any(filled_image[ny, nx] != 0):
                        neighbors.append(filled_image[ny, nx])

            # Якщо є сусіди, обчислюємо середнє значення і заповнюємо дірочку
            if neighbors:
                neighbors_avg = np.mean(neighbors, axis=0)
                filled_image[y, x] = neighbors_avg
            else:
                # Якщо немає дійсних сусідів, пропускаємо цей піксель
                pass

        return filled_image

    holes = find_inner_black_pixels(output_img)
    # Заповнюємо дірочки
    output_img = fill_holes_average_of_neighbors(output_img, holes)

    # Після виявлення дірочок накладаємо зображення 1 на фінальне зображення
    y_slice = slice(-min_y, -min_y + rows1)
    x_slice = slice(-min_x, -min_x + cols1)
    output_img[y_slice, x_slice] = img1

    return crop_image(output_img)

import time
def measure_execution_time(func, *args, repetitions=3):
    times = []
    for _ in range(repetitions):
        start_time = time.time()  # Початок вимірювання
        func(*args)  # Виклик функції warpImages з аргументами
        end_time = time.time()  # Кінець вимірювання
        
        elapsed_time = (end_time - start_time) * 1000  # Перетворюємо секунди в мілісекунди
        times.append(elapsed_time)
        print(f"Час виконання: {elapsed_time:.2f} мс")
    
    average_time = sum(times) / repetitions
    print(f"Середній час виконання після {repetitions} запусків: {average_time:.2f} мс")
    return average_time



# Крок 9 - Визначення гомографії та об'єднання зображень
# Встановлюємо мінімальну кількість відповідностей для об'єднання
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Отримуємо координати ключових точок для обох зображень
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Обчислюємо матрицю гомографії
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    measure_execution_time(warpImages, img2, img1, M )

    # Об'єднуємо зображення на основі гомографії
    result = warpImages(img2, img1, M)

    # Зберігаємо результат
    cv2.imwrite('panorama_result_ORB.jpg', result)

# Крок 10 - Виділення найбільшого контуру
# Перетворюємо результат у відтінки сірого
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Створюємо маску через порогове значення
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Знаходимо контури на зображенні
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Вибираємо найбільший контур за площею
largest_contour = max(contours, key=cv2.contourArea)

# Крок 11 - Створення маски на основі контуру
# Знаходимо опуклу оболонку найбільшого контуру (hull)
hull = cv2.convexHull(largest_contour)
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)

# Крок 12 - Бітові операції для прозорого фону
# Створюємо біле тло
bg = np.zeros(result.shape, dtype=np.uint8)
bg[:, :, :3] = (255, 255, 255)

# Використовуємо маску для об'єднання фону і переднього плану
img_bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
img_fg = cv2.bitwise_and(result, result, mask=mask)

# Кінцеве об'єднання переднього плану та фону
final = cv2.add(img_bg, img_fg)

# Зберігаємо фінальне зображення з прозорим фоном
cv2.imwrite('transparent1.jpg', final)