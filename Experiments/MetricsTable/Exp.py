import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

# Крок 1 - Завантаження зображень
# Завантажуємо два зображення для подальшої обробки
img1 = cv2.imread("DJI_0726.JPG")
img2 = cv2.imread("DJI_0728.JPG")

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

def crop_image(img):
    # Обрізає ряди чорних пікселів зверху та знизу
    non_black_rows = np.where(np.any(np.any(img, axis=2), axis=1))[0]
    cropped_img = img[non_black_rows[0]:non_black_rows[-1] + 1, :]

    # Обрізає стовпці чорних пікселів зліва та справа
    non_black_cols = np.where(np.any(np.any(cropped_img, axis=2), axis=0))[0]
    cropped_img = cropped_img[:, non_black_cols[0]:non_black_cols[-1] + 1]

    return cropped_img

def warpImages(img1, img2, H):
    # Розмір зображень
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Створюємо координати для всіх пікселів у зображенні 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]

    # Округлюємо трансформовані координати до цілих значень
    transformed_coords = np.round(transformed_coords[:, :2]).astype(int)

    # Визначаємо межі області для об'єднання зображень
    min_x = min(0, np.min(transformed_coords[:, 0]), np.min([0, cols1]))
    max_x = max(cols1, np.max(transformed_coords[:, 0]), np.max([0, cols1]))
    min_y = min(0, np.min(transformed_coords[:, 1]), np.min([0, rows1]))
    max_y = max(rows1, np.max(transformed_coords[:, 1]), np.max([0, rows1]))

    # Створюємо порожнє зображення, яке буде вміщувати обидва зображення
    output_img = np.zeros((max_y - min_y, max_x - min_x, img1.shape[2]), dtype=np.uint8)

    # Визначення області, куди слід помістити зображення 1 на порожньому зображенні
    y_slice = slice(-min_y, -min_y + rows1)
    x_slice = slice(-min_x, -min_x + cols1)

    # Розміщення зображення 1 на відповідних позиціях порожнього зображення
    output_img[y_slice, x_slice] = img1

    # Фільтруємо пікселі, які виходять за межі зображення та є чорними
    valid_pixels_mask = np.logical_and.reduce([
        transformed_coords[:, 0] >= min_x,
        transformed_coords[:, 0] < max_x,
        transformed_coords[:, 1] >= min_y,
        transformed_coords[:, 1] < max_y,
        ~np.all(img2.reshape(-1, 3)[..., :3] == 0, axis=1)
    ])

    # Переносимо пікселі, що задовільняють умову, на фінальне зображення
    output_coords = transformed_coords[valid_pixels_mask] - [min_x, min_y]
    output_img[output_coords[:, 1], output_coords[:, 0]] = img2.reshape(-1, 3)[valid_pixels_mask, :3]

    return crop_image(output_img)

# Крок 9 - Визначення гомографії та об'єднання зображень
# Встановлюємо мінімальну кількість відповідностей для об'єднання
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Отримуємо координати ключових точок для обох зображень
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Обчислюємо матрицю гомографії
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Об'єднуємо зображення на основі гомографії
    result = warpImages(img2, img1, M)

    # Зберігаємо результат
    cv2.imwrite('panorama_result_ORB.jpg', result)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import math

# Крок 10 - Оцінка якості ключових точок

# 1. Оцінка правильності співставлення (R)
def calculate_R(good_matches, total_matches, N=2):
    R = (len(good_matches) / total_matches) * (N - 1)
    return R

# 3. Оцінка стабільності точок (S)
def calculate_S(avg_response, max_response):
    S = avg_response / max_response
    return S

# Загальна інтегральна оцінка КТ
def calculate_KT(R, D, S, N=2):
    KT = (0.4 * R + 0.3 * D + 0.3 * S) / math.log2(N)
    return KT

# Використання обчислень на основі знайдених точок та відповідностей
total_matches = len(matches)
good_matches = good
N = 2  # кількість фото (для цієї реалізації - два зображення)

# Обчислення R
R = calculate_R(good_matches, total_matches, N)

def get_solid_overlap_area(img1, img2, H):
    """ 
    Визначає та зберігає суцільну зону перекриття між двома зображеннями, 
    використовуючи морфологічну обробку для заповнення можливих дірок.
    """
    # Розмір зображень
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Створюємо координати для всіх пікселів у зображенні 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]

    # Округлюємо трансформовані координати до цілих значень
    transformed_coords = np.round(transformed_coords[:, :2]).astype(int)

    # Визначаємо межі накладеного зображення
    min_x = min(0, np.min(transformed_coords[:, 0]), np.min([0, cols1]))
    max_x = max(cols1, np.max(transformed_coords[:, 0]), np.max([0, cols1]))
    min_y = min(0, np.min(transformed_coords[:, 1]), np.min([0, rows1]))
    max_y = max(rows1, np.max(transformed_coords[:, 1]), np.max([0, rows1]))

    # Створюємо порожню маску для визначення накладення
    overlay_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)

    # Заповнюємо зображення 1 на масці як "зона 1"
    y_slice = slice(-min_y, -min_y + rows1)
    x_slice = slice(-min_x, -min_x + cols1)
    overlay_mask[y_slice, x_slice] = 1

    # Накладаємо трансформовані координати зображення 2 на маску
    valid_pixels_mask = np.logical_and.reduce([
        transformed_coords[:, 0] >= min_x,
        transformed_coords[:, 0] < max_x,
        transformed_coords[:, 1] >= min_y,
        transformed_coords[:, 1] < max_y
    ])
    output_coords = transformed_coords[valid_pixels_mask] - [min_x, min_y]
    overlay_mask[output_coords[:, 1], output_coords[:, 0]] += 1

    # Зона перекриття - це область, де значення маски дорівнює 2
    overlap_area = np.where(overlay_mask == 2, 255, 0).astype(np.uint8)

    # Застосування морфологічного закриття для забезпечення суцільності
    kernel = np.ones((5, 5), np.uint8)  # Налаштуйте розмір ядра за необхідності
    solid_overlap_area = cv2.morphologyEx(overlap_area, cv2.MORPH_CLOSE, kernel)

    # Збереження суцільної зони перекриття як окреме зображення
    cv2.imwrite("solid_overlap_area.jpg", solid_overlap_area)
    print("Суцільна зона перекриття збережена як 'solid_overlap_area.jpg'")

    return solid_overlap_area

# Виклик функції для збереження суцільної області перекриття
solid_overlap_area = get_solid_overlap_area(img1, img2, M)

def calculate_D_in_overlap(keypoints, overlap_mask):
    """
    Обчислює D для точок у зоні перекриття, використовуючи суцільну маску перекриття.
    """
    # Конвертуємо координати ключових точок до цілого типу для зручного індексування
    keypoints_in_overlap = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Перевіряємо, чи точка потрапляє в зону перекриття на основі маски
        if 0 <= x < overlap_mask.shape[1] and 0 <= y < overlap_mask.shape[0] and overlap_mask[y, x] == 255:
            keypoints_in_overlap.append(kp)

    # Перевіряємо, чи є достатньо точок у зоні перекриття
    if len(keypoints_in_overlap) == 0:
        print("Немає точок у зоні перекриття.")
        return 0

    # Обчислюємо відстані для точок у зоні перекриття
    distances = [np.linalg.norm(kp.pt) for kp in keypoints_in_overlap]
    sigma_distances = np.std(distances)
    max_distance = max(distances)
    D = 1 - (sigma_distances / max_distance) if max_distance > 0 else 0
    return D

# Обчислення D для точок у зоні перекриття
D = calculate_D_in_overlap(keypoints1 + keypoints2, solid_overlap_area)

print(f"D (рівномірність розподілу в зоні перекриття): {D}")


# Обчислення S (з урахуванням стабільності відповіді детектора для кожного зображення)
responses = [kp.response for kp in keypoints1 + keypoints2]
avg_response = np.mean(responses)
max_response = np.max(responses)
S = calculate_S(avg_response, max_response)

# Обчислення інтегрального показника КТ
KT = calculate_KT(R, D, S, N)

# Вивід результатів
print(f"R: {R}")
print(f"D: {D}")
print(f"S: {S}")
print(f"KT: {KT}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_rotation_angle_from_homography(H):
    """
    Обчислює кут повороту в градусах з гомографічної матриці H.
    """
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    return angle

angle = 0

def calculate_artifacts_in_transformed_area(img2, H):
    """
    Деформує зображення 2 за допомогою гомографії та підраховує кількість артефактів 
    (чорних пікселів) у межах деформованого чотирикутника.
    """
    # Розмір зображення 2
    rows2, cols2 = img2.shape[:2]

    # Створюємо координати для всіх пікселів у зображенні 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]
    transformed_coords = np.round(transformed_coords[:, :2]).astype(int)

    # Отримуємо кут повороту з гомографії
    angle = calculate_rotation_angle_from_homography(H)
    print(f"Кут повороту зображення (в градусах): {angle:.2f}")

    # Визначаємо межі області для розміщення зображення 2
    min_x = min(0, np.min(transformed_coords[:, 0]))
    max_x = max(cols2, np.max(transformed_coords[:, 0]))
    min_y = min(0, np.min(transformed_coords[:, 1]))
    max_y = max(rows2, np.max(transformed_coords[:, 1]))

    # Створюємо порожнє зображення для розміщення деформованого зображення 2
    output_img = np.zeros((max_y - min_y, max_x - min_x, img2.shape[2]), dtype=np.uint8)

    # Фільтруємо пікселі, які входять у межі зображення і не є чорними
    valid_pixels_mask = np.logical_and.reduce([
        transformed_coords[:, 0] >= min_x,
        transformed_coords[:, 0] < max_x,
        transformed_coords[:, 1] >= min_y,
        transformed_coords[:, 1] < max_y,
        ~np.all(img2.reshape(-1, 3)[..., :3] == 0, axis=1)
    ])

    # Переносимо пікселі, що задовольняють умову, на фінальне зображення
    output_coords = transformed_coords[valid_pixels_mask] - [min_x, min_y]
    output_img[output_coords[:, 1], output_coords[:, 0]] = img2.reshape(-1, 3)[valid_pixels_mask, :3]

    # Визначаємо координати чотирьох кутів зображення 2
    corners = np.float32([[0, 0], [cols2, 0], [cols2, rows2], [0, rows2]]).reshape(-1, 1, 2)
    # Трансформуємо кути за допомогою гомографії
    transformed_corners = cv2.perspectiveTransform(corners, H)
    transformed_corners = transformed_corners.reshape(-1, 2).astype(int)

    # Створюємо маску для чотирикутника, всередині якого будемо рахувати чорні пікселі
    mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    cv2.fillConvexPoly(mask, transformed_corners - [min_x, min_y], 255)

    # Площа зони пошуку артефактів (кількість пікселів у чотирикутнику)
    total_pixels_in_area = np.sum(mask == 255)
    print(f"Площа зони для підрахунку артефактів (в пікселях): {total_pixels_in_area}")

    # Підраховуємо чорні пікселі (артефакти) всередині чотирикутника
    black_pixels_mask = np.all(output_img == [0, 0, 0], axis=2)
    artifacts_count = np.sum(black_pixels_mask & (mask == 255))

    # Підсвічуємо чорні пікселі червоним кольором для візуалізації
    highlighted_img = output_img.copy()
    highlighted_img[(black_pixels_mask & (mask == 255))] = [0, 0, 255]  # Червоний колір

    # Зберігаємо результат
    cv2.imwrite("highlighted_artifacts_transformed_area.jpg", highlighted_img)
    # print("Зображення з підсвіченими артефактами збережено як 'highlighted_artifacts_transformed_area.jpg'")

    return artifacts_count, highlighted_img, total_pixels_in_area, angle

# Виклик функції для підрахунку артефактів та створення зображення з підсвіченням
artifacts_count, highlighted_img, total_pixels_in_area, angle = calculate_artifacts_in_transformed_area(img2, M)
print(f"Кількість артефактів: {artifacts_count}")

# Обчислення Art_score на основі кількості артефактів
N = 2  # кількість фото (для цієї реалізації - два зображення)
art_score = np.exp(-9 * (artifacts_count / total_pixels_in_area))
print(f"Art_score: {art_score}")

def calculate_reprojection_error_using_warp(src_pts, dst_pts, H):
    """
    Обчислює середню репроєкційну похибку між відповідними точками, використовуючи підхід з обчисленням координат вручну.
    """
    # Перетворення координат src_pts у гомогенні, додаємо третій стовпець з 1
    coords = np.hstack([src_pts.reshape(-1, 2), np.ones((src_pts.shape[0], 1))])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]

    # Відбираємо тільки x і y координати
    transformed_coords = np.round(transformed_coords[:, :2])

    # Обчислюємо евклідову відстань між трансформованими координатами і реальними точками призначення
    errors = np.linalg.norm(dst_pts.reshape(-1, 2) - transformed_coords, axis=1)
    
    # Середнє значення похибки
    mean_error = np.mean(errors)
    return mean_error

# Викликаємо функцію для обчислення похибки
alignment_error = calculate_reprojection_error_using_warp(src_pts, dst_pts, M)
print(f"Alignment Error (Похибка вирівнювання): {alignment_error}")

# Обчислення Err_score
N = 2  # кількість фото (для цієї реалізації - два зображення)
Err_score = 1 - min(alignment_error / (50 * np.sqrt(N)), 1)
print(f"Err_score: {Err_score}")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
 

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    # Перетворення зображень в сірий формат
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_value, ssim_map = ssim(img1_gray, img2_gray, full=True)

    # Output SSIM value
    return ssim_value

ssim_value = 0

def warpImages_calculate_ssim(img1, img2, H):
    # Розмір зображень
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Створюємо координати для всіх пікселів у зображенні 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])

    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]

    # Округлюємо трансформовані координати до цілих значень
    transformed_coords = np.round(transformed_coords[:, :2]).astype(int)

    # Визначаємо межі області для об'єднання зображень
    min_x = min(0, np.min(transformed_coords[:, 0]), np.min([0, cols1]))
    max_x = max(cols1, np.max(transformed_coords[:, 0]), np.max([0, cols1]))
    min_y = min(0, np.min(transformed_coords[:, 1]), np.min([0, rows1]))
    max_y = max(rows1, np.max(transformed_coords[:, 1]), np.max([0, rows1]))

    # Створюємо порожнє зображення, яке буде вміщувати обидва зображення
    output_img = np.zeros((max_y - min_y, max_x - min_x, img1.shape[2]), dtype=np.uint8)

    # Визначення області, куди слід помістити зображення 1 на порожньому зображенні
    y_slice = slice(-min_y, -min_y + rows1)
    x_slice = slice(-min_x, -min_x + cols1)

    # Розміщення зображення 1 на відповідних позиціях порожнього зображення
    output_img[y_slice, x_slice] = img1

    # Фільтруємо пікселі, які виходять за межі зображення та є чорними
    valid_pixels_mask = np.logical_and.reduce([
        transformed_coords[:, 0] >= min_x,
        transformed_coords[:, 0] < max_x,
        transformed_coords[:, 1] >= min_y,
        transformed_coords[:, 1] < max_y,
        ~np.all(img2.reshape(-1, 3)[..., :3] == 0, axis=1)
    ])

    # Переносимо пікселі, що задовільняють умову, на фінальне зображення
    output_coords = transformed_coords[valid_pixels_mask] - [min_x, min_y]
    output_img[output_coords[:, 1], output_coords[:, 0]] = img2.reshape(-1, 3)[valid_pixels_mask, :3]

    # Вирізання області з output_img, куди було розміщено img1
    placed_area = output_img[y_slice, x_slice]

    # Виведення розмірів зображень, які будемо порівнювати
    print("Shape of placed_area: ", placed_area.shape)
    print("Shape of original_img1: ", img1.shape)
    cv2.imwrite('original_img1.jpg', img1)
    cv2.imwrite('placed_area.jpg', placed_area)

    # Розрахунок SSIM для області перекриття
    if placed_area.shape == img1.shape:
        ssim_value = calculate_ssim(img1, placed_area)
        print("Calculate_ssim: " + str(ssim_value))
    else:
        print("Error: Placed area and original image have different shapes, cannot calculate SSIM.")

    return ssim_value


ssim_value = warpImages_calculate_ssim(img2, img1, M)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import json
import os

def save_metrics_to_json(angle, SSIM, Err_score, Art_score, KT):
    filename = 'testes.json'
    
    # Перевіряємо наявність файла і завантажуємо існуючі дані, якщо файл існує
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        # Ініціалізуємо новий список для збереження даних, якщо файл не існує
        data = {"Tests": []}

    # Створюємо запис з новими даними
    new_entry = {
        "Angle": angle,
        "Metrics": {
            "SSIM": SSIM,
            "Err_score": Err_score,
            "Art_score": Art_score,
            "KT": KT
        }
    }

    # Додаємо новий запис до списку
    data["Tests"].append(new_entry)

    # Записуємо оновлені дані назад в файл
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Викликаємо функцію для збереження метрик, передаючи реальні значення
save_metrics_to_json(angle, ssim_value, Err_score, art_score, KT)