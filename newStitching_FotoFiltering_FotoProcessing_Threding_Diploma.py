import psutil
import cv2
import numpy as np
import os
import shutil
import exifread
import re

from datetime import datetime
from PIL import Image

import exifread
from fractions import Fraction

import logging
import threading

from concurrent.futures import ThreadPoolExecutor

# Перевіряємо GPS-висоту
def check_gps_and_copy(source_file, destination_dir):
    try:
        filename = os.path.basename(source_file)
        dest_file = os.path.join(destination_dir, filename)
        shutil.copy(source_file, dest_file)

        with open(dest_file, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='GPS GPSAltitude')
            if 'GPS GPSAltitude' in tags:
                altitude_fraction = tags['GPS GPSAltitude'].values[0]
                altitude = float(Fraction(altitude_fraction))
                if altitude < 150:  # Видаляємо файл, якщо висота < 150 м
                    os.remove(dest_file)
                    return None
        return dest_file  # Повертаємо шлях до файлу, якщо він валідний
    except Exception as e:
        print(f"Помилка з файлом {source_file}: {e}")
        return None

# Видалення метаданих
def remove_metadata(file_path):
    try:
        with Image.open(file_path) as img:
            data = list(img.getdata())
            new_img = Image.new(img.mode, img.size)
            new_img.putdata(data)
            new_img.save(file_path)
    except Exception as e:
        print(f"Помилка при видаленні метаданих {file_path}: {e}")

# Основна функція
def process_images(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Список файлів для обробки
    source_files = [
        os.path.join(source_dir, filename)
        for filename in os.listdir(source_dir)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Копіюємо файли і перевіряємо GPS у потоках
    valid_files = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda file: check_gps_and_copy(file, destination_dir), source_files)
        valid_files = [file for file in results if file]  # Файли, які пройшли перевірку GPS

    # Сортуємо файли за часом створення
    sorted_files = sorted(valid_files, key=lambda x: datetime.strptime(
        Image.open(x)._getexif()[306], '%Y:%m:%d %H:%M:%S').timestamp())

    # Перейменовуємо файли
    for i, file_path in enumerate(sorted_files):
        try:
            timestamp = Image.open(file_path)._getexif()[306]
            time_str = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S').strftime('%Y%m%d_%H%M%S')
            new_filename = f"{i}_{time_str}.png"
            new_path = os.path.join(destination_dir, new_filename)
            os.rename(file_path, new_path)
            sorted_files[i] = new_path  # Оновлюємо список з новим шляхом
        except Exception as e:
            print(f"Помилка при перейменуванні {file_path}: {e}")

    # Видаляємо метадані у потоках
    with ThreadPoolExecutor() as executor:
        executor.map(remove_metadata, sorted_files)

def stitch_image_in_sub_directory(input_directory, output_directory, progress_callback=None):

    # Add logging configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        if os.path.isdir(input_directory):
            # Виклик функції для обробки зображень
            process_images(input_directory, os.path.join(output_directory, 'temp'))

            # Отримуємо список усіх файлів
            for dirpath, _, filenames in os.walk(os.path.join(output_directory, 'temp')):
                filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
                print('Stitching is starting...')
                print('---------------------------------')

                # Розрахунок загальної кількості кроків
                num_parts = 8
                filenames_parts = np.array_split(filenames, num_parts)
                total_steps = len(filenames) + len(filenames_parts) + 1  # Кількість ітерацій + груп + фінальна склейка
                step_counter = 0  # Лічильник прогресу

                # Створення ThreadPool з 5 потоками
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(max_workers=4)

                def stitch_image_part(filenames, output_directory, group_number):
                    nonlocal step_counter
                    try:
                        logger.info(f'Поток {threading.get_ident()} - Група {group_number} - Початок склеювання: {filenames}')

                        # Створення директорії Analysis і групової директорії
                        analysis_dir = os.path.join(output_directory, 'Analysis')
                        group_dir = os.path.join(analysis_dir, f'Group_{group_number}')
                        if not os.path.exists(group_dir):
                            os.makedirs(group_dir)

                        # Створіть зображення-колаж з першого зображення
                        image_collage = cv2.imread(os.path.join(output_directory, 'temp', filenames[0]), cv2.IMREAD_UNCHANGED)

                        # Пройдіть по решті зображень у частині
                        for i, filename in enumerate(filenames[1:], start=1):
                            # Завантажте наступне зображення
                            main_image = cv2.imread(os.path.join(output_directory, 'temp', filename))

                            # Створення папки для ітерації
                            iteration_dir = os.path.join(group_dir, f'Iteration_{i}')
                            if not os.path.exists(iteration_dir):
                                os.makedirs(iteration_dir)

                            # Збереження зображень, які склеюються
                            cv2.imwrite(os.path.join(iteration_dir, 'image_1.png'), image_collage)
                            cv2.imwrite(os.path.join(iteration_dir, 'image_2.png'), main_image)

                            # Склейте два зображення
                            image_collage = first_step(image_collage, main_image)

                            # Оновлення прогресу
                            step_counter += 1
                            if progress_callback:
                                progress_callback(step_counter, total_steps)

                        # Збережіть зображення-колаж
                        save_image(output_directory, f'temp\\temp_image_stitched_{group_number:04d}.png', image_collage)
                        
                        logger.info(f'Поток {threading.get_ident()} - Група {group_number} - Завершено склеювання')

                        # Оновлення прогресу
                        step_counter += 1
                        if progress_callback:
                            progress_callback(step_counter, total_steps)

                    except Exception as e:
                        print(f"An error occurred: {e}")


                def stitch_image_pair(filenames, output_directory):
                    nonlocal step_counter
                    try:
                        logger.info('Поток %s: Початок склеювання пари', threading.get_ident())

                        filename1, filename2 = filenames[:2]

                        # Отримання чисел з імен файлів
                        id1 = re.findall(r'\d{4}', filename1)[0]
                        id2 = re.findall(r'\d{4}', filename2)[0]

                        # Склейте два зображення
                        image_collage = first_step(cv2.imread(os.path.join(output_directory, 'temp', filename1)),
                                                cv2.imread(os.path.join(output_directory, 'temp', filename2)))

                        # Збережіть зображення-колаж
                        save_image(output_directory, f'temp\\temp_image_stitched_{id1}_{id2}.png', image_collage)

                        new_filenames.append(f'temp_image_stitched_{id1}_{id2}.png')

                        logger.info('Поток %s: Завершено склеювання пари', threading.get_ident())

                        # Оновлення прогресу
                        step_counter += 1
                        if progress_callback:
                            progress_callback(step_counter, total_steps)

                    except Exception as e:
                        print(f"An error occurred: {e}")


                # Додавання завдань до пулу для кожної частини filenames_parts
                for i, filenames_part in enumerate(filenames_parts):
                    pool.submit(stitch_image_part, filenames_part, output_directory, i)

                # Зачекайте на завершення всіх завдань
                pool.shutdown(wait=True)

                # Відфільтруйте файли, які відповідають шаблону
                new_filenames = [filename for filename in os.listdir(os.path.join(output_directory, 'temp')) if filename.startswith('temp_image_stitched_')]

                # Сортування перед фінальним склеюванням
                new_filenames = sorted(new_filenames, key=lambda x: int(x.split('.')[0][-4:]))

                while len(new_filenames) > 1:
                    pool = ThreadPoolExecutor(max_workers=2)

                    # Сортування перед склеюванням наступного ряду
                    new_filenames = sorted(new_filenames, key=lambda x: int(x.split('.')[0][-4:]))
                    print('Sort happen')
                    print('Sorted array')
                    print(new_filenames)

                    while len(new_filenames) >= 2:
                        filename1, filename2 = new_filenames[:2]
                        print('---------------------------------')
                        print(f'Stitch thread create {filename1}, {filename2} go stitch')
                        pool.submit(stitch_image_pair, [filename1, filename2], output_directory)
                        print(f'Delete {filename1}, {filename2} from array')
                        new_filenames = new_filenames[2:]

                    pool.shutdown(wait=True)
                    print('---------------------------------')

                # Завершення фінальної склейки
                if len(new_filenames) == 1:
                    final_image = cv2.imread(os.path.join(output_directory, 'temp', new_filenames[0]))
                    save_image(output_directory, 'final_image_stitched.png', final_image)

                    step_counter += 1
                    if progress_callback:
                        progress_callback(step_counter, total_steps)
                
                print("---Final---")

    except Exception as e:
        logging.exception("An error occurred during image processing: %s", e)

def crop_image(img):
    # Обрізає ряди чорних пікселів зверху та знизу
    non_black_rows = np.where(np.any(np.any(img, axis=2), axis=1))[0]
    cropped_img = img[non_black_rows[0]:non_black_rows[-1] + 1, :]

    # Обрізає стовпці чорних пікселів зліва та справа
    non_black_cols = np.where(np.any(np.any(cropped_img, axis=2), axis=0))[0]
    cropped_img = cropped_img[:, non_black_cols[0]:non_black_cols[-1] + 1]

    return cropped_img

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

    from scipy.ndimage import convolve
    # Крок 8.1: Створюємо маску для чорних пікселів всередині зображення, ігноруючи чорні ділянки на краях
    def find_inner_black_pixels(image):
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

        # Розмірність зображення
        h, w = image.shape[:2]

        # Створюємо маску чорних пікселів
        black_mask_expanded = np.zeros((h, w), dtype=bool)
        black_mask_expanded[holes_coords[:, 0], holes_coords[:, 1]] = True

        # Створюємо вікно 3x3 для зсувів
        kernel = np.ones((3, 3), dtype=np.float32)

        # Ініціалізуємо масив для зберігання результатів
        mean_colors = np.zeros_like(image, dtype=np.float32)

        # Підраховуємо кількість валідних сусідів для кожного пікселя
        valid_mask = (image[:, :, 0] > 10) & (image[:, :, 1] > 10) & (image[:, :, 2] > 10)
        neighbor_count = convolve(valid_mask.astype(np.float32), kernel, mode='constant', cval=0)

        # Для кожного каналу BGR
        for channel in range(3):
            # Маска кольору з валідних сусідів
            channel_data = image[:, :, channel] * valid_mask

            # Зсув кольору сусідів за допомогою згортки
            neighbor_sum = convolve(channel_data.astype(np.float32), kernel, mode='constant', cval=0)

            # Рахуємо середнє значення кольорів сусідів
            mean_colors[:, :, channel] = np.divide(
                neighbor_sum,
                neighbor_count,
                out=np.zeros_like(neighbor_sum),  # Уникаємо ділення на нуль
                where=neighbor_count > 0
            )

        # Заповнюємо чорні пікселі середнім кольором
        image[black_mask_expanded] = mean_colors[black_mask_expanded].astype(np.uint8)
        # Зберігаємо оригінальне зображення з виправленими чорними пікселями
        cv2.imwrite("image_with_filled_black_pixels.jpg", image)
        
        return image

    output_img = find_inner_black_pixels(output_img)

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

    return crop_image(output_img)

def print_memory_info():
    memory = psutil.virtual_memory()
    print("Memory available is {:.2f}GB ({:.2f}%)".format(memory.available / (1024.0 ** 3), memory.available * 100 / memory.total))

def save_image(directory, file_name, image):
    #     check directory is exist and create if not exit
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory+'\\'+file_name, image)

def preprocess_image(image):
  # Перетворення на чорно-білий
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = gray_image.astype("uint8")

  # Фільтр Канні
  # edges_image = cv2.Canny(gray_image, 100, 200)

  # Зменшення розміру
  # resized_image = cv2.resize(edges_image, (0, 0), fx=1, fy=1)

  # Видалення чорного кольору
  thresh = 254
  black_mask = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY_INV)[1]
  gray_image = cv2.bitwise_and(gray_image, black_mask)

  # Гістограма вирівнювання: Цей метод покращує контрастність зображення, що може допомогти знайти більше деталей.
  gray_image = cv2.equalizeHist(gray_image)
 
  kernel_size = (11, 11)
  sigma = 2.0
  gray_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

  # cv2.imshow('оброблене фото', gray_image)
  # cv2.waitKey(0)

  return gray_image

def first_step(img1, img2):
    if img1 is None or img2 is None:
        print("One or both images are not loaded correctly.")
        return None
    # Create our ORB detector and detect keypoints and descriptors
    sift = cv2.SIFT_create(nfeatures=4000)

    # Попередня обробка
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = sift.detectAndCompute(processed_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(processed_img2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher()

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)

    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.78 * n.distance:
            good.append(m)

    # Set minimum match condition
    MIN_MATCH_COUNT = 50

    if len(good) >= MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = warpImages(img2, img1, M)
        #cv2.imshow('result', result)
        #cv2.waitKey(0)
        return result

    else: return None


#input_directory = 'project\\fullSource'
#output_directory = 'output'
#stitch_image_in_sub_directory(input_directory, output_directory)
