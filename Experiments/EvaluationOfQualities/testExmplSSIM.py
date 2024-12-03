import psutil
import cv2
import numpy as np
import os

def stitch_image_in_sub_directory(input_directory, output_directory):
    file_name = 'final_image_stitched.jpg'
    try:
        if os.path.isdir(input_directory):
            for dirpath, _, filenames in os.walk(input_directory):
                filenames = sorted(filenames)
                print('Stitching is starting...')
                print('---------------------------------')
                image_collage = cv2.imread(os.path.join(dirpath, filenames[0]))
                print("Loading image:", os.path.join(dirpath, filenames[0]))
                previous_image = filenames[0]
                temp_num = 1

                for index, filename in enumerate(filenames[1:], start=1):
                    # print_memory_info()
                    print("Loading image:", os.path.join(dirpath, filename))
                    main_image = cv2.imread(os.path.join(dirpath, filename))

                    if psutil.virtual_memory().available * 100 / psutil.virtual_memory().total < 40:
                        print('Reaching limit')
                        file_name = f'temp_image_stitched_{temp_num:04d}.jpg'
                        save_image(output_directory, file_name, image_collage)
                        image_collage = main_image
                        temp_num += 1
                        print('---------------------------------')
                        continue

                    print(f"{index}. Stitching {previous_image} AND {filename} in process")
                    print('---------------------------------')
                    image_collage = first_step(image_collage, main_image)
                    previous_image = filename



                save_image(output_directory, file_name, image_collage)
    except Exception as e:
        print(f"An error occurred: {e}")

def save_image(directory, file_name, image):
    #     check directory is exist and create if not exit
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory+'\\'+file_name, image)

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

# Масиви для збереження значень метрик
ssim_values = []
psnr_values = []
mean_color_diff_values = []

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    # Перетворення зображень в сірий формат
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_value, ssim_map = ssim(img1_gray, img2_gray, full=True)

    # Output SSIM value
    return ssim_value

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

    ssim_values.append(ssim_value)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def warpImages_calculate_psnr(img1, img2, H):
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

    # Область перекриття (можливо зберегти як зображення)
    overlap_x_start = max(0, -min_x)
    overlap_y_start = max(0, -min_y)
    overlap_x_end = overlap_x_start + min(cols1, max_x)
    overlap_y_end = overlap_y_start + min(rows1, max_y)

    overlap_area = output_img[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]
    cv2.imwrite('overlapping_area_psnr.jpg', overlap_area)  # Збереження області перекриття

    # Створення маски перекриття
    # overlap_mask = np.max(output_img, axis=2) > 0  # Переконуємося, що маска має бути бінарною
    # overlap_mask = overlap_mask.astype(np.uint8) * 255  # Конвертація в потрібний тип
    # overlap_area_img1 = cv2.bitwise_and(img1, img1, mask=overlap_mask[y_slice, x_slice])
    # cv2.imwrite('overlap_area_img1.jpg', overlap_area_img1)  # Збереження області перекриття
    # overlap_area_img2 = cv2.bitwise_and(output_img, output_img, mask=overlap_mask)
    # cv2.imwrite('overlap_area_img2.jpg', overlap_area_img2)  # Збереження області перекриття

    print("Calculate_psnr: " + str(calculate_psnr(overlap_area, img1)))

    psnr_values.append(calculate_psnr(overlap_area, img1))


def calculate_mean_color_difference(img1, img2):
    # Переконаємося, що зображення мають однакові розміри
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions to compute mean color difference.")

    # Обчислюємо абсолютну різницю між відповідними пікселями
    diff = np.abs(img1 - img2)

    # Обчислюємо середню різницю для кожного кольорового каналу
    mean_diff_per_channel = np.mean(diff, axis=(0, 1))

    # Середня різниця кольору
    mean_color_diff = np.mean(mean_diff_per_channel)

    return mean_color_diff

def warpImages_calculate_mean_color_difference(img1, img2, H):
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

    # Область перекриття (можливо зберегти як зображення)
    overlap_x_start = max(0, -min_x)
    overlap_y_start = max(0, -min_y)
    overlap_x_end = overlap_x_start + min(cols1, max_x)
    overlap_y_end = overlap_y_start + min(rows1, max_y)

    overlap_area = output_img[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]
    cv2.imwrite('overlapping_area color difference.jpg', overlap_area)  # Збереження області перекриття

    # Створення маски перекриття
    # overlap_mask = np.max(output_img, axis=2) > 0  # Переконуємося, що маска має бути бінарною
    # overlap_mask = overlap_mask.astype(np.uint8) * 255  # Конвертація в потрібний тип
    # overlap_area_img1 = cv2.bitwise_and(img1, img1, mask=overlap_mask[y_slice, x_slice])
    # cv2.imwrite('overlap_area_img1.jpg', overlap_area_img1)  # Збереження області перекриття
    # overlap_area_img2 = cv2.bitwise_and(output_img, output_img, mask=overlap_mask)
    # cv2.imwrite('overlap_area_img2.jpg', overlap_area_img2)  # Збереження області перекриття

    print("Calculate_mean color difference: " + str(calculate_mean_color_difference(overlap_area, img1)))
    mean_color_diff_values.append(calculate_mean_color_difference(overlap_area, img1))


def print_memory_info():
    memory = psutil.virtual_memory()
    print("Memory available is {:.2f}GB ({:.2f}%)".format(memory.available / (1024.0 ** 3), memory.available * 100 / memory.total))

def first_step(img1, img2):
    if img1 is None or img2 is None:
        print("One or both images are not loaded correctly.")
        return None
    # Create our ORB detector and detect keypoints and descriptors
    sift = cv2.SIFT_create(nfeatures=6000)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher()

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)

    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    # Set minimum match condition
    MIN_MATCH_COUNT = 100

    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = warpImages(img2, img1, M)
        warpImages_calculate_ssim(img2, img1, M)
        warpImages_calculate_psnr(img2, img1, M)
        warpImages_calculate_mean_color_difference(img2, img1, M)

        return result
    else: return None

# Виклик функції для об'єднання
input_directory = 'D:\\Programing\\Cursova\\Experiments\\EvaluationOfQualities\\frames_output'  # Шлях до директорії з фото

output_directory = 'output'
stitch_image_in_sub_directory(input_directory, output_directory)

# Збереження масивів значень метрик в блокнот
with open('metrics_output.txt', 'w') as f:
    f.write('SSIM Values:\n')
    f.write('\n'.join(map(str, ssim_values)) + '\n')
    f.write('\nPSNR Values:\n')
    f.write('\n'.join(map(str, psnr_values)) + '\n')
    f.write('\nMean Color Difference Values:\n')
    f.write('\n'.join(map(str, mean_color_diff_values)) + '\n')



