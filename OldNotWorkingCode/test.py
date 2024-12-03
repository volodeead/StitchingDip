import cv2
import numpy as np

# Завантаження зображення
image = cv2.imread('final_image_stitched.png')

# Перетворення в формат BGRA
bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

# Пошук абсолютно чорних пікселів
black_pixels = np.all(bgra_image == [0, 0, 0, 255], axis=-1)

# Заміна чорних пікселів на прозорі
bgra_image[black_pixels] = [0, 0, 0, 0]

# Збереження результату
cv2.imwrite('output.png', bgra_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])