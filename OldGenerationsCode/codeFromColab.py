import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

# Load our images
img1 = cv2.imread("output\\temp\\0_20230515_114629.png")
img2 = cv2.imread("output\\temp\\1_20230515_114639.png")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create our ORB detector and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=8000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Create a BFMatcher object.
# It will find all of the matching keypoints on two images
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

# Find matching points
matches = bf.knnMatch(descriptors1, descriptors2,k=2)

# Finding the best matches
good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

from scipy.interpolate import griddata

def warpImages(img1, img2, H):
    # Розміри зображень
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    # Створюємо координати для всіх пікселів у зображенні 2
    y_coords, x_coords = np.indices((rows2, cols2))
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(rows2 * cols2)])
    
    # Трансформуємо координати за допомогою гомографії
    transformed_coords = np.dot(coords, H.T)
    
    transformed_coords[:, 0] /= transformed_coords[:, 2]
    transformed_coords[:, 1] /= transformed_coords[:, 2]
    
    # Визначаємо межі області для об'єднання зображень
    min_x = min(0, np.min(transformed_coords[:, 0]))
    max_x = max(cols1, np.max(transformed_coords[:, 0]))
    min_y = min(0, np.min(transformed_coords[:, 1]))
    max_y = max(rows1, np.max(transformed_coords[:, 1]))
    
    # Створюємо сітку для виходного зображення
    grid_x, grid_y = np.mgrid[min_x:max_x, min_y:max_y]
    
    # Згладжуємо перехід між координатами
    dest_image_reshaped = img2.reshape(-1, 3)
    valid_points = ~np.all(dest_image_reshaped == 0, axis=1)
    dest_image_reshaped = dest_image_reshaped[valid_points]
    transformed_coords = transformed_coords[valid_points]

    # Інтерполюємо відсутні пікселі
    dst = griddata(transformed_coords[:, :2], dest_image_reshaped, (grid_x, grid_y), method='linear')
    
    # Використовуємо тільки валідні значення
    valid_idx = ~np.isnan(dst[..., 0])
    
    # Створюємо порожнє зображення, яке буде вміщувати обидва зображення
    output_img = np.zeros_like(dst, dtype=img1.dtype)
    
    # Вставляємо перше зображення
    output_img[grid_y - min_y, grid_x - min_x] = img1[min_y:max_y, min_x:max_x]
    
    # Вставляємо трансформоване зображення 2
    output_img[valid_idx] = dst[valid_idx]
    
    return output_img
    
# Set minimum match condition
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Конвертуємо ключові точки в аргумент для findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Встановлюємо гомографію
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Об'єднуємо зображення
    result = warpImages(img2, img1, M)


    # Відображаємо та зберігаємо результат
    #cv2.imshow('result', result)
    cv2.imwrite('panorama_result_ORB.jpg', result)  # Зберігаємо результат
    #cv2.waitKey(0)


gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Create a mask using thresholding (adjust parameters as needed)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour based on area
largest_contour = max(contours, key=cv2.contourArea)

# Find the convex hull of the largest contour
hull = cv2.convexHull(largest_contour)

#cnt = contours[0]  # Take the first contour

# Draw the chosen contour on the mask
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)

# ... Use the mask for object cutout, background removal, etc. ...

# Bitwise operations to make the background transparent
bg = np.zeros(result.shape, dtype=np.uint8)
bg[:, :, :3] = (255, 255, 255)

img_bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
img_fg = cv2.bitwise_and(result, result, mask=mask)

# Combine foreground and background
final = cv2.add(img_bg, img_fg)


cv2.imwrite('transparent1.jpg', final)