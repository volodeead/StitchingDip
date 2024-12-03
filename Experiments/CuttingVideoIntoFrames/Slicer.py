import cv2
import os

# Параметри
video_path = 'city_perfect_video.mp4'  # Шлях до відео
output_folder = 'frames_output'  # Папка для збереження кадрів
frames_per_second = 1  # Кількість кадрів для збереження кожну секунду

# Створення вихідної папки, якщо вона не існує
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Завантаження відео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не вдалося відкрити відео")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Отримуємо частоту кадрів відео
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Загальна кількість кадрів
video_duration = frame_count / fps  # Тривалість відео в секундах
print(f"Частота кадрів: {fps}, Загальна кількість кадрів: {frame_count}, Тривалість відео: {video_duration:.2f} секунд")

frame_interval = fps // frames_per_second  # Інтервал кадрів для збереження
frame_id = 0
saved_frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Збереження кадру, якщо він підходить за інтервалом
    if frame_id % frame_interval == 0:
        frame_name = f"frame_{saved_frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        saved_frame_count += 1
        print(f"Збережено кадр: {frame_name}")

    frame_id += 1

# Закриваємо відео
cap.release()
print(f"Готово! Збережено {saved_frame_count} кадрів у папку '{output_folder}'")