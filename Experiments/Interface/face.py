import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

import sys

# Додаємо шлях до кореневої папки в PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from newStitching_FotoFiltering_FotoProcessing_Threding_Diploma import stitch_image_in_sub_directory

# Додаємо шлях до папки з Exp_in_derectory_Analis.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Experiments/MetricsTable')))

from Exp_in_derectory_Analis import load_images_in_pairs

class PanoramaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panorama Stitcher")
        self.directory_path = None  # Змінна для збереження шляху до директорії
        self.stitching_thread = None  # Потік для склеювання
        self.stop_event = threading.Event()  # Подія для зупинки
        self.create_main_window()

    def create_main_window(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="PANORAMA STITCHER", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        # Кнопка для вибору директорії
        ttk.Button(frame, text="Вибрати директорію з фото", command=self.select_directory).grid(row=1, column=0, columnspan=2, pady=10)

        # Місце для відображення вибраного шляху
        self.directory_label = ttk.Label(frame, text="Директорію не вибрано", font=("Arial", 10), foreground="red")
        self.directory_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Місце для відображення кількості зображень
        self.image_count_label = ttk.Label(frame, text="Знайдено: N зображень", font=("Arial", 12))
        self.image_count_label.grid(row=3, column=0, columnspan=2, pady=5)

        self.save_intermediate = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Зберігати проміжні результати", variable=self.save_intermediate).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(frame, text="Почати склейку", command=self.start_processing).grid(row=5, column=0, columnspan=2, pady=10)


    def select_directory(self):
        # Відкриття діалогового вікна вибору директорії
        selected_directory = filedialog.askdirectory(title="Виберіть директорію з фото")

        if selected_directory:
            # Зберігаємо шлях до змінної
            self.directory_path = selected_directory

            # Рахуємо кількість зображень у директорії
            image_count = sum(
                1 for file in os.listdir(selected_directory)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))
            )

            # Оновлюємо текст в Label для шляху
            self.directory_label.config(text=f"Вибрано: {self.directory_path}", foreground="green")

            # Оновлюємо текст з кількістю зображень
            self.image_count_label.config(
                text=f"Знайдено: {image_count} зображень" if image_count > 0 else "Зображень не знайдено",
                foreground="black" if image_count > 0 else "red"
            )
        else:
            # Якщо користувач скасував вибір
            self.directory_label.config(text="Директорію не вибрано", foreground="red")
            self.image_count_label.config(text="Знайдено: N зображень", foreground="black")

    def start_processing(self):
        if not self.directory_path:
            self.directory_label.config(text="Будь ласка, виберіть директорію перед початком!", foreground="red")
            return

        # Показуємо вікно очікування
        self.create_processing_window()

        # Запускаємо склеювання у новому потоці
        self.stop_event.clear()
        self.stitching_thread = threading.Thread(target=self.run_stitching)
        self.stitching_thread.start()

    def create_processing_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="ПРОЦЕС ПАНОРАМУВАННЯ", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(frame, text=f"Склейка виконується з директорії:\n{self.directory_path}", font=("Arial", 10)).grid(row=1, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", length=300)
        self.progress.grid(row=2, column=0, columnspan=2, pady=10)

        self.progress_label = ttk.Label(frame, text="Прогрес: 0%", font=("Arial", 10))
        self.progress_label.grid(row=3, column=0, columnspan=2, pady=5)

        ttk.Button(frame, text="[Стоп]", command=self.stop_processing).grid(row=4, column=0, columnspan=2, pady=10)

    def stop_processing(self):
        self.stop_event.set()
        if self.stitching_thread and self.stitching_thread.is_alive():
            self.stitching_thread.join(timeout=1)
        self.root.destroy()

    def run_stitching(self):
        try:
            output_directory = "output"

            def update_progress(current, total):
                progress_percentage = (current / total) * 100
                self.progress["value"] = progress_percentage
                self.progress_label.config(text=f"Прогрес: {int(progress_percentage)}%")
                self.root.update_idletasks()

            stitch_image_in_sub_directory(self.directory_path, output_directory, update_progress)

            if not self.stop_event.is_set():
                self.create_results_window()
        except Exception as e:
            self.directory_label.config(text=f"Помилка склеювання: {e}", foreground="red")

    def create_results_window(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="РЕЗУЛЬТАТИ ПАНОРАМУВАННЯ", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(frame, text="Склейка завершена успішно", font=("Arial", 12)).grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="[Аналіз якості]", command=self.create_quality_analysis).grid(row=2, column=0, pady=10)
        ttk.Button(frame, text="[Зберегти панораму]").grid(row=2, column=1, pady=10)

    def create_quality_analysis(self):
        # Очищуємо головне вікно
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="АНАЛІЗ ЯКОСТІ", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        # Прогрес-бар
        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", length=300)
        self.progress.grid(row=1, column=0, columnspan=2, pady=10)

        # Метка для прогресу
        self.progress_label = ttk.Label(frame, text="Прогрес: 0%", font=("Arial", 12))
        self.progress_label.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(frame, text="Скасувати", command=self.stop_analysis).grid(row=3, column=0, pady=10)

        # Запускаємо аналіз у новому потоці
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.start()

    def stop_analysis(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1)
        self.root.destroy()

    def run_analysis(self):
        try:
            # Шлях до директорії для аналізу
            analysis_directory = os.path.join("output", "Analysis")

            if not os.path.exists(analysis_directory):
                self.progress_label.config(text="Директорія для аналізу не знайдена!", foreground="red")
                return

            # Отримуємо кількість пар зображень
            total_pairs = sum(
                len(files) - 1
                for _, _, files in os.walk(analysis_directory)
                if len(files) > 1
            )

            if total_pairs <= 0:
                self.progress_label.config(text="Немає пар для аналізу!", foreground="red")
                return

            def update_progress(current, total):
                progress_percentage = (current / total) * 100
                self.progress["value"] = progress_percentage
                self.progress_label.config(text=f"Прогрес: {int(progress_percentage)}%")
                self.root.update_idletasks()

            # Викликаємо функцію аналізу з прогрес-баром
            load_images_in_pairs(analysis_directory, progress_callback=update_progress)

            # Завершення
            self.progress_label.config(text="Аналіз завершено успішно!", foreground="green")

            # Автоматичний перехід до вікна аналізу результатів
            self.create_analysis_window()

        except Exception as e:
            self.progress_label.config(text=f"Помилка: {str(e)}", foreground="red")


    def create_analysis_window(self):
        # Завантажуємо дані з JSON
        import json
        json_file_path = "D:\\Programing\\Diplomna\\testes.json"
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)
        except Exception as e:
            print(f"Не вдалося завантажити JSON файл: {e}")
            return

        # Очищення вікна
        for widget in self.root.winfo_children():
            widget.destroy()

        # Створення фрейму
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="ДЕТАЛЬНИЙ АНАЛІЗ", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        # Створення дерева
        self.tree = ttk.Treeview(frame)
        self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        # Групування даних
        groups = {}
        for test in data.get("Tests", []):
            group_id = test["Group"]
            iteration = test["Iteration"]
            if group_id not in groups:
                groups[group_id] = {}
            groups[group_id][iteration] = test

        # Додаємо дані в дерево
        self.tree_items = {}  # Для зберігання метрик
        for group_id, iterations in groups.items():
            group_item = self.tree.insert("", "end", text=f"Група {group_id}", open=False)
            for iteration_id, test_data in iterations.items():
                iteration_item = self.tree.insert(group_item, "end", text=f"Ітерація {iteration_id}")
                self.tree_items[iteration_item] = test_data  # Зберігаємо дані для вузла

        # Зв'язуємо вибір вузла дерева з функцією
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Створення секції для відображення метрик
        self.metrics_frame = ttk.Frame(frame)
        self.metrics_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        self.metrics_label = ttk.Label(self.metrics_frame, text="Обрані метрики ітерації будуть показані тут.", font=("Arial", 10))
        self.metrics_label.pack()

        # Кнопки
        ttk.Button(frame, text="← Назад", command=self.create_results_window).grid(row=3, column=0, pady=10)
        ttk.Button(frame, text="[Зберегти всі метрики в JSON]").grid(row=3, column=1, pady=10)

    def on_tree_select(self, event):
        # Отримуємо обраний елемент дерева
        selected_item = self.tree.selection()[0]
        
        # Оновлюємо метрики відповідно до вузла
        test_data = self.tree_items.get(selected_item)
        if test_data:
            metrics = test_data["Metrics"]
            angle = test_data["Angle"]
            group = test_data["Group"]
            iteration = test_data["Iteration"]
            metrics_text = (
                f"Група: {group}\n"
                f"Ітерація: {iteration}\n"
                f"Кут: {angle:.2f}\n"
                f"SSIM: {metrics['SSIM']:.4f}\n"
                f"Err_score: {metrics['Err_score']:.4f}\n"
                f"Art_score: {metrics['Art_score']:.4f}\n"
                f"KT: {metrics['KT']:.4f}\n"
                f"Transform_ratio: {metrics['Transform_ratio']:.4f}\n"
                f"Smooth_trans_ratio: {metrics['Smooth_trans_ratio']:.4f}\n"
                f"Artifact_count: {metrics['Artifact_count']}"
            )
            self.metrics_label.config(text=metrics_text)
        else:
            self.metrics_label.config(text="Обрані метрики ітерації будуть показані тут.")


# Запуск програми
if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramaApp(root)
    root.mainloop()
