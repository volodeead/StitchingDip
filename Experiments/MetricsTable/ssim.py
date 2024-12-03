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