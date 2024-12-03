import pycolmap

# Завантажте проект COLMAP
project = pycolmap.Project.load("D:\Programing\Cursova\output\database.db")

# Додайте фотографії до проекту
project.add_images("D:\Programing\Cursova\project\source")

# Запустіть об'єднання фотографій
project.run_mapper()

# Збережіть результат
project.export_ply("D:\Programing\Cursova\output\map.ply")
