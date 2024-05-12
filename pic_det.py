import face_recognition
import os
import shutil
import face_recognition

print(face_recognition.__version__)
source_folder = 'D:/WUJia/phd-project/YiXinLi/pythonProject1/coverimage'
target_folder = 'D:/WUJia/phd-project/YiXinLi/pythonProject1/coverimage/coverimage_result'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for image_name in os.listdir(source_folder):
    image_path = os.path.join(source_folder, image_name)

    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            shutil.copy2(image_path, os.path.join(target_folder, image_name))

print("人脸检测完成")
