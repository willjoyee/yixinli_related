import face_recognition
import os
import shutil
import face_recognition

print(face_recognition.__version__)
# 指定原始图片文件夹和目标文件夹
source_folder = 'D:/WUJia/phd-project/YiXinLi/pythonProject1/coverimage'
# 目标文件夹即输出结果的文件夹
target_folder = 'D:/WUJia/phd-project/YiXinLi/pythonProject1/coverimage/coverimage_result'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
#  如果不存在目标文件夹,则创造一个

# 遍历原始文件夹中的图片文件
for image_name in os.listdir(source_folder):
    image_path = os.path.join(source_folder, image_name)

    # 确保是文件而且是图片
    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 加载图片并检测人脸
        image = face_recognition.load_image_file(image_path)
        # 基于hog机器学习模型进行人脸识别，不能使用gpu加速
        face_locations = face_recognition.face_locations(image)

        # alternative model
        # 基于cnn识别人脸,是否使用gpu看装机环境
        # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

        # 我注释掉的



        # 如果检测到人脸，复制图片到目标文件夹
        if len(face_locations) > 0:
            shutil.copy2(image_path, os.path.join(target_folder, image_name))

print("人脸检测完成")