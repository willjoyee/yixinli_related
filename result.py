import requests
import os
import csv
from json import JSONDecoder, JSONDecodeError
from requests.exceptions import SSLError,Timeout
from PIL import Image
def preprocess_image(image_path, max_size=4096):
    with Image.open(image_path) as img:
        width, height = img.size
        if width > max_size or height > max_size:
            # 计算缩放比例
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # 调整图像大小
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # 直接覆盖保存原图像
            img.close()  # 需要先关闭原图像，以便覆盖保存
            resized_img.save(image_path, quality=100)  # 指定保存质量
            return image_path
        else:
            # 图像大小符合要求，无需调整
            return image_path
# 图像预处理函数，确保图像不超过最大尺寸
# def preprocess_image(image_path, max_size=4096):
#     with Image.open(image_path) as img:
#         width, height = img.size
#         if width > max_size or height > max_size:
#             # 计算缩放比例
#             scale = min(max_size / width, max_size / height)
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             # 调整图像大小
#             resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#             # 保存到临时文件
#             temp_path = image_path + ".temp.jpg"
#             resized_img.save(temp_path, quality=100)  # 指定保存质量
#             return temp_path
#         else:
#             # 图像大小符合要求，无需调整
#             return image_path

# Face++ API URL和密钥
http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
key = "FdP2aKhCjjNbknmi4xjI31KfOFEk8_FB"
secret = "jXA4MAyoCE1rG6Hqj1ljvVJ0UU4_ryHt"

# 要分析的图片文件夹路径
folder_path = "D:/image_result"
# folder_path = "D:/pic_test"

# CSV文件的路径
csv_file_path = "D:/pic_results.csv"

# 返回属性
return_attributes = "gender,age,smiling,eyestatus,headpose,facequality,blur,emotion,beauty,mouthstatus,eyegaze,skinstatus,nose_occlusion,chin_occlusion,face_occlusion"
i=0
# 准备CSV文件和写入器
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    # 写入CSV头部
    csv_writer.writerow([
        "Image_Number", "Aspect_Ratio", "Gender", "Age", "Smile", "Pitch_Angle",
        "Roll_Angle", "Yaw_Angle", "Blurness", "Motionblur", "Gaussianblur",
        "Emotion", "Facequality", "Beauty", "Mouthstatus", "Skinstatus", "Glass"
    ])

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 提取图片编号
                image_number = filename.split('第')[1].split('张')[0]
                # image_id = filename.split('第')[1].split('张')[0]
                # match = re.search(r"第\d+张(\d+)", filename)
                # if match:
                #     image_id = match.group(1)
                #     print("提取到的图片id:", image_id)
                # else:
                #     print("未找到匹配的图片id")

                filepath = os.path.join(folder_path, filename)
                print(image_number)
                # 预处理图像
                filepath = preprocess_image(filepath)

                # 打开文件
                with open(filepath, "rb") as image_file:
                    files = {"image_file": image_file}
                    data = {"api_key": key, "api_secret": secret, "return_attributes": return_attributes}

                    # 发送请求
                    response = requests.post(http_url, data=data, files=files)

                    # 解析响应
                    req_con = response.content.decode('utf-8')
                    req_dict = JSONDecoder().decode(req_con)

                    # 检查是否有错误消息
                    if 'error_message' in req_dict:
                        print(f"处理图片 {filename} 时出现错误: {req_dict['error_message']}")
                        continue  # 跳过当前图片，继续下一张
                files['image_file'].close()

            # 处理每张脸的数据
                for face in req_dict['faces']:
                    face_rectangle = face['face_rectangle']
                    attributes = face['attributes']
                    aspect_ratio = face_rectangle['width'] / face_rectangle['height']
                    gender = attributes['gender']['value']
                    age = attributes['age']['value']
                    smile = attributes['smile']['value']
                    pitch_angle = attributes['headpose']['pitch_angle']
                    roll_angle = attributes['headpose']['roll_angle']
                    yaw_angle = attributes['headpose']['yaw_angle']
                    blurness = attributes['blur']['blurness']['value']
                    motionblur = attributes['blur']['motionblur']['value']
                    gaussianblur = attributes['blur']['gaussianblur']['value']
                    emotion = max(attributes['emotion'], key=attributes['emotion'].get)
                    facequality = attributes['facequality']['value']
                    beauty = attributes['beauty']['male_score'] if gender == 'Male' else attributes['beauty'][
                        'female_score']
                    mouthstatus = max(attributes['mouthstatus'], key=attributes['mouthstatus'].get)
                    skinstatus = max(attributes['skinstatus'], key=attributes['skinstatus'].get)
                    glass = attributes['glass']['value']

                    # 写入CSV
                    csv_writer.writerow([
                        image_id, image_number, aspect_ratio, gender, age, smile, pitch_angle,
                        roll_angle, yaw_angle, blurness, motionblur, gaussianblur,
                        emotion, facequality, beauty, mouthstatus, skinstatus, glass
                    ])
            except JSONDecodeError as e:
                print(f"处理图片 {filename} 时发生JSON解码错误: {e}")
                continue  # 跳过当前图片，继续下一张
            except SSLError as e:
                print(f"处理图片 {filename} 时发生SSL错误: {e}")
                continue  # 跳过当前图片，继续下一张
            except (ConnectionError, Timeout) as e:
                print(f"处理图片 {filename} 时发生网络连接错误: {e}")
                continue  # 跳过当前图片，继续下一张
            except Exception as e:
                print(f"处理图片 {filename} 时发生其他异常: {e}")
                continue  # 跳过当前图片，继续下一张

            # 提取图片编号
            # image_number = filename.split('第')[1].split('张')[0]
            # filepath = os.path.join(folder_path, filename)
            # print(image_number)
            # filepath = preprocess_image(filepath)
            # # 调用Face++ API
            # files = {"image_file": open(filepath, "rb")}
            # data = {"api_key": key, "api_secret": secret, "return_attributes": return_attributes}
            # response = requests.post(http_url, data=data, files=files, verify=False)
            # try:
            #     req_con = response.content.decode('utf-8')
            #     req_dict = JSONDecoder().decode(req_con)
            # except JSONDecodeError as e:
            #     print('JSON decode error:', e)
            #     continue  # 如果发生错误，跳过当前图片的处理
            #     # 检查是否有错误消息
            # if 'error_message' in req_dict:
            #     print(f"处理图片 {filename} 时出现错误: {req_dict['error_message']}")
            #     continue  # 跳过当前图片，继续下一张
            # # 关闭文件
            # files['image_file'].close()


print("处理完成，结果已保存到CSV文件。")