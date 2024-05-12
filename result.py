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
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img.close()  
            resized_img.save(image_path, quality=100) 
            return image_path
        else:
            return image_path
            
# def preprocess_image(image_path, max_size=4096):
#     with Image.open(image_path) as img:
#         width, height = img.size
#         if width > max_size or height > max_size:
#             scale = min(max_size / width, max_size / height)
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#             temp_path = image_path + ".temp.jpg"
#             resized_img.save(temp_path, quality=100)  # 指定保存质量
#             return temp_path
#         else:
#             return image_path

# Face++ API URL and link where  I purchase
http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
key = "FdP2aKhCjjNbknmi4xjI31KfOFEk8_FB"
secret = "jXA4MAyoCE1rG6Hqj1ljvVJ0UU4_ryHt"

folder_path = "D:/image_result"
# folder_path = "D:/pic_test"

csv_file_path = "D:/pic_results.csv"

return_attributes = "gender,age,smiling,eyestatus,headpose,facequality,blur,emotion,beauty,mouthstatus,eyegaze,skinstatus,nose_occlusion,chin_occlusion,face_occlusion"
i=0
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([
        "Image_Number", "Aspect_Ratio", "Gender", "Age", "Smile", "Pitch_Angle",
        "Roll_Angle", "Yaw_Angle", "Blurness", "Motionblur", "Gaussianblur",
        "Emotion", "Facequality", "Beauty", "Mouthstatus", "Skinstatus", "Glass"
    ])

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
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
                filepath = preprocess_image(filepath)

                with open(filepath, "rb") as image_file:
                    files = {"image_file": image_file}
                    data = {"api_key": key, "api_secret": secret, "return_attributes": return_attributes}

                    response = requests.post(http_url, data=data, files=files)

                    req_con = response.content.decode('utf-8')
                    req_dict = JSONDecoder().decode(req_con)

                    if 'error_message' in req_dict:
                        print(f"处理图片 {filename} 时出现错误: {req_dict['error_message']}")
                        continue  
                files['image_file'].close()

            # process individual face
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

                    # write in to csv
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



print("处理完成，结果已保存到CSV文件。")
