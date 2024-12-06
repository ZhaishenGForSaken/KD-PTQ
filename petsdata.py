import os
import shutil

# 路径设置
data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 确保训练和测试目录存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 文件路径
trainval_path = os.path.join(data_dir, 'trainval.txt')
test_path = os.path.join(data_dir, 'test.txt')

# 将图片分到相应的训练或测试文件夹
def organize_images(file_path, target_dir):
    with open(file_path, 'r') as file:
        for line in file:
            image_name, class_id, species, breed_id = line.strip().split()
            folder_name = f"{species}_{breed_id}"
            source_path = os.path.join(images_dir, f"{image_name}.jpg")
            class_folder = os.path.join(target_dir, folder_name)

            os.makedirs(class_folder, exist_ok=True)
            if os.path.isfile(source_path):
                shutil.copy(source_path, class_folder)
            else:
                print(f"File not found: {source_path}")

# 处理训练和测试数据
organize_images(trainval_path, train_dir)
organize_images(test_path, test_dir)

print("数据集组织完毕！")