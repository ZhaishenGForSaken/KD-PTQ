import os
import shutil

def rename_folders_by_images(root_path):
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            # 检查文件夹内的所有文件
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    # 从文件名提取类名
                    class_name = filename.split('_')[0]
                    new_folder_path = os.path.join(root_path, class_name)
                    if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)
                    # 移动文件到新的文件夹
                    shutil.move(os.path.join(folder_path, filename), os.path.join(new_folder_path, filename))
            # 检查原始文件夹是否为空，如果为空，则删除
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
            print(f"Files from {folder_path} moved to {new_folder_path} and original folder removed if empty")

# 调用函数
rename_folders_by_images('data/test')
rename_folders_by_images('data/train')