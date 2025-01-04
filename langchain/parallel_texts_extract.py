import os

# 定义双语数据文件夹路径
root_dir = '双语数据'  # 请将此修改为实际文件夹路径
output_file = 'modern_to_classical.txt'  # 输出的结果文件

# 打开输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    # 遍历文件夹及其子文件夹
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 只处理 bitext.txt 文件
            if file == 'bitext.txt':
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()  # 读取所有行
                    classical, modern = None, None  # 用于存储临时的古文和现代文
                    for line in lines:
                        line = line.strip()  # 去掉首尾空白字符
                        if line.startswith("古文："):
                            classical = line.replace("古文：", "").strip()
                        elif line.startswith("现代文："):
                            modern = line.replace("现代文：", "").strip()
                            # 如果同时有古文和现代文，写入文件
                            if classical and modern:
                                outfile.write(f'{modern}\t{classical}\n')
                                classical, modern = None, None  # 清空，等待下一组
