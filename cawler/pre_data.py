# python批量更换后缀名
import datetime
import os
import sys

from tqdm import tqdm

os.chdir(r'D:\桌面\创作\AI预测高考作文\Spider-People-s-daily')

# 列出当前目录下所有的文件
files = os.listdir('./')
print('files', files)

for root, dirs, files in os.walk(r'D:\桌面\创作\AI预测高考作文\Spider-People-s-daily\data'):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>2", root, dirs, files)
    input_col = []
    output_col = []
    count = 0
    bar = tqdm(files,
               total=len(files),
               desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process NFT, POAP, Donation",
               ncols=150)
    for filename in bar:
        print('filename', filename)
        portion = os.path.splitext(filename)
        os.chdir(root)
        # 如果后缀是.dat
        if portion[1] == ".html":
            #把原文件后缀名改为 txt
            newName = portion[0] + ".txt"
            os.renames(filename, newName)