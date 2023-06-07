"""
作者：ZWP
日期：2023.06.07
"""
import os
import random
#random.sample()
#try:except
#str.join(list)

def main():
    random.seed(20030920)
    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    files_name=([file.split(".")[0] for file in os.listdir(files_path)])
    files_num=len(files_name)
    val_index=random.sample(range(0,files_num),k=int(files_num*val_rate))
    train_files,val_files=[],[]
    for index,filename in enumerate(files_name):
        if index in val_index:
            val_files.append(filename)
        else:
            train_files.append(filename)
    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)

if __name__== '__main__':
    main()

