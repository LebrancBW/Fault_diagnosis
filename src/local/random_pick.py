#!/usr/bin/python3
#encoding:utf-8
'''
    随机从数据集中选取数据组成训练集和测试集,方式：分层抽样
    因为数据集中的数据按照样本标签统一放置，故分层抽样更能平均反应样本性能
    测试集：训练集 = 1 : 3
'''
import random
from itertools import dropwhile
from itertools import compress

def main():
    '''
        主程序
    '''
    prefix = "./dataset/raw/"
    filepath = prefix + "steel_plate"
    output_train = prefix + "train_set"
    output_validate = prefix + "validate_set"
    
    # 打开数据文件
    f = open(filepath, 'r')
    content = f.readlines()

    # 生成随机下标
    random_index = list(map(lambda item:random.randint(item, item+4) ,range(0, len(content) - 1, 4)))

    sub_index = filter(lambda i:i not in random_index, range(len(content)))

    # 提取数据集 存到文件中
    train_file = open(output_train, 'w')
    train_file.writelines(compress(content, sub_index))
    validate_file = open(output_validate, 'w')
    validate_file.writelines(compress(content, random_index))


    f.close()
    validate_file.close()
    train_file.close()

if __name__ == '__main__':
    main()