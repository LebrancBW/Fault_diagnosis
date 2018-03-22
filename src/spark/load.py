#!/usr/local/spark/spark-submit
#encoding:utf-8
'''
    load dataset from the HDFS directory and transfrom the dataset into DataFrame
    
'''
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

def load_dataset(filePath):
    '''
        input the filepath of the dataset(csv format)
    '''
    sc = spark.sparkContext
    lines = sc.textFile(filePath)
    parts = lines.map(lambda l:l.spilt(", "))
    return parts.map(lambda p:Row(feature = Vectors.dense(p[:-1]), label=p[-1])) 


