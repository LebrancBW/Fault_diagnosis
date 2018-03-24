#!/usr/local/spark/bin/spark-submit
#encoding:utf-8
'''
    load dataset from the HDFS directory and transfrom the dataset into DataFrame
    
'''
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession(sc)
def load_dataset(filePath):
    '''
        input the filepath of the dataset(csv format)
        outputformat:
            Row(<feature>, <label>)
    '''
    lines = sc.textFile(filePath)
    parts = lines.map(lambda l:l.split(", "))
    rdd = parts.map(lambda p:Row(feature = Vectors.dense(p[:-1]), label=p[-1]))
    return spark.createDataFrame(data=rdd)

if __name__ == 'main':
    print("this module cannot execute directly, try to import it instead")

