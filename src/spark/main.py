#!/usr/local/spark/bin/spark-submit
#encoding:utf-8
'''
    load dataset from the HDFS directory and transfrom the dataset into DataFrame
    
'''
from load_data import load_dataset
from dimesion_reduce import pca_opreator
from random_forest import random_forest_opreator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    '''
        combine all the opreations together
    '''
    #1 define the path 
    FILE_DIRECTORY = "hdfs://master:9000/user/hadoop/fault_diagnosis/dataset/"
    TRAIN_PATH = FILE_DIRECTORY + "train_set.csv"
    VALIDATE_PATH = FILE_DIRECTORY + "validate_set.csv"

    #2 train
    rdd = load_dataset(TRAIN_PATH)
    dataframe1, PCA_model = pca_opreator(rdd, 2)
    dataframe2, rf_model = random_forest_opreator(dataframe1)
    # model = Pipeline(stages=[PCA_model, rf_model])
    # rdd2 = load_dataset(VALIDATE_PATH)
    # dataset2 = model.transfrom(rdd2)
    dataframe2.show(20)

    #3 evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(dataframe2)
    print("Test Error = %g" % (1.0 - accuracy))
if __name__ == '__main__':
    main()