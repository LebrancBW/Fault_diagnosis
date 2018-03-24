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

    train_set = load_dataset(TRAIN_PATH)
    validate_set = load_dataset(VALIDATE_PATH)
    #2 model define
    
    PCA_model = pca_opreator(2)
    rf_model = random_forest_opreator()
    pipeline = Pipeline(stages=[PCA_model, rf_model])
    model = pipeline.fit(train_set)
    #3
    train_dataframe = model.transform(train_set)
    validate_dataframe = model.transform(validate_set)

    validate_dataframe.show(10)
    #4 evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="accuracy")
    accuracy_train = evaluator.evaluate(train_dataframe)
    accuracy_validate = evaluator.evaluate(validate_dataframe)
    print("accuracy on train set = %g" % (accuracy_train))
    print("accuracy on validation set = %g" % (accuracy_validate))
if __name__ == '__main__':
    main()