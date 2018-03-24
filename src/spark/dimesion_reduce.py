#!/usr/local/spark/spark-submit
#encoding:utf-8
'''
    execute PCA oprecation on dataframe
'''
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
def pca_opreator(k):
    '''
        do PCA on dataset
        inputformat:(DataFrame)
            Row(<feature>, <label>)
        outputformat:(DataFrame)
            Row(<feature>, <label>, <PCA_feature>)
    '''
    # df = SQLContext.createDateFrame(RDD)
    model = PCA(k=k, inputCol='feature', outputCol='PCA_feature')
    # pca_df = model.transform(df)
    # model.write().overwrite().save("fault_diagnosis/models/pca.model")
    # pca_df.select('pca_feature').show(truncate=False)
    return model


if __name__ == 'main':
    print("this module cannot execute directly, try to import it instead")