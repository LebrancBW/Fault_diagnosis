#encoding:utf-8
'''
    random forest opreation
'''
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def random_forest_opreator(dataframe):
    '''
        input:
            Row(<feature>, <label>, <PCA_feature>)
        output:
            Row(<feature>, <label>, <PCA_feature>, <predicted_label>, <indexed_label>)

    '''
    #1 
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexed_label").fit(dataframe)
    rf = RandomForestClassifier(labelCol="indexed_label", featuresCol="PCA_feature", numTrees=5, maxDepth=15, maxBins=128)
    labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_label",labels=labelIndexer.labels)
    pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

    #2 
    model = pipeline.fit(dataframe)
    prediction = model.transform(dataframe)
    return prediction, model