import argparse
import numpy as np
import pandas as pd
from string import ascii_lowercase
from sklearn.datasets import make_classification

import pyspark
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from parallelm.mlops import mlops as mlops
from parallelm.mlops import StatCategory as st
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.common.spark_pipeline_model_helper import SparkPipelineModelHelper


def parse_args():
    """
    Parse Arguments from component
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", help="Maximum number of iterations for Logistic Regression Algorithm")
    parser.add_argument("--output-model", help="Path of output model to create")
    parser.add_argument("--temp-shared-path", help="Shared path for model transfer")
    options = parser.parse_args()
    return options


def main():
    # Initialize spark and mlops
    spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()
    mlops.init(spark.sparkContext)

    # parse the arguments to component
    options = parse_args()

    # Generate synthetic data using scikit learn
    num_samples = 50
    num_features = 20
    
    X,y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=2, n_redundant=1, n_classes=3, n_clusters_per_class=1, random_state=42)
    X = X + np.random.uniform(0, 5) * np.random.normal(0, 1, (num_samples,num_features))
    feature_names = ["".join(ascii_lowercase[a]) for a in range(num_features + 1)]
    feature_names[0] = "label"

    # Create a spark dataframe from the synthetic data generated 
    trainingData = spark.createDataFrame(pd.DataFrame(np.concatenate((y.reshape(-1,1),X),axis=1),columns=feature_names))

    # Histogram of label distribution
    value, counts = np.unique(y, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    ########## Start of ParallelM instrumentation ##############
    # Report label distribution as a BarGraph
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:,0]).astype(str).tolist()).data((label_distribution[:,1]).tolist())
    mlops.set_stat(bar)
    ########## End of ParallelM instrumentation ################

    ########## Start of ParallelM instrumentation ############
    # Report the data distribution using mlops
    mlops.set_data_distribution_stat(trainingData)
    ########## End of ParallelM instrumentation ##############

    # Fit a logsitic regression model
    assembler = VectorAssembler(inputCols=feature_names[1:num_features+1], outputCol="features")
    classifier = LogisticRegression(maxIter=int(options.max_iter), regParam=0.3, elasticNetParam=0.8)

    pipeline=Pipeline(stages=[assembler,classifier])
    model = pipeline.fit(trainingData)
    predictions = model.transform(trainingData)

    # Select (prediction, true label) and compute training error
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    #################### Start of ParallelM instrumentation ################
    # Report accuracy of the chosen model
    mlops.set_stat("Accuracy", accuracy, st.TIME_SERIES)
    #################### End of ParallelM instrumentation ################
    
    # Save the spark model 
    SparkPipelineModelHelper()\
        .set_shared_context(spark_context=spark.sparkContext)\
        .set_local_path(local_path=options.output_model)\
        .set_shared_path_prefix(shared_path_prefix=options.temp_shared_path)\
        .save_sparkml_model(model)
    
    # Stop spark context and mlops
    spark.sparkContext.stop()
    mlops.done()
    


if __name__ == "__main__":
    main()
