import argparse
from string import ascii_lowercase

import numpy as np
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.common.spark_pipeline_model_helper import SparkPipelineModelHelper
from parallelm.mlops.stats.bar_graph import BarGraph

def parse_args():
    """
    Parse Arguments from component
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trees", help="Number of trees in the forest")
    parser.add_argument("--max-depth", help="Maximum depth of each tree in the forest")
    parser.add_argument("--output-model", help="Path of output model to create")
    parser.add_argument("--temp-shared-path", help="Shared path for model transfer")
    options = parser.parse_args()
    return options


def main():
    # Initialize spark and MLOps
    spark = SparkSession.builder.appName("RandomForestClassifier").getOrCreate()
    mlops.init(spark.sparkContext)

    # parse the arguments to component
    options = parse_args()
    print("PM: Configuration:")
    print("PM: Number of trees:                [{}]".format(options.num_trees))
    print("PM: Maximum depth:                  [{}]".format(options.max_depth))
    print("PM: Output model:                   [{}]".format(options.output_model))
    print("PM: Temp shared path:               [{}]".format(options.temp_shared_path))

    # Generate synthetic data using scikit learn
    num_samples = 50
    num_features = 20
    num_classes = 3
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=2, n_redundant=1,
                               n_classes=num_classes, n_clusters_per_class=1, random_state=42)
    X = X + np.random.uniform(0, 5) * np.random.normal(0, 1, (num_samples, num_features))

    feature_names = ["".join(ascii_lowercase[a]) for a in range(num_features + 1)]
    feature_names[0] = "label"

    # Create a spark dataframe from the synthetic data generated 
    trainingData = spark.createDataFrame(
        pd.DataFrame(np.concatenate((y.reshape(-1, 1), X), axis=1), columns=feature_names))

    # Histogram of label distribution
    value, counts = np.unique(y, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    # Output label distribution as a BarGraph using MCenter
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:, 0]).astype(str).tolist()).data(
        (label_distribution[:, 1]).tolist())
    mlops.set_stat(bar)

    # Output Health Statistics to MCenter
    # Report features whose distribution should be compared during inference
    mlops.set_data_distribution_stat(trainingData)

    # Fit a random forest classifiction model
    assembler = VectorAssembler(inputCols=feature_names[1:num_features + 1], outputCol="features")
    layers = [num_features, 5, 4, num_classes]
    classifier = RandomForestClassifier(numTrees=int(options.num_trees), maxDepth=int(options.max_depth))

    pipeline = Pipeline(stages=[assembler, classifier])
    model = pipeline.fit(trainingData)
    predictions = model.transform(trainingData)

    # Select (prediction, true label) and compute training error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Report accuracy of the chosen model using MCenter
    mlops.set_stat("Accuracy", accuracy, st.TIME_SERIES)

    # Save the spark model 
    SparkPipelineModelHelper() \
        .set_shared_context(spark_context=spark.sparkContext) \
        .set_local_path(local_path=options.output_model) \
        .set_shared_path_prefix(shared_path_prefix=options.temp_shared_path) \
        .save_sparkml_model(model)

    # Stop spark context and MLOps
    spark.sparkContext.stop()
    mlops.done()


if __name__ == "__main__":
    main()
