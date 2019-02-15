from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.linalg import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline, PipelineModel
from parallelm.mlops import mlops as pm
from parallelm.mlops import StatCategory as st
from parallelm.mlops.stats.graph import MultiGraph
from parallelm.mlops.common.spark_pipeline_model_helper import SparkPipelineModelHelper
import numpy as np
from pyspark.sql.functions import sum, sqrt, min, max
import math
import pandas as pd
import argparse
from random import *


def Histogram_calc(df,spark, num_bins, label, pm_options, cg):
    """
    The function calculates the histogram with num_bins bins
    :param df:
    :param spark:
    :param num_bins: number of bins
    :param label: label for the histogram
    :param pm_options:
    :param cg: graph to add
    :return:
    """

    # Find the range between (mean +- 2*sigma)
    for column in df.columns:
        df_col = df.select(column).rdd.flatMap(lambda x: x)
        mean_df_col = df_col.mean()
        stdv_df_col = df_col.stdev()
        min_edge = mean_df_col - 2 * stdv_df_col
        max_edge = mean_df_col + 2 * stdv_df_col

        # Calculate the histogram
        if np.abs(max_edge - min_edge) <= 0.01 :
            edges = [np.finfo(np.float64).min, min_edge - 0.01, min_edge + 0.01, np.finfo(np.float64).max]
            bins = [0, 1, 0]
        else:
            edges = [np.finfo(np.float64).min]
            edges1 = np.linspace(min_edge, max_edge, num = num_bins).tolist()
            edges.extend(edges1)
            edges.append(np.finfo(np.float64).max)
            bins = df_col.histogram(edges)

        # Add a line graph of the histogram
        edges_avg = []
        edges_avg.append(edges[1])
        for index_hist1 in range(2,len(edges)-1):
            edges_avg.append(0.5*(edges[index_hist1] + edges[index_hist1-1]))
        edges_avg.append(edges[len(edges)-2])
        print("edges_avg = ", edges_avg)
        print("bins[1] = ", bins[1])
        cg.add_series(label="distance for " + str(label), x=edges_avg, y=bins[1])

    return


def eq_dist(x1, y1):
    """
    Calculate Euclidean distance between 2 vectors
    :param x1:
    :param y1:
    :return:
    dist: distance between vectors
    """
    x = np.asarray(x1)
    y = np.asarray(y1)
    len_x = len(x)
    len_y = len(y)
    if not len_x == len_y:
        print("error: elements not of the same shape")
        dist = 0
    else:
        dist = np.sqrt(np.sum((x-y)**2))
    return float(dist)


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", help="Path of input model to create")
    parser.add_argument("--input-file", help = "Path of the file to process")
    parser.add_argument("--threshold", help = "Threshold on minimum distance from closes center")
    parser.add_argument("--temp-shared-path", help="Temporary shared path for model transfer")
    options = parser.parse_args()
    return options


def main():
    import tarfile
    import os
    import time
    import shutil
    import tempfile
    pm_options = parse_args()
    spark = SparkSession.builder.appName("KmeansInference").getOrCreate()
    sc = spark.sparkContext

    pm.init(sc)

    # Get input model from input file. Untar it. end if file not there
    if pm_options.input_model is not None:
        try:

            print("pm_options.input_model = ", pm_options.input_model)
            dirpath = tempfile.mkdtemp()
            print("Untarring to {}".format(dirpath))
            shutil.copyfile(pm_options.input_model, dirpath + "/rsKmeans.tar")
            print("File found, path changed")
            time.sleep(5)
            tar_obj = tarfile.open(dirpath + "/rsKmeans.tar", mode='r:gz')
            pm.set_stat("model_file", 1)

        except Exception as e:
            print("Model not found")
            print("Got exception: " + str(e))
            pm.set_stat("model_file", 0)
            sc.stop()
            spark.stop()
            pm.done()
            return 0
    tar_obj.extractall(dirpath)
    tar_obj.close()
    time.sleep(3)

    # Load the kmeans model
    model_kmeans = \
        SparkPipelineModelHelper()\
        .set_shared_context(spark_context=sc)\
        .set_local_path(local_path=dirpath + "/KMEANS_MODEL")\
        .set_shared_path_prefix(shared_path_prefix=pm_options.temp_shared_path)\
        .load_sparkml_model()
    shutil.rmtree(dirpath)

    # Load the data
    input_data = (spark.read.format("csv")
                   .option("header", "true")
                   .option("ignoreLeadingWhiteSpace", "true")
                   .option("ignoreTrailingWhiteSpace", "true")
                   .option("inferschema", "true")
                   .load(pm_options.input_file)).repartition(10)

    # Number of samples in the file
    num_samples = input_data.count()
    num_bins = 11

    # Conversions are the number of samples whose distance to nearest cluster is below a threshold
    predicted_df = model_kmeans.transform(input_data)
    kmeans_centers = model_kmeans.stages[1].clusterCenters()
    num_conversions = 0
    cg2 = MultiGraph().name("Distance Distribution").set_continuous()
    graph_add = False
    for centerIndex in range(0, len(kmeans_centers)):
        # Filter the samples belonging to cluster index `centerIndex`
        filtered_df = predicted_df.filter(predicted_df["prediction"] == centerIndex)
        num_points = filtered_df.count()
        # Filter points that are less than threshold distance from the respective centers
        filtered_df = filtered_df.withColumn('distances', udf(eq_dist, FloatType())(col("features"), array([lit(v) for v in kmeans_centers[centerIndex]])))
        filtered_df_threshold = filtered_df.filter(filtered_df["distances"] < pm_options.threshold)
        num_conversions += filtered_df_threshold.count()
        # add prediction histogram
        if (num_points > num_bins):
            Histogram_calc(filtered_df.select("distances"), spark, num_bins=num_bins, label = "center " + str(centerIndex), pm_options=pm_options, cg=cg2)
            graph_add = True
    if graph_add:
        cg2.annotate(label="distance threshold = {}".format(int(pm_options.threshold)), x=int(pm_options.threshold))
        cg2.x_title("Distance")
        cg2.y_title("#samples in bin")
        pm.set_stat(cg2)


    try:
        pm.set_stat("samples", num_samples, st.TIME_SERIES)
        pm.set_stat("conversions", num_conversions, st.TIME_SERIES)

    except Exception as e:
        print("Got exception while getting stats: {}".format(e))
        pm.set_stat("error", 1, st.TIME_SERIES)

    sc.stop()
    pm.done()


if __name__ == "__main__":
    main()
