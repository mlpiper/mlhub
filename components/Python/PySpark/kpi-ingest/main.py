from __future__ import print_function

from pyspark import SparkContext

import argparse
import os
import re
import time

# $example off$

from parallelm.mlops import mlops as pm
from parallelm.mlops.stats.kpi_value import KpiValue


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--kpi-file", help="KPI file to use as input")
    parser.add_argument("--generate", action="store_true", default=False, help="Generate sample kpi file")
    options = parser.parse_args()
    return options


def generate_sample_kpi_file(options):
    """
    This is a method to generate a sample KPI file to be used for testing
    :param options: program options from command line
    :return:
    """
    kpi_name = "kpi_example"
    default_file = "/tmp/kpi.csv"
    if options.kpi_file is None:
        print("Using default file")
        options.kpi_file = default_file

    f = open(options.kpi_file, "w")

    now_in_sec = int(time.time())
    ten_minutes_in_sec = 60*10

    for i in range(0, 30):
        ts = now_in_sec - (i * ten_minutes_in_sec)
        kpi_line = "{}, {}, {}\n".format(kpi_name, ts, i)
        f.write(kpi_line)
        f.flush()
    f.close()


def report_kpi_stats(file_present=True, error_reading_file=0, number_of_kpis=0):
    pm.set_stat("KPI_file_present", int(file_present))
    pm.set_stat("Error reading file", int(error_reading_file))
    pm.set_stat("Number of KPIs lines", number_of_kpis)


def process_kpi_line(kpi_line):
    parts = re.split("\s*,\s*", kpi_line)
    if len(parts) < 3:
        raise Exception("KPI line is expected to have at least 3 parts: name, timestamp, value")

    kpi_name = parts[0]
    kpi_ts = parts[1]
    kpi_value = parts[2]
    try:
        kpi_value = float(kpi_value)
    except Exception as e:
        print("Error converting kpi_value to float")
        raise e

    print("detected KPI: name: [{}] timestamp: [{}] value: [{}]".format(kpi_name, kpi_ts, kpi_value))
    pm.set_kpi(kpi_name, kpi_value, kpi_ts, KpiValue.TIME_SEC)


def main():
    options = parse_args()

    if options.generate:
        generate_sample_kpi_file(options)
        exit(0)

    sc = SparkContext(appName="kpi-ingest")

    pm.init(sc)

    print("Ingesting kpi from a file")
    print("Provided file is: {}".format(options.kpi_file))

    is_file_present = False
    error_processing_file = False
    number_of_kpis = 0

    try:
        if options.kpi_file is not None:
            is_file_present = os.path.exists(options.kpi_file)
            if is_file_present:
                kpi_lines = open(options.kpi_file).readlines()
                number_of_kpis = len(kpi_lines)
                for kpi_line in kpi_lines:
                    process_kpi_line(kpi_line)
    except Exception as e:
        print("Got exception while trying to process file containing KPI values: {}".format(e))
        error_processing_file = True

    report_kpi_stats(file_present=is_file_present, error_reading_file=error_processing_file,
                     number_of_kpis=number_of_kpis)
    print("Done reporting KPI statistics")

    sc.stop()
    pm.done()


if __name__ == "__main__":
    main()
