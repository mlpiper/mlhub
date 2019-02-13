from __future__ import print_function

from datetime import datetime, timedelta

from parallelm.mlops.examples import canary_comparator_base
from parallelm.mlops.examples.utils import RunModes
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictionHistogramA", default="predictionHistogram",
                        help="Stat key that stores prediction histogram of main pipeline")
    parser.add_argument("--predictionHistogramB", default="predictionHistogram",
                        help="Stat key that stores prediction histogram of canary pipeline")
    parser.add_argument("--nodeA", help="node running main pipeline")
    parser.add_argument("--nodeB", help="node running canary pipeline")
    parser.add_argument("--agentA", help="agent running main pipeline")
    parser.add_argument("--agentB", help="agent running canary pipeline")
    options = parser.parse_args()

    return options


def main():
    options = parse_args()

    print("Canary comparator")

    now = datetime.utcnow()
    last_hour = (now - timedelta(hours=1))

    print("Hour before: {}".format(last_hour))
    print("Now:         {}".format(now))

    canary_comparator_base.canary_comparator(options, last_hour, now, RunModes.PYTHON)


if __name__ == "__main__":
    main()
