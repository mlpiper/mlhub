from __future__ import print_function

import argparse
from datetime import datetime, timedelta
from parallelm.mlops.examples import ab_testing_base
from parallelm.mlops.examples.utils import RunModes

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--confidence", type=int, default=95, help="confidence 90/95/99", choices=[90, 95, 99])
    parser.add_argument("--uplift", type=int, default=10, help="uplift percentage")
    parser.add_argument("--samplesA", default="samples", help="Stat storing samples of a pipeline")
    parser.add_argument("--samplesB", default="samples", help="Stat storing samples of b pipeline")
    parser.add_argument("--conversionsA", default="conversions", help="Stat storing conversions of a pipeline")
    parser.add_argument("--conversionsB", default="conversions", help="Stat storing conversions of b pipeline")
    parser.add_argument("--nodeA", help="node running a pipeline")
    parser.add_argument("--nodeB", help="node running b pipeline")
    parser.add_argument("--agentA", help="agent running a pipeline")
    parser.add_argument("--agentB", help="agent running b pipeline")
    parser.add_argument("--champion", default="A", help="champion model name")
    parser.add_argument("--challenger", default="B", help="challenger model name")
    options = parser.parse_args()

    return options


def main():
    options = parse_args()

    print("PM AB Comparator")

    now = datetime.utcnow()
    last_hour = (now - timedelta(hours=1))

    print("Hour before: {}".format(last_hour))
    print("Now:         {}".format(now))

    ab_testing_base.ab_test(options, last_hour, now, RunModes.PYTHON)


if __name__ == "__main__":
    main()
