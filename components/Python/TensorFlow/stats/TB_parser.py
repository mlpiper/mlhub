from __future__ import print_function

import argparse
import itertools
import os
import math
import sys
import time

from parallelm.mlops import mlops

import tensorflow as tf
from tensorflow.python.platform import gfile

tf_version = tf.__version__
major, minor, patch = tf_version.split(".")


if major == "1" and minor == "1":
    from tensorflow.tensorboard.backend.event_processing import event_accumulator
    from tensorflow.tensorboard.backend.event_processing import event_file_loader
else:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.backend.event_processing import event_file_loader


def generate_event_from_file(filepath):
    return event_file_loader.EventFileLoader(filepath).Load()


class TBParser:
    def __init__(self):
        self._sleep_time = 1
        self._log_dir = None
        self._time_stamp_start = 0
        self._verbose = False

        self._file_size = {}
        self._file_time = {}
        self._print_prefix = "tb_parser: "

    def verbose(self, verbose):
        self._verbose = verbose
        return self

    def _print_verbose(self, msg):
        if self._verbose:
            print(self._print_prefix + msg, file=sys.stderr)

    def _print(self, msg):
        print(self._print_prefix + msg, file=sys.stderr )

    def log_dir(self, log_dir):
        self._log_dir = log_dir
        return self

    def time_stamp_start(self, time_stamp_start):
        self._time_stamp_start = time_stamp_start
        return self

    def sleep_time(self, sleep_time):
        self._sleep_time = sleep_time
        return self

    def _report_event(self, tb_parse_event, time_stamp_start):
        """
        Process the TensorBoard events
        only `summary` events are scanned and updated using
        mlops-stats API, entries supported are scalar values only.
        """
        if tb_parse_event.HasField('summary') and (time_stamp_start < tb_parse_event.wall_time):
            for tf_value in tb_parse_event.summary.value:
                self._print_verbose("calling mlops.set_stats {}".format(tf_value.tag))
                mlops.set_stat(tf_value.tag, data=tf_value.simple_value)

    def _report_events(self, events_list, time_stamp_start):
        for event in events_list:
            self._report_event(event, time_stamp_start)

    def run(self):
        """
        Process the Directory path provided scanning for the TB events.
        Continues to run until terminated, scan and reports only new
        events logged to the respective TB-files

        1st scan of directory: read and log all TB events to the DB
        using the MLOps-stats API

        Subsequent scan of directory: check for only the newly appended
        TB files and update DB using the MLOps-stats API.
        """

        if not self._log_dir:
            raise Exception("log_dir was not given to TBParser: {}".format(self._log_dir))

        self._print("calling mlops_init()")
        mlops.init()

        flist = []
        files_found = []

        while True:
            try:
                if gfile.IsDirectory(self._log_dir):
                    files_found = gfile.ListDirectory(self._log_dir)
                    self._print_verbose("found log dir [{}]".format(self._log_dir))
                    self._print_verbose("Found files: {}".format(files_found))
                else:
                    self._print_verbose("could not find log dir [{}] will sleep".format(self._log_dir))
                    time.sleep(self._sleep_time)
                    continue
            except Exception as e:
                self._print("Error: READ Directory attempt failed: {}".format(e))
                break

            # Get the files in directory and respective filesize
            # And continue rescan of the directory for changes
            for file in files_found:
                file_path = os.path.join(self._log_dir, file)


                # TODO: move this to a separate routine - adding a new file
                # Confirm file has been seen before, if not add to list
                if file not in flist:
                    # add file to the known file list
                    flist.append(file)
                    time_stamp_start = 0

                    try:
                        self._file_size[file] = gfile.Stat(file_path).length
                        self._file_time[file] = time.time()

                        is_tf_events_file = event_accumulator.IsTensorFlowEventsFile(file_path)
                        if self._file_size[file] > 0 and is_tf_events_file:
                            event_list = itertools.chain(*[generate_event_from_file(file_path)])
                            self._report_events(event_list, time_stamp_start)

                    except Exception as e:
                        self._print("exception : {0}".format(e))
                        time.sleep(self._sleep_time)

                # stat files to compare length
                if self._file_size[file] < gfile.Stat(file_path).length and \
                        event_accumulator.IsTensorFlowEventsFile(file_path):

                    self._file_size[file] = gfile.Stat(file_path).length

                    try:
                        time_stamp_start = self._file_time[file]
                        self._file_time[file] = time.time()
                        event_list = itertools.chain(*[generate_event_from_file(file_path)])
                        self._report_events(event_list, time_stamp_start)

                    except Exception as e:
                        self._print("exception: {0}".format(e))

            time.sleep(self._sleep_time)
            continue

        mlops.done()


def prog_args():
    parg = argparse.ArgumentParser(description='PM Reflex TensorBoard component event stats')

    parg.add_argument('--logdir', required=False, default='./log',
                      help='directory location of the Log file')
    parg.add_argument('--time_start', type=int, required=False, default=0,
                      help='Start time stamp of events')
    parg.add_argument('--sleep-time', type=int, required=False, default=1,
                      help="Amount of time in seconds to sleep after each cycle")
    parg.add_argument('--verbose', action="store_true", required=False, default=False,
                       help="Emmit extra information while running")

    return parg.parse_args()


if __name__ == '__main__':
    print("TB_parser main is starting", file=sys.stderr)
    args = prog_args()

    tb_parser = TBParser().verbose(args.verbose)\
        .log_dir(args.logdir)\
        .time_stamp_start(args.time_start)\
        .sleep_time(args.sleep_time)

    tb_parser.run()

    # This line should not be reached - the printout is for validation
    print("TB_parser done", file=sys.stderr)
