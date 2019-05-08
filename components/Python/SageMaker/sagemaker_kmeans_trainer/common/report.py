from time import gmtime, strftime

from parallelm.mlops import mlops
from parallelm.mlops.stats.table import Table


class Report(object):

    @staticmethod
    def job_status(job_name, status):
        tbl = Table().name("SageMaker Job Status").cols(["Job Name", "Update Time", "Status"])
        tbl.add_row([job_name, strftime("%Y/%m/%d %H:%M:%S", gmtime()), status])
        mlops.set_stat(tbl)

    @staticmethod
    def job_metric(metric_name, value):
        mlops.set_stat(metric_name, value)
