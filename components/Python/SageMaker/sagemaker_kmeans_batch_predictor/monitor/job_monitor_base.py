import abc
import logging
import pprint
import time

from monitor.report import Report
from monitor.sm_api_constants import SMApiConstants
from parallelm.common.mlcomp_exception import MLCompException


class JobMonitorBase(object):
    MONITOR_INTERVAL_SEC = 10.0

    def __init__(self, sagemaker_client, job_name, logger):
        self._logger = logger
        self._sagemaker_client = sagemaker_client
        self._job_name = job_name
        self._on_complete_callback = None

    def monitor(self):
        self._logger.info("Monitoring job ... {}".format(self._job_name))
        start_running_time_sec = time.time() - 1
        while True:
            response = self._describe_job()
            running_time_sec = int(time.time() - start_running_time_sec)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(pprint.pformat(response, indent=4))

            status = self._job_status(response)
            Report.job_status(self._job_name, running_time_sec, status)
            if status == SMApiConstants.JOB_COMPLETED:
                self._logger.info("Job '{}' completed!".format(self._job_name))
                self._report_final_metrics(response)
                if self._on_complete_callback:
                    self._on_complete_callback(response)
                break
            elif status == SMApiConstants.JOB_FAILED:
                msg = "Job '{}' failed! message: {}".format(self._job_name, response[SMApiConstants.FAILURE_REASON])
                self._logger.error(msg)
                raise MLCompException(msg)
            elif status != SMApiConstants.JOB_IN_PROGRESS:
                self._logger.warning("Unexpected job status! job-name: {}, status: {}".format(self._job_name, status))

            self._report_online_metrics(response)
            self._logger.info("Job '{}' is still running ... {} sec"
                              .format(self._job_name, running_time_sec))
            time.sleep(JobMonitorBase.MONITOR_INTERVAL_SEC)

    def set_on_complete_callback(self, on_complete_callback):
        # The prototype of the callback is 'callback(describe_response)'
        self._on_complete_callback = on_complete_callback
        return self

    @abc.abstractmethod
    def _describe_job(self):
        pass

    @abc.abstractmethod
    def _job_status(self, describe_response):
        pass

    @abc.abstractmethod
    def _report_online_metrics(self, describe_response):
        pass

    @abc.abstractmethod
    def _report_final_metrics(self, describe_response):
        pass
