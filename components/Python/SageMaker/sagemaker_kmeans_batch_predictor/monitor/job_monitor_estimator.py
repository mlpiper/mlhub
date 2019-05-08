from datetime import datetime

from sagemaker import TrainingJobAnalytics

from monitor.job_monitor_base import JobMonitorBase
from monitor.report import Report
from monitor.sm_api_constants import SMApiConstants


class JobMonitorEstimator(JobMonitorBase):
    def __init__(self, sagemaker_client, job_name, logger):
        super(self.__class__, self).__init__(sagemaker_client, job_name, logger)

        self._metric_names = None
        self._analytics = TrainingJobAnalytics(training_job_name=self._job_name,
                                               metric_names=self._metric_names_for_training_job(),
                                               start_time=datetime.utcnow())

    def _describe_job(self):
        return self._sagemaker_client.describe_training_job(TrainingJobName=self._job_name)

    def _job_status(self, describe_response):
        return describe_response[SMApiConstants.Estimator.JOB_STATUS]

    def _report_online_metrics(self, describe_response):
        metrics_df = self._analytics.dataframe(force_refresh=True)
        if not metrics_df.empty:
            for index, row in metrics_df.iterrows():
                Report.job_metric(row.get(SMApiConstants.Estimator.DF_METRIC_NAME, "Unknown"),
                                  row.get(SMApiConstants.Estimator.DF_METRIC_VALUE, 0))
        else:
            for metric_name in self._metric_names_for_training_job():
                Report.job_metric(metric_name, 0)

    def _report_final_metrics(self, describe_response):
        for metric in describe_response[SMApiConstants.Estimator.FINAL_METRIC_DATA_LIST]:
            Report.job_metric(metric.get(SMApiConstants.Estimator.METRIC_NAME, "Unknown"),
                              metric.get(SMApiConstants.Estimator.METRIC_VALUE, 0))

    def _metric_names_for_training_job(self):
        if self._metric_names is None:
            training_description = self._sagemaker_client.describe_training_job(TrainingJobName=self._job_name)

            metric_definitions = training_description[SMApiConstants.Estimator.ALGO_SPEC][SMApiConstants.Estimator.METRIC_DEFS]
            self._metric_names = [md[SMApiConstants.Estimator.METRIC_DEF_NAME] for md in metric_definitions
                                  if md[SMApiConstants.Estimator.METRIC_DEF_NAME].startswith(SMApiConstants.Estimator.TRAIN_PREFIX)]

        return self._metric_names


