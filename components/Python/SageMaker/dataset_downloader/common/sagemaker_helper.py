import time


def monitor_job(self, describe_job_func, in_args, job_name, polling_time_sec=10.0):
    index = 1
    while True:
        response = self._sagemaker_client.describe_transform_job(TransformJobName=self._job_name)
        status = response['TransformJobStatus']
        Report.job_status(self._job_name, status)
        if status == 'Completed':
            self._logger.info("Transform job ended with status: {}".format(status))
            break
        if status == 'Failed':
            message = response['FailureReason']
            msg = 'Transform failed with the following error: {}'.format(message)
            self._logger.error(msg)
            raise MLCompException(msg)
        self._logger.info("Transform job is still in status: {} ... {} sec".format(status, index * polling_time_sec))
        index += 1
        time.sleep(polling_time_sec)
