

class SMApiConstants:

    # Constants from sagemaker_client.describe_training_job, sagemaker.describe_transform_job
    JOB_COMPLETED = 'Completed'
    JOB_IN_PROGRESS = 'InProgress'
    JOB_FAILED = 'Failed'

    FAILURE_REASON = 'FailureReason'

    class Estimator:
        JOB_STATUS = 'TrainingJobStatus'
        FINAL_METRIC_DATA_LIST = 'FinalMetricDataList'
        METRIC_NAME = 'MetricName'
        METRIC_VALUE = 'Value'
        ALGO_SPEC = 'AlgorithmSpecification'
        METRIC_DEFS = 'MetricDefinitions'
        METRIC_DEF_NAME = 'Name'
        DF_METRIC_NAME = 'metric_name'
        DF_METRIC_VALUE = 'value'
        TRAIN_PREFIX = 'train:'

    class Transformer:
        JOB_STATUS = 'TransformJobStatus'
        NAMESPACE = '/aws/sagemaker/TransformJobs'

        METRIC_CPU_UTILIZATION = 'CPUUtilization'
        METRIC_MEMORY_UTILIZATION = 'MemoryUtilization'

        STAT_AVG = 'Average'
        STAT_MIN = 'Minimum'
        STAT_MAX = 'Maximum'

        LIST_METRICS_NAME = 'Metrics'
        LIST_METRICS_DIM = 'Dimensions'
        LIST_METRICS_DIM_VALUE = 'Value'

        START_TIME = 'TransformStartTime'
        END_TIME = 'TransformEndTime'
        TIMESTAMP_ASC = 'TimestampAscending'

        METRICS_RESULTS = 'MetricDataResults'

        HOST_KEY = 'Host'


