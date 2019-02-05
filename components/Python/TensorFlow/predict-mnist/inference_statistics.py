from abc import ABCMeta, abstractmethod


class InferenceStatistics(object):
    def __init__(self, report_interval):
        self._report_interval = report_interval
        self._total = 0
        self._correct = 0
        self._low_conf = 0
        self._cum_total = 0

    @abstractmethod
    def infer_stats(self, sample, label, inference):
        pass

    def get_total(self):
        return self._total

    def get_cum_total(self):
    	return self._cum_total

    def get_correct(self):
        return self._correct

    def get_low_conf(self):
        return self._low_conf

    def get_report_interval(self):
        return self._report_interval

    def increment_total(self, num_to_add=1):
        self._total += num_to_add
        self._cum_total += num_to_add

    def increment_correct(self, num_correct=1):
        self._correct += num_correct

    def increment_low_conf(self, low_conf=1):
        self._low_conf += low_conf

    def reset(self):
        self._total = 0
        self._correct = 0
        self._low_conf = 0

    def is_time_to_report(self):
        return self._total % self._report_interval == 0

    @abstractmethod
    def report_stats(self):
        pass

    def __del__(self):
        pass
