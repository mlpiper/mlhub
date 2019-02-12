# Copyright 2018 ParallelM, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from abc import ABCMeta, abstractmethod
from random import randint


class StreamInput(object):
    def __init__(self, total_records, stop_at_record=-1, random=False):
        self._total_records = total_records
        self._stop_at_record = stop_at_record
        self._random = random
        self._records_returned = 0

    def get_next_input_index(self):
        if self._stop_at_record >= 0 and self._records_returned >= self._stop_at_record:
            return -1

        if self._random:
            next_index = randint(0, self._total_records)
        else:
            next_index = self._records_returned % self._total_records
        self._records_returned += 1
        return next_index

    @abstractmethod
    def get_next_input(self):
        pass

    def __del__(self):
        pass
