from __future__ import print_function

import sys
import os
import pandas

from parallelm.components import ConnectableComponent


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for read_file_to_df
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        if len(parent_data_objs) is not 0:
            file_path = str(parent_data_objs[0])
        else:
            file_path = self._params.get('filename')

        self._logger.info("file: {}".format(file_path))
        df = self.read_file_to_df(file_path)
        return [df]

    def read_file_to_df(self, filepath):
        """
        Read file and return DataFrame
        """

        if not os.path.exists(filepath):
            self._logger.info("stderr- failed to find {}".format(filepath), file=sys.stderr)
            raise Exception("file path does not exist: {}".format(filepath))

        df = pandas.read_csv(filepath)
        return df
