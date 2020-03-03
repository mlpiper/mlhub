from __future__ import print_function

from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph


class MCenterStatsComponentAdapter(ConnectableComponent):
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        mlops.init()

        df = parent_data_objs[0]
        dataframe_is = self._params.get("dataframe_is", "input_data")

        if dataframe_is == "input_data":
            self._handle_input_data(df)
        elif dataframe_is == "categorical_predictions_probabilities":
            self._handle_categorical_predictions(df)
        elif dataframe_is == "other":
            pass
        else:
            self._logger("Error: argument value is not supported: {}".format(dataframe_is))

        mlops.done()

        return[df]

    def _handle_input_data(self, df):
        mlops.set_data_distribution_stat(df)

    def _handle_categorical_predictions(self, df):

        df_max_col = df.idxmax(axis=1)
        series_value_count = df_max_col.value_counts(normalize=True)

        col_values = []
        for col in df.columns:
            col_values.append(series_value_count.at[col])

        bg = BarGraph().name("Categorical Prediction Distribution").cols(list(df.columns)).data(col_values)
        mlops.set_stat(bg)
