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
import tensorflow as tf

from model import Model


class SavedModel(Model):
    def __init__(self, model_dir, signature_def):
        super(SavedModel, self).__init__(model_dir, signature_def)

        # loads the metagraphdef(s) into the provided session
        # restores variables, gets assets, initializes the assets into the main function
        self._sess = tf.Session()

        # For now, we only set the default tf_serving tag_set
        tag_set = "serve"
        tf.saved_model.loader.load(self._sess, [tag_set], self._model_dir)
        graph = tf.get_default_graph()

        self._input_node = graph.get_tensor_by_name(self.get_input_name())
        self._model = graph.get_tensor_by_name(self.get_output_name())

    def infer(self, sample):
        inferences = self._sess.run(self._model, {self._input_node: [sample]})

        # For now, we only process one inference at a time
        return inferences[0]

    def get_num_categories(self):
        return self.get_output_shape()

    def __del__(self):
        self._sess.close()
        super(SavedModel, self).__del__()
