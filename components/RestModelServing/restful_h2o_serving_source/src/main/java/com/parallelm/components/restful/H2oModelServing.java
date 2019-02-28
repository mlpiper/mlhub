package com.parallelm.components.restful;

import com.google.gson.Gson;
import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.MojoModel;

import com.parallelm.mlcomp.MCenterRestfulComponent;


public class H2oModelServing extends MCenterRestfulComponent {
    private EasyPredictModelWrapper model;
    private int wid = 0;
    private boolean isVerbose = false;

    public void setEnvAttributes(int wid, boolean isVerbose) {
        this.wid = wid;
        this.isVerbose = isVerbose;
    }

    public void loadModel(String modelPath) throws IOException {
        model = new EasyPredictModelWrapper(MojoModel.load(modelPath));
        if (isVerbose) {
            System.out.println(String.format("(java side) wid: %d, model loaded successfully, columns: %s",
                    wid, Arrays.toString(model.m.getNames())));
        }
    }

    public MCenterRestfulComponent.Result predict(String query_string, String body_data) throws PredictException {

        Gson gson = new Gson();
        Map<String, Integer> predictionVector = new HashMap<String, Integer>();
        String jsonData = body_data != null && !body_data.isEmpty() ? body_data : query_string;
        predictionVector = (Map<String, Integer>) gson.fromJson(jsonData, predictionVector.getClass());

        if (predictionVector.size() != model.m.getNames().length) {
            throw new PredictException(String.format("Invalid prediction vector lenght! expected: %d, received: %d",
                                              model.m.getNames().length, predictionVector.size()));
        }

        RowData row = new RowData();
        for (Map.Entry<String, Integer> entry : predictionVector.entrySet()) {
            row.put(entry.getKey(), entry.getValue());
        }

        BinomialModelPrediction p = model.predictBinomial(row);

        MCenterRestfulComponent.Result result = new MCenterRestfulComponent.Result();

        Map<String, Object> resultAttrs = new HashMap<String, Object>();
        resultAttrs.put("penetrated", p.label);
        resultAttrs.put("probabilities", p.classProbabilities);

        result.returned_code = 200;
        result.json = gson.toJson(resultAttrs);

        return result;
    }

    public static void main(String[] args) throws Exception {

        H2oModelServing modelServing = new H2oModelServing();

        modelServing.loadModel("GBM_model_python_1542317998658_5.zip");

        String json = "{\"AGE\": 68, \"RACE\": 2, \"DCAPS\": 2, \"VOL\": 0, \"GLEASON\": 6}";

        MCenterRestfulComponent.Result result = modelServing.predict(null, json);

        System.out.println("Retrurned code: " + result.returned_code);
        System.out.println("Retrurned json: " + result.json);

        System.out.println("");
    }
}
