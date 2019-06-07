package org.mlpiper.mlhub.components.restful;

import com.google.gson.Gson;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.parallelm.mlcomp.MCenterRestfulComponent;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;


public class PmmlModelServing extends MCenterRestfulComponent {
    private int wid = 0;
    private boolean isVerbose = false;
    private Evaluator evaluator;

    public void setEnvAttributes(int wid, boolean isVerbose) {
        this.wid = wid;
        this.isVerbose = isVerbose;
    }

    public void loadModel(String modelPath) throws IOException {
        try {
            ModelEvaluatorFactory evaluatorInstance = ModelEvaluatorFactory.newInstance();
            List<String> modelList = Files.readAllLines(Paths.get(modelPath));
            String modelString = String.join("\n", modelList);
            SAXSource src =
                    JAXBUtil.createFilteredSource(
                            new InputSource(new StringReader(modelString)), new ImportFilter());
            evaluator = evaluatorInstance.newModelEvaluator(JAXBUtil.unmarshalPMML(src));
        } catch (Exception e) {
            throw new IOException("Invalid model at " + modelPath);
        }
        if (isVerbose) {
            System.out.println(String.format("(java side) wid: %d, model loaded successfully, " +
                            "model type %s", wid, evaluator.getMiningFunction().toString()));
        }
    }

    public MCenterRestfulComponent.Result predict(String query_string, String body_data) throws Exception {

        Gson gson = new Gson();
        Map<String, String> predictionVector = new HashMap<String, String>();
        String jsonData = body_data != null && !body_data.isEmpty() ? body_data : query_string;
        predictionVector = (Map<String, String>) gson.fromJson(jsonData, predictionVector.getClass());


        List<InputField> inputField = evaluator.getInputFields();
        Map<FieldName, FieldValue> row = new HashMap<FieldName, FieldValue>();

        for (InputField field : inputField) {
            try {
                FieldValue fv = field.prepare(predictionVector.get(field.getName().getValue()));
                row.put(field.getName(), fv);
            } catch (Exception e) {
                throw new InvalidObjectException("Json has missing or invalid value for " +
                        field.getName().getValue());
            }
        }

        try {
            Map<FieldName, ?> output = evaluator.evaluate(row);
            MCenterRestfulComponent.Result result = new MCenterRestfulComponent.Result();

            Map<String, Object> resultAttrs = new HashMap<String, Object>();
            for(FieldName key: output.keySet()) {
                resultAttrs.put(key.getValue(), output.get(key));
            }

            result.returned_code = 200;
            result.json = gson.toJson(resultAttrs);
            return result;
        } catch (Exception e) {
            throw new Exception("Evaluation failed with " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {

        PmmlModelServing modelServing = new PmmlModelServing();

        modelServing.loadModel("./model/modelForRf");

        Map<String, Double> data = new HashMap<>();
        data.put("c0", 1.0);
        data.put("c1", 1.0);
        data.put("c2", 1.0);
        data.put("c3", 1.0);
        data.put("c4", 1.0);
        data.put("c5", 1.0);
        data.put("c6", 1.0);
        data.put("c7", 1.0);
        data.put("c8", 1.0);
        data.put("c9", 1.0);
        data.put("c10", 1.0);
        data.put("c11", 1.0);
        data.put("c12", 1.0);
        data.put("c13", 1.0);
        data.put("c14", 1.0);
        data.put("c15", 1.0);
        data.put("c16", 1.0);
        data.put("c17", 1.0);
        data.put("c18", 1.0);
        data.put("c19", 1.0);
        data.put("c20", 1.0);
        data.put("c21", 1.0);
        data.put("c22", 1.0);
        data.put("c23", 1.0);
        data.put("c24", 1.0);

        Gson gson = new Gson();
        String json = gson.toJson(data);
        System.out.println(json);

        MCenterRestfulComponent.Result result = modelServing.predict(null, json);

        System.out.println("Retrurned code: " + result.returned_code);
        System.out.println("Retrurned json: " + result.json);

        System.out.println();
    }
}
