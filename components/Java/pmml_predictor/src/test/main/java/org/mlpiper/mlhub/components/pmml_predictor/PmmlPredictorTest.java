package org.mlpiper.mlhub.components.pmml_predictor;

import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.*;
import org.junit.rules.TestName;

import static java.lang.System.out;

/**
 * Unit tests for Pmml Predictor
 */

public class PmmlPredictorTest {
    @Rule public TestName name = new TestName();
    @Test
    public void testCmdlineFailPMMLModelImport() throws Exception{
        URL resource = PmmlPredictorTest.class.getResource("/testSVM2.txt");
        File samples_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + null,
                "--samples-file=" + samples_file,
                "--output-file=/tmp/predictions.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            PmmlPredictor.main(arguments);
            Assert.assertEquals(1,2);
        } catch (Exception e) {
            if (e.getCause() != null) {
                out.println(name.getMethodName() + " FAILED");
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlineFailPMMLInputSample() throws Exception{
        URL resource = PmmlPredictorTest.class.getResource("/modelForRf");
        File model_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + model_file,
                "--samples-file=" + null,
                "--output-file=/tmp/predictions.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            PmmlPredictor.main(arguments);
            Assert.assertEquals(1,2);
        } catch (Exception e) {
            if (e.getCause() != null) {
                out.println(name.getMethodName() + " FAILED");
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlinePMMLModelImport() throws Exception{
        URL resource = PmmlPredictorTest.class.getResource("/modelforRf");
        File model_file = Paths.get(resource.toURI()).toFile();
        resource = PmmlPredictorTest.class.getResource("/testSVM2.txt");
        File samples_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + model_file,
                "--samples-file=" + samples_file,
                "--output-file=/tmp/predictions.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            PmmlPredictor.main(arguments);
        } catch (ParseException e) {
            out.println(name.getMethodName() + "FAILED");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testPMMLModelLoad() throws Exception{
        PmmlPredictor predComp = new PmmlPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = PmmlPredictorTest.class.getResource("/modelForRf");
        File model_file = Paths.get(resource.toURI()).toFile();

        out.println(model_file.getAbsolutePath());

        resource = PmmlPredictorTest.class.getResource("/testSVM2.txt");
        File samples_file = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", model_file.getAbsolutePath());
        params.put("output_file", "/tmp/predictions.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        parentObjs.add(samples_file.getAbsolutePath());
        predComp.configure(params);

        try {
            predComp.loadModel(parentObjs);
        } catch (Exception e) {
            out.println(e.getMessage());
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testFailedPMMLModelLoad() throws Exception{
        PmmlPredictor predComp = new PmmlPredictor();
        URL resource = PmmlPredictorTest.class.getResource("/testSVM2.txt");
        File samples_file = Paths.get(resource.toURI()).toFile();
        List<Object> parentObjs = new ArrayList<Object>();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", "BAD_PATH");
        params.put("output_file", "/tmp/predictions.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        predComp.configure(params);
        parentObjs.add(samples_file.getAbsolutePath());

        try {
            predComp.loadModel(parentObjs);
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        } catch (Exception e) {
            // expected exception cause : "null"
            if(e.getCause() != null) {
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testPMMLPredict() throws Exception{
        PmmlPredictor predComp = new PmmlPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = PmmlPredictorTest.class.getResource("/modelForRf");
        File model_file = Paths.get(resource.toURI()).toFile();

        resource = PmmlPredictorTest.class.getResource("/testSVM2.txt");
        File samples_file = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", model_file.getAbsolutePath());
        params.put("output_file", "/tmp/predictions.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        parentObjs.add(samples_file.getAbsolutePath());
        predComp.configure(params);

        try {
            parentObjs.add(samples_file.getAbsolutePath());
            predComp.materialize(parentObjs);
        } catch (Exception e) {
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testPMMLPredictResults() throws Exception{
        String output_file = "/tmp/predictions.csv";

        try {
            URL resource = PmmlPredictorTest.class.getResource("/predictions.csv");

            String diff_results = Paths.get(resource.toURI()).toString();
            BufferedReader s1 = new BufferedReader(new FileReader(diff_results));
            BufferedReader s2 = new BufferedReader(new FileReader(output_file));

            // For now we compare if both the file contents are identical
            // this check will need to get smarter, to allow for tolerance/thresholds

            String s1_line = null;
            String s2_line = null;

            for(;((((s1_line = s1.readLine()) != null)
                    && ((s2_line = s2.readLine()) != null))
                    && s1_line.equals(s2_line));) {
            }

            if (s1_line == null) {
                s2_line = s2.readLine();
                if (s1_line != s2_line) {
                    Assert.fail("File diff failed");
                }
            }
        } catch (Exception e) {
            out.println(e.getMessage());
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        }
    }
}
