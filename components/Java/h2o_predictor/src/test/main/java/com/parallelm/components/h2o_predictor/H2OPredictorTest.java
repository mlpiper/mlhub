package com.parallelm.components.h2o_predictor;

import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.System.out;

import net.sourceforge.argparse4j.inf.Namespace;
import org.junit.*;


/**
 * Unit tests for H2o (Binomial) Predictor
 */

public class H2OPredictorTest {

    @Test
    public void testCmdlineFailH2OModelImport() throws Exception{
        out.println("Test 1: testFailH2OModelImport in progress.. ");
        URL resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + null,
                "--samples-file=" + samples_file,
                "--output-file=/tmp/out-mini.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            H2OPredictor.main(arguments);
            Assert.assertEquals(1,2);
        } catch (Exception e) {
            if (e.getCause() == null) {
                out.println(String.format("Expected file (Model:%s) not found", e.getCause()));
            } else {
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlineFailH2OInputSample() throws Exception{
        out.println("Test 2: testFailH2OmodelImport in progress.. ");
        URL resource = H2OPredictorTest.class.getResource("/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip");
        File model_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + model_file,
                "--samples-file=" + null,
                "--output-file=/tmp/out-mini.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            H2OPredictor.main(arguments);
            Assert.assertEquals(1,2);
        } catch (Exception e) {
            if (e.getCause() == null) {
                out.println(String.format("Expected file (samples:%s) not found", e.getCause()));
            } else {
                out.println("Test 2: FAILED, testFailH2OmodelImport .. ");
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlineH2OModelImport() throws Exception{
        out.println("Test 3: testH2OmodelImport in progress.. ");
        URL resource = H2OPredictorTest.class.getResource("/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip");
        File model_file = Paths.get(resource.toURI()).toFile();
        resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();
        String[] arguments = new String[] {"--input-model=" + model_file,
                "--samples-file=" + samples_file,
                "--output-file=/tmp/out-mini.csv",
                "--convert-unknown-categorical-levels-to-na=True",
                "--convert-invalid-numbers-to-na=True"};
        try {
            H2OPredictor.main(arguments);
            out.println("Test 3: PASSED, testCmdlineH2OModelImport .. ");
        } catch (ParseException e) {
            out.println("Test 3: FAILED, testCmdlineH2OModelImport .. ");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testH2OModelLoad() throws Exception{
        H2OPredictor predComp = new H2OPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = H2OPredictorTest.class.getResource("/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip");
        File model_file = Paths.get(resource.toURI()).toFile();

        resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", model_file.getAbsolutePath());
        params.put("output_file", "/tmp/out-mini.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        parentObjs.add(samples_file.getAbsolutePath());
        predComp.configure(params);

        try {
            out.println("Test 4: testH2OModelLoad in progress.. ");
            predComp.loadModel();
            out.println("Test 4: PASSED testH2OModelLoad .. ");
        } catch (Exception e) {
            out.println("Test 4: FAILED, testH2OModelLoad .. ");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testFailedH2OModelLoad() throws Exception{
        H2OPredictor predComp = new H2OPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", "BAD_PATH");
        params.put("output_file", "/tmp/out-mini.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        parentObjs.add(samples_file.getAbsolutePath());
        predComp.configure(params);

        try {
            out.println("Test 5: testFailH2OmodelLoad in progress.. ");
            predComp.loadModel();
            out.println("Test 5: FAILED, testFailH2OModelLoad .. ");
            Assert.fail("Fail");
        } catch (Exception e) {
            out.println("Test 5: PASSED, testFailH2OModelLoad .. ");
        }
    }

    @Test
    public void testH2OPredict() throws Exception{
        H2OPredictor predComp = new H2OPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = H2OPredictorTest.class.getResource("/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip");
        File model_file = Paths.get(resource.toURI()).toFile();

        resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", model_file.getAbsolutePath());
        params.put("output_file", "/tmp/out-mini.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        parentObjs.add(samples_file.getAbsolutePath());
        predComp.configure(params);

        try {
            out.println("Test 6: testH2OPredict in progress.. ");
            parentObjs.add(samples_file.getAbsolutePath());
            predComp.materialize(parentObjs);
            out.println("Test 6: PASSED, testH2OPredict .. ");
        } catch (Exception e) {
            out.println("Test 6: FAILED, testH2OPredict .. ");
            Assert.fail("Fail");
        }
    }
}


