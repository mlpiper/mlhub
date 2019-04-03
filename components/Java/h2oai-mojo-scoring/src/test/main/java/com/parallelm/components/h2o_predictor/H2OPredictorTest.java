package com.parallelm.components.h2o_predictor;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import deepwater.datasets.FileUtils;
import org.junit.*;
import org.junit.rules.TestName;

import static java.lang.System.out;

/**
 * Unit tests for H2o (Binomial) Predictor
 */

public class H2OPredictorTest {
    @Rule public TestName name = new TestName();

    @Test
    public void testCmdlineFailH2OModelImport() throws Exception{
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
            if (e.getCause() != null) {
                out.println(name.getMethodName() + " FAILED");
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlineFailH2OInputSample() throws Exception{
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
            if (e.getCause() != null) {
                out.println(name.getMethodName() + " FAILED");
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testCmdlineH2OModelImport() throws Exception{
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
        } catch (ParseException e) {
            out.println(name.getMethodName() + "FAILED");
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
            predComp.loadModel();
        } catch (Exception e) {
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        }
    }

    @Test
    public void testFailedH2OModelLoad() throws Exception{
        H2OPredictor predComp = new H2OPredictor();
        URL resource = H2OPredictorTest.class.getResource("/mini.csv");

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", "BAD_PATH");
        params.put("output_file", "/tmp/out-mini.csv");
        params.put("convert_unknown_categorical_levels_to_na", true);
        params.put("convert_invalid_numbers_to_na", true);

        predComp.configure(params);

        try {
            predComp.loadModel();
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
    public void testH2OPredict() throws Exception{
        H2OPredictor predComp = new H2OPredictor();
        List<Object> parentObjs = new ArrayList<Object>();

        URL resource = H2OPredictorTest.class.getResource("/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip");
        File model_file = Paths.get(resource.toURI()).toFile();

        resource = H2OPredictorTest.class.getResource("/mini.csv");
        File samples_file = Paths.get(resource.toURI()).toFile();

        resource = H2OPredictorTest.class.getResource("/diff-mini.csv");
        File diff_results = Paths.get(resource.toURI()).toFile();

        Map<String,Object> params = new HashMap<>();
        params.put("input_model", model_file.getAbsolutePath());
        params.put("output_file", "/tmp/out-mini.csv");
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
    public void testH2OPredictResults() throws Exception{
        String output_file = "/tmp/out-mini.csv";

        try {
            URL resource = H2OPredictorTest.class.getResource("/diff-mini.csv");

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
            out.println(name.getMethodName() + " FAILED");
            Assert.fail("Fail");
        }
    }
}
