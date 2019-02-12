package com.parallelm.components.h2o_predictor;


import java.io.*;
import java.net.URL;
import java.nio.file.Paths;
import java.text.ParseException;
import static java.lang.System.out;


import org.junit.*;

/**
 * Unit tests for H2o (Binomial) Predictor
 */

public class H2OPredictorTest {

    H2OPredictor PredComp = new H2OPredictor();

    @Test
    public void testFailH2OModelImport() throws Exception{
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
    public void testFailH2OInputSample() throws Exception{
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
                Assert.fail("Fail");
            }
        }
    }

    @Test
    public void testH2OmodelImport() throws Exception{
        out.println("Test 1: testH2OmodelImport in progress.. ");
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
            Assert.fail("Failure here");
        }
    }

}


