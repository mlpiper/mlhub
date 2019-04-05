package ai.h2o.mojo.parallelm.components;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import ai.h2o.mojo.parallelm.common.CSVPredictionWriter;
import ai.h2o.mojo.parallelm.common.PredictionWriter;
import ai.h2o.mojo.parallelm.common.PrettyPrintingMap;
import ai.h2o.mojos.runtime.MojoPipeline;
import ai.h2o.mojos.runtime.frame.MojoFrame;
import ai.h2o.mojos.runtime.frame.MojoFrameBuilder;
import ai.h2o.mojos.runtime.frame.MojoFrameMeta;
import ai.h2o.mojos.runtime.frame.MojoRowBuilder;
import com.parallelm.mlcomp.MCenterComponent;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;

import static java.lang.System.out;


public class H2ODriverlessAiPredictor extends MCenterComponent {

    private Path modelFilePath;
    private Path inputSamplesFilePath;
    private Path outputPredictionsFilePath;
    private Path licenseFilePath;
    private MojoPipeline mojoPipeline;
    private MojoFrameMeta mojoFrameMeta;

    private static Logger logger = Logger.getLogger(H2ODriverlessAiPredictor.class);

    private final String tmpDir = "/tmp";

    public static void main(String[] args) throws Exception {
        ConsoleAppender console = new ConsoleAppender();
        // Configure console appender
        String PATTERN = "%d [%p|%c|%C{1}] %m%n";
        console.setLayout(new PatternLayout(PATTERN));
        console.setThreshold(Level.DEBUG);
        console.activateOptions();
        // Add appender to any Logger (here is root)
        Logger.getRootLogger().addAppender(console);
        Logger.getRootLogger().setLevel(Level.INFO);

        H2ODriverlessAiPredictor middleComponent = new H2ODriverlessAiPredictor();
        ArgumentParser parser = ArgumentParsers.newFor("Checksum").build()
                .defaultHelp(true)
                .description("Calculate the checksum of given files");

        parser.addArgument("--input-model")
                .help("Path to input model to consume");
        parser.addArgument("--samples-file")
                .help("Path to samples to predict");
        parser.addArgument("--output-file")
                .help("Path to record predictions made");
        parser.addArgument("--license-file")
                .help("Path to license file (license.sig) for Driverless AI");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        System.out.println(options);
        out.println("Options: " + options);

        middleComponent.configure(options.getAttrs());
        List<Object> parentObjs = new ArrayList<Object>();
        parentObjs.add(options.get("samples_file"));
        List<Object> outputs = middleComponent.materialize(parentObjs);
        for (Object obj : outputs) {
            System.out.println("Output: " + obj.toString());
        }
    }

    private void checkArgs(List<Object> parentDataObjects) throws Exception {

        String inputSamplesFileStr;
        // From params
        String modelPathStr = (String) params.get("input_model");
        String licensePathStr = (String) params.get("license_file");
        System.out.println("param - input_model:  " + modelPathStr);
        String outputPredictionsFileStr = (String) params.getOrDefault("output_file", null);
        System.out.println("param - output_file:  " + outputPredictionsFileStr);

        if (parentDataObjects.size() != 0) {
            // From component input
            inputSamplesFileStr = (String) parentDataObjects.get(0);
            System.out.println("Connected component input parentDataObjects - 0: " + inputSamplesFileStr);
        } else {
            // get sample-file path from param
            inputSamplesFileStr = (String) params.getOrDefault("samples_file", null);
            System.out.println("param - samples_file:  " + inputSamplesFileStr);
        }

        inputSamplesFilePath = Paths.get(inputSamplesFileStr);
        if (!inputSamplesFilePath.toFile().exists()) {
            throw new Exception(String.format("Input samples file [%s] does not exist", inputSamplesFilePath));
        }
        modelFilePath = Paths.get(modelPathStr);
        if (!modelFilePath.toFile().exists()) {
            throw new Exception(String.format("Model file [%s] does not exist", modelFilePath));
        }
        licenseFilePath = Paths.get(licensePathStr);
        if (!licenseFilePath.toFile().exists()) {
            throw new Exception(String.format("License file [%s] does not exist", licenseFilePath));
        }
        String outputFile = "h2o_driverlessai_predictions_" + UUID.randomUUID().toString() + ".out";
        if (outputPredictionsFileStr != null) {
            outputPredictionsFilePath = Paths.get(outputPredictionsFileStr);
            if (outputPredictionsFilePath.toFile().exists()) {
                if (outputPredictionsFilePath.toFile().isDirectory()) {
                    outputPredictionsFilePath = Paths.get(outputPredictionsFileStr, outputFile);
                }
            }
        } else {
            outputPredictionsFilePath = Paths.get(tmpDir, outputFile);
            System.out.println(String.format("No output file/dir was given - using [%s]", outputPredictionsFilePath));
        }

        String desc = "";
        desc += "Model:            %s\n";
        desc += "Samples file:     %s\n";
        desc += "Predictions file  %s\n";

        System.out.println(String.format(desc, modelFilePath, inputSamplesFileStr, outputPredictionsFilePath));
    }

    private void loadModel() throws Exception {
        System.setProperty("ai.h2o.mojos.runtime.license.file", licenseFilePath.toString());

        mojoPipeline = MojoPipeline.loadFrom(modelFilePath.toString());
        mojoFrameMeta = mojoPipeline.getInputMeta();
    }

    private boolean checkForHeader(String[] colNames, String[] firstRow) {
        for(int i = 0; i < colNames.length; i++) {
            logger.info(String.format("String 1: [%s]", colNames[i]));
            logger.info(String.format("String 2: [%s]", firstRow[i]));
            if (!colNames[i].trim().equals(firstRow[i].trim())) {
                return true;
            }
        }
        logger.info("First Row Doesn't Look Like A Header");
        return false;
    }

    private MojoRowBuilder constructInputMojoRow(MojoRowBuilder mojoRowBuilder, String nextRow, int sampleIndex) {
        String[] rowArray = nextRow.split(",");
        if (rowArray.length != mojoFrameMeta.size()) {
            logger.info(String.format("Row %d: does not have the same number of features as expected [%d]",
                    sampleIndex, mojoFrameMeta.size()));
            return null;
        }
        for(int i = 0; i < mojoFrameMeta.size(); i++) {
            mojoRowBuilder.setValue(mojoFrameMeta.getColumnName(i), rowArray[i]);
        }
        return mojoRowBuilder;
    }

    private ArrayList<Object> convertMojoPredictionsToCSVRow(MojoFrame outputFrame, int sampleIndex) {
        ArrayList<Object> resultRecord = new ArrayList<>();
        resultRecord.add(sampleIndex);
        for (int i = 0; i < outputFrame.getNcols(); i++) {
            String[] values = outputFrame.getColumn(i).getDataAsStrings();
            resultRecord.add(values[0]);
        }
        return resultRecord;
    }

    private void predict() throws Exception {
        String nextLine;
        int sampleIndex = 0;
        PredictionWriter predictionWriter = new CSVPredictionWriter(outputPredictionsFilePath);
        ArrayList<String> predictionHeader = new ArrayList<>();
        predictionHeader.add("index");
        predictionHeader.addAll(Arrays.asList(mojoPipeline.getOutputMeta().getColumnNames()));
        predictionWriter.writeHeader(predictionHeader);
        try (BufferedReader br = new BufferedReader(new FileReader(inputSamplesFilePath.toFile()))) {
            while ((nextLine = br.readLine()) != null) {
                MojoFrameBuilder mojoFrameBuilder = mojoPipeline.getInputFrameBuilder();
                MojoRowBuilder mojoRowBuilder = mojoFrameBuilder.getMojoRowBuilder();
                if (sampleIndex == 0) {
                    if (!checkForHeader(mojoFrameMeta.getColumnNames(), nextLine.split(","))) {
                        logger.info("Skipping First Row in CSV File:");
                        logger.info(String.format("[%s] matches [%s] from Mojo Metadata",
                                Arrays.toString(nextLine.split(",")),
                                Arrays.toString(mojoFrameMeta.getColumnNames())));
                    } else {
                        mojoRowBuilder = constructInputMojoRow(mojoRowBuilder, nextLine, sampleIndex);
                    }
                } else {
                    mojoRowBuilder = constructInputMojoRow(mojoRowBuilder, nextLine, sampleIndex);
                }
                if (mojoRowBuilder != null) {
                    mojoFrameBuilder.addRow(mojoRowBuilder);
                    MojoFrame inputFrame = mojoFrameBuilder.toMojoFrame();
                    MojoFrame outputFrame = mojoPipeline.transform(inputFrame);
                    ArrayList<Object> resultRecord = convertMojoPredictionsToCSVRow(outputFrame, sampleIndex);
                    predictionWriter.writePrediction(resultRecord);
                } else {
                    logger.info(String.format("Skipped Row %d", sampleIndex));
                }
                sampleIndex++;
            }
            if (mlops != null) {
                mlops.setStat("pipelinestat.count", sampleIndex);
            }
        } finally {
            predictionWriter.close();
        }
    }

    @Override
    public List<Object> materialize(List<Object> parentDataObjects) throws Exception {
        logger.info("H2O.ai Driverless AI Predictor - materialize");
        logger.info(new PrettyPrintingMap<>(params));
        checkArgs(parentDataObjects);
        loadModel();
        predict();
        List<Object> outputs = new ArrayList<>();
        outputs.add(outputPredictionsFilePath.toString());
        return outputs;
    }
}