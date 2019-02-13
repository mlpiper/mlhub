package com.parallelm.components.h2o_predictor;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import com.parallelm.mlcomp.MCenterComponent;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;
import hex.genmodel.MojoModel;

abstract class SampleReader {

    abstract public RowData nextSample() throws Exception;
}

class CSVSampleReader extends  SampleReader {
    CSVParser csvParser;
    Iterator<CSVRecord> csvRecordIterator;
    Map<String, Integer> headerMap;

    public CSVSampleReader(Path csvSampleFilePath) throws Exception {

        // TODO: add header as an option
        new BufferedReader(new FileReader(csvSampleFilePath.toString()));
        Reader reader = Files.newBufferedReader(Paths.get(csvSampleFilePath.toString()));
        csvParser = new CSVParser(reader, CSVFormat.DEFAULT
                .withFirstRecordAsHeader()
                .withIgnoreHeaderCase()
                .withTrim());
        csvRecordIterator = csvParser.iterator();
        headerMap = csvParser.getHeaderMap();
    }

    public Map<String, Integer> getHeader() {
        return headerMap;
    }

    @Override
    public RowData nextSample() throws Exception {

        RowData sample = new RowData();

        if (!csvRecordIterator.hasNext()) {
            return null;
        }

        CSVRecord csvRecord = csvRecordIterator.next();

        for (Map.Entry<String,Integer> entry : headerMap.entrySet()) {
            Integer idx = entry.getValue();
            sample.put(entry.getKey(), csvRecord.get(idx));
        }
        return sample;
    }
}

abstract class PredictionWriter {
    abstract public void writeHeader(ArrayList<String> header) throws Exception;
    abstract public void writePrediction(ArrayList<Object> record) throws Exception;
    abstract public void close() throws Exception;

}

class CSVPredictionWriter extends PredictionWriter {
    private Path predictionFilePath;
    private CSVPrinter csvPrinter;

    public CSVPredictionWriter(Path predictionFilePath) throws Exception {
        this.predictionFilePath = predictionFilePath;
        csvPrinter = new CSVPrinter(new FileWriter(predictionFilePath.toString()), CSVFormat.DEFAULT);
    }

    @Override
    public void writeHeader(ArrayList<String> header) throws Exception {
        csvPrinter.printRecord(header);
    }
    @Override
    public void writePrediction(ArrayList<Object> record) throws Exception {
        System.out.println("Record: " + record.toString());
        csvPrinter.printRecord(record);
    }

    @Override
    public void close() throws Exception {
        csvPrinter.close();
    }
}

public class H2OPredictor extends MCenterComponent
{
   private Path modelFilePath;
   private Path inputSamplesFilePath;
   private Path outputPredictionsFilePath;
   EasyPredictModelWrapper h2oModel;

   private final String tmpDir = "/tmp";

    private void checkArgs(List<Object> parentDataObjects) throws Exception {

        String inputSamplesFileStr;
        // From params
        String modelPathStr = (String) params.get("input_model");
        System.out.println("param - input_model:    " + modelPathStr);
        String outputPredictionsFileStr = (String) params.getOrDefault("output_file", null);
        System.out.println("param - output_file:    " + outputPredictionsFileStr);

        if (parentDataObjects.size() != 0) {
            // From component input
            inputSamplesFileStr = (String) parentDataObjects.get(0);
            System.out.println("Connected component input parentDataObjects - 0: " + inputSamplesFileStr);
        } else {
            // get sample-file path from param
            inputSamplesFileStr = (String) params.getOrDefault("samples_file", null);
            System.out.println("param - samples_file:            " + inputSamplesFileStr);
        }

        inputSamplesFilePath = Paths.get(inputSamplesFileStr);
        if (!inputSamplesFilePath.toFile().exists()) {
            throw new Exception(String.format("Input samples file [%s] does not exists", inputSamplesFilePath));
        }

        modelFilePath = Paths.get(modelPathStr);
        if (!modelFilePath.toFile().exists()) {
            throw new Exception(String.format("Model file [%s] does exits", modelFilePath));
        }

        String outputFile = "h2o_predictions_" + UUID.randomUUID().toString() + ".out";
        if (outputPredictionsFileStr != null) {
            outputPredictionsFilePath = Paths.get(outputPredictionsFileStr);
            if (outputPredictionsFilePath.toFile().exists()) {
                if (outputPredictionsFilePath.toFile().isDirectory()) {
                    outputPredictionsFilePath = Paths.get(outputPredictionsFileStr.toString(), outputFile);
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
        h2oModel = new EasyPredictModelWrapper(MojoModel.load(modelFilePath.toString()));
    }

    private static ArrayList<String> fixHeader(Map<String, Integer> header) {
        ArrayList<String> header2 = new ArrayList<String>();
        header2.ensureCapacity(header.size());
        for (int idx=0 ; idx < header.size() ; idx++) {
            header2.add("aa");
        }
        for (Map.Entry<String, Integer> entry: header.entrySet()) {
            header2.set(entry.getValue(), entry.getKey());
        }
        return header2;
    }

    @Override
    public List<Object> materialize(List<Object> parentDataObjects) throws Exception {
        System.out.println("H2OPredictor - materialize");

        checkArgs(parentDataObjects);
        loadModel();

        CSVSampleReader sampleReader = new CSVSampleReader(inputSamplesFilePath);
        PredictionWriter predictionWriter = new CSVPredictionWriter(outputPredictionsFilePath);
        ArrayList<String> predictionHeader = new ArrayList<>();
        predictionHeader.add("index");
        predictionHeader.add("label");
        predictionWriter.writeHeader(predictionHeader);
        predict(sampleReader, predictionWriter);

        predictionWriter.close();
        List<Object> outputs = new ArrayList<>();
        outputs.add(outputPredictionsFilePath.toString());
        return outputs;
    }

    private void predict(SampleReader sampleReader, PredictionWriter predictionWriter) throws Exception {

        RowData sample;
        int sampleIndex = 0;
        while((sample = sampleReader.nextSample()) != null) {
            BinomialModelPrediction prediction = h2oModel.predictBinomial(sample);
            System.out.println(sample + " : " + prediction.label + " probabilities : "
                    + prediction.classProbabilities[0] + " " + prediction.classProbabilities[1]);
            ArrayList<Object> resultRecord = new ArrayList<>();
            resultRecord.add(sampleIndex);
            resultRecord.add(prediction.label);
            resultRecord.add("" + prediction.classProbabilities[0] + " " + prediction.classProbabilities[1]);
            predictionWriter.writePrediction(resultRecord);
            sampleIndex++;
        }
        if (mlops != null) {
            mlops.setStat("pipelinestat.count", sampleIndex);
        }
    }

    public static void main(String[] args ) throws Exception {
        H2OPredictor middleComponent = new H2OPredictor();
        ArgumentParser parser = ArgumentParsers.newFor("Checksum").build()
                                               .defaultHelp(true)
                                               .description("Calculate checksum of given files.");

        parser.addArgument("--input-model")
              .help("Path to input model to consume");

        parser.addArgument("--samples-file")
                .help("Path to samples to predict");

        parser.addArgument("--output-file")
                .help("Path to record the predictions made");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        System.out.println(options);

        middleComponent.configure(options.getAttrs());
        List<Object> parentObjs = new ArrayList<Object>();
        parentObjs.add(options.get("sample_file"));
        List<Object> outputs = middleComponent.materialize(parentObjs);
        for (Object obj: outputs) {
            System.out.println("Output: " + obj.toString());
        }
    }
}
