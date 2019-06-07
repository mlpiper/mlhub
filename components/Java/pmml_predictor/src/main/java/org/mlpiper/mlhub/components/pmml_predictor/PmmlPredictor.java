package org.mlpiper.mlhub.components.pmml_predictor;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;

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

import org.apache.log4j.PatternLayout;

import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.evaluator.*;
import org.jpmml.model.JAXBUtil;
import org.jpmml.model.ImportFilter;
import org.xml.sax.InputSource;

import javax.xml.transform.sax.SAXSource;

class CSVSampleReader {
    CSVParser csvParser;
    Iterator<CSVRecord> csvRecordIterator;
    Map<String, Integer> headerMap;
    Evaluator jpmmlEvaluator = null;

    public CSVSampleReader(Path csvSampleFilePath, Evaluator evaluator) throws Exception {

        // TODO: add header as an option
        new BufferedReader(new FileReader(csvSampleFilePath.toString()));
        Reader reader = Files.newBufferedReader(Paths.get(csvSampleFilePath.toString()));
        csvParser = new CSVParser(reader, CSVFormat.DEFAULT
                .withFirstRecordAsHeader()
                .withIgnoreHeaderCase()
                .withTrim());
        csvRecordIterator = csvParser.iterator();
        headerMap = csvParser.getHeaderMap();
        jpmmlEvaluator = evaluator;
    }

    public Map<String, Integer> getHeader() {
        return headerMap;
    }

    public Map<FieldName, FieldValue> nextSample() throws Exception {
        Map<FieldName, FieldValue> row = new HashMap<FieldName, FieldValue>();

        if (!csvRecordIterator.hasNext()) {
            return null;
        }

        CSVRecord csvRecord = csvRecordIterator.next();

        List<InputField> inputField = jpmmlEvaluator.getInputFields();

        for (InputField field : inputField) {
            Integer idx = headerMap.get(field.getName().getValue());
            FieldValue fv = field.prepare(csvRecord.get(idx));
            row.put(field.getName(), fv);
        }
        return row;
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
        csvPrinter.printRecord(record);
    }

    @Override
    public void close() throws Exception {
        csvPrinter.close();
    }
}

public class PmmlPredictor extends MCenterComponent
{
    private Path modelFilePath;
    private Path inputSamplesFilePath;
    private Path outputPredictionsFilePath;
    private static Logger logger = Logger.getLogger(PmmlPredictor.class);

    private Evaluator evaluator;

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
            throw new Exception(String.format("Model file [%s] already exists", modelFilePath));
        }

        String outputFile = "pmml_predictions_" + UUID.randomUUID().toString() + ".out";
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

    public void loadModel(List<Object> parentDataObjects) throws Exception {
        checkArgs(parentDataObjects);
        ModelEvaluatorFactory evaluatorInstance = ModelEvaluatorFactory.newInstance();
        List<String> modelList = Files.readAllLines(modelFilePath);
        String modelString = String.join("\n", modelList);
        SAXSource src =
                JAXBUtil.createFilteredSource(
                        new InputSource(new StringReader(modelString)), new ImportFilter());
        evaluator = evaluatorInstance.newModelEvaluator(JAXBUtil.unmarshalPMML(src));
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
        logger.info("PmmlPredictor - materialize");
        checkArgs(parentDataObjects);
        loadModel(parentDataObjects);

        CSVSampleReader sampleReader = new CSVSampleReader(inputSamplesFilePath, evaluator);
        PredictionWriter predictionWriter = new CSVPredictionWriter(outputPredictionsFilePath);
        predict(sampleReader, predictionWriter, true);

        predictionWriter.close();
        List<Object> outputs = new ArrayList<>();
        outputs.add(outputPredictionsFilePath.toString());
        return outputs;
    }

    private void predict(CSVSampleReader sampleReader, PredictionWriter predictionWriter,
                         boolean writeHeader) throws Exception {

        Map<FieldName, FieldValue> sample;
        int sampleIndex = 0;
        logger.info("PMML predictor - Starting predict loop");

        while((sample = sampleReader.nextSample()) != null) {
            Map<FieldName, ?> output = evaluator.evaluate(sample);

            ArrayList<Object> resultRecord = new ArrayList<>();
            ArrayList<Object> headerRecord = new ArrayList<>();
            headerRecord.add("index");
            resultRecord.add(sampleIndex);

            for(FieldName fn: sample.keySet()) {
                headerRecord.add(fn.getValue());
                resultRecord.add(sample.get(fn).asString());
            }

            for(FieldName key: output.keySet()) {
                headerRecord.add(key);
                resultRecord.add(output.get(key));
                if (sampleIndex == 0) {
                    logger.info("name: " + key.toString() + " value: " + output.get(key));
                }
            }

            if (sampleIndex == 0 && writeHeader) {
                predictionWriter.writePrediction(headerRecord);
            }

            predictionWriter.writePrediction(resultRecord);
            sampleIndex++;
        }
        if (mlops != null) {
            mlops.setStat("pipelinestat.count", sampleIndex);
        }
    }

    public static void main(String[] args ) throws Exception {
        ConsoleAppender console = new ConsoleAppender(); //create appender
        //configure the appender
        String PATTERN = "%d [%p|%c|%C{1}] %m%n";
        console.setLayout(new PatternLayout(PATTERN));
        console.setThreshold(Level.DEBUG);
        console.activateOptions();
        //add appender to any Logger (here is root)
        Logger.getRootLogger().addAppender(console);
        Logger.getRootLogger().setLevel(Level.INFO);

        PmmlPredictor middleComponent = new PmmlPredictor();

        ArgumentParser parser = ArgumentParsers.newFor("Checksum").build()
                                               .defaultHelp(true)
                                               .description("Calculate checksum of given files.");

        parser.addArgument("--input-model")
              .help("Path to input model to consume");

        parser.addArgument("--samples-file")
                .help("Path to samples to predict");

        parser.addArgument("--output-file")
                .help("Path to record the predictions made");

        parser.addArgument("--convert-unknown-categorical-levels-to-na")
              .type(Boolean.class)
              .setDefault(false)
              .help("Set the convert_unknown_categorical_levels_to_na property of the Mojo model predictor");

        parser.addArgument("--convert-invalid-numbers-to-na")
              .type(Boolean.class)
              .setDefault(false)
              .help("Set the convert-invalid-numbers-to-na property of the Mojo model predictor");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }

        middleComponent.configure(options.getAttrs());
        List<Object> parentObjs = new ArrayList<Object>();
        parentObjs.add(options.get("samples_file"));
        List<Object> outputs = middleComponent.materialize(parentObjs);
        for (Object obj: outputs) {
            System.out.println("Output: " + obj.toString());
        }
    }
}
