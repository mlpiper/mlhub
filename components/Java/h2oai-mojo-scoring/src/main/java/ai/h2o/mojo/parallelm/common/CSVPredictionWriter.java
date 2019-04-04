package ai.h2o.mojo.parallelm.common;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.FileWriter;
import java.nio.file.Path;
import java.util.ArrayList;

public class CSVPredictionWriter extends PredictionWriter {
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