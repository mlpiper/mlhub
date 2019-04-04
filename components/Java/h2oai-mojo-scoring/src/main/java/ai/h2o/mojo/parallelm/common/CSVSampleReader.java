package ai.h2o.mojo.parallelm.common;

import hex.genmodel.easy.RowData;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.Map;


public class CSVSampleReader extends SampleReader{

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
