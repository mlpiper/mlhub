package ai.h2o.mojo.parallelm.common;

import java.util.ArrayList;

public abstract class PredictionWriter {
    abstract public void writeHeader(ArrayList<String> header) throws Exception;
    abstract public void writePrediction(ArrayList<Object> record) throws Exception;
    abstract public void close() throws Exception;
}