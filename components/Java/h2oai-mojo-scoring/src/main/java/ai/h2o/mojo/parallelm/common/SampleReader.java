package ai.h2o.mojo.parallelm.common;

import hex.genmodel.easy.RowData;

public abstract class SampleReader {
    abstract public RowData nextSample() throws Exception;
}
