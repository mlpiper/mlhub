#!/usr/bin/env Rscript

# Required for reticulate to work properly
Sys.setenv(MKL_THREADING_LAYER = "GNU")

# Import packages
library("reticulate")
library("optparse")
py_discover_config()
py_config()

# Import mlops APIs
mlops <- import("parallelm.mlops", convert = TRUE)
mlops <- mlops$mlops

bar.graph <- import("parallelm.mlops.stats.bar_graph", convert = TRUE)
BarGraph <- bar.graph$BarGraph

predefined.stats <- import('parallelm.mlops.predefined_stats', convert = TRUE)
PredefinedStats <- predefined.stats$PredefinedStats

# Parse the arguments to the component
option.list = list(make_option(c("--input-model"), type = "character", default = NULL, help = "input model to use for predictions", metavar = "character"))
opt.parser = OptionParser(option_list = option.list)
opt = parse_args(opt.parser, convert_hyphens_to_underscores = TRUE)

# Iniitialize mlops API and check if the required arguments are provided.
mlops$init()
if (is.null(opt$input_model)) {
  print_help(opt_parser)
  stop("At least one argument must be supplied (input model).n", call. = FALSE)
}

# Load the model and infer if available
if (file.exists(opt$input_model)) {

  # Load model
  model <- readRDS(opt$input_model)
  num.samples = sample(50:100, 1)
  
  # Generate synthetic data for linear regression algorithm
  inference.data <- data.frame(x1 = rnorm(num.samples), x2 = rnorm(num.samples), x3 = rnorm(num.samples))

  ########## Start of ParallelM instrumentation ############
  mlops$set_stat(PredefinedStats$PREDICTIONS_COUNT, num.samples)
  mlops$set_data_distribution_stat(data = inference.data)
  ########## End of ParallelM instrumentation ##############

  # Make predictions
  y <- predict.lm(model, inference.data)
  prediction.histogram <- hist(y)
  
  ########## Start of ParallelM instrumentation ############
  # Prediction distributiion
  mlt_cont <- BarGraph()$name("Prediction Distribution")$cols(prediction.histogram$breaks)$data(prediction.histogram$counts)$as_continuous()
  mlops$set_stat(mlt_cont)
  ########## End of ParallelM instrumentation ##############
  
} else {
  print("file not found: ")
}

## MLOps done to stop the library
mlops$done()

