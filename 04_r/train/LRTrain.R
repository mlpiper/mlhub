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


# Parse the arguments to the component
option.list = list(make_option(c("--num-samples"), type="integer", default=50,
                                 help="Number of synthetic data samples to generate", metavar="integer"),
                   make_option(c("--output-model"), type="character", default=NULL,
                                help="output model to use for predictions", metavar="character"));

opt.parser = OptionParser(option_list=option.list);
opt = parse_args(opt.parser, convert_hyphens_to_underscores=TRUE);

# Iniitialize mlops API and check if the required arguments are provided.
mlops$init()
if (is.null(opt$output_model)){
    print_help(opt.parser)
    stop("At least one argument must be supplied (output model).n", call.=FALSE)
}

# Generate synthetic data for linear regression algorithm
training.data <- data.frame(x1 = rnorm(opt$num_samples), x2 = rnorm(opt$num_samples), x3 = rnorm(opt$num_samples))
training.data <- transform(training.data, y = 5 + (2.3 * x1) + (10 * x2) + (1.3 * x3) + rnorm(opt$num_samples, mean = runif(1,1,5)))

########## Start of ParallelM instrumentation ############
# Report the data distribution using mlops
mlops$set_data_distribution_stat(data=training.data)
########## End of ParallelM instrumentation ##############

prediction.histogram <- hist(training.data$y)

########## Start of ParallelM instrumentation ############
# Label distributiion
mlt = BarGraph()$name("Label Distribution")$cols(c(prediction.histogram$breaks))$data(c(prediction.histogram$counts))$as_continuous()
mlops$set_stat(mlt)
########## End of ParallelM instrumentation ##############


# Fit the linear model and calcualte rmse
model <- lm(y ~ x1 + x2 + x3, data = training.data)
rmse = sqrt(mean(residuals(model)^2))

########## Start of ParallelM instrumentation ############
# Report RMSE values as a time-series graph
mlops$set_stat('RMSE', rmse)
########## End of ParallelM instrumentation ##############


# Save this model
saveRDS(model, file = opt$output_model)

# MLOps done to stop the library
mlops$done()
