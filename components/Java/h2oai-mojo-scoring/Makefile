MKDIR_P := mkdir -p
COMP_DIR := h2o_predictor
CP_FILES := build/libs/h2oai-dai-parallelm-scorer.jar

#ai.h2o.mojo.parallelm.components.H2ODriverlessAiPredictor

.PHONEY: verify run_dai run_h2o3
verify:
	python -m json.tool < component.json > /dev/null
	./gradlew shadowJar
	${MKDIR_P} $(COMP_DIR)
	

$(COMP_DIR):
	${MKDIR_P} $(COMP_DIR)

run_h2o3:
	./H2O3_run.sh \
		./src/test/resources/Model_GBM1_32734b00-d570-4828-8a39-271a45c3f06a.zip \
	      	./src/test/resources/mini.csv

run_dai:
	./DAI_run.sh \
		./src/test/resources/default_payment_pipeline.mojo \
		./src/test/resources/test_default_payment.csv \
		/Users/dhananjoydas/PM/git/nickDIA/components/Java/h2oai-mojo-scoring/license.sig

clean:
	./gradlew clean build
	rm -rf $(COMP_DIR)
