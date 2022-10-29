.PHONY: clean data requirements 

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME=$(shell basename $(PROJECT_DIR))

# Derive python version from the file *pyversion*
PYTHONVERSION=$(shell cat pyversion)

# Detect Conda install
ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create conda environment.
create_environment:
ifeq (True,$(HAS_CONDA))
	echo $(PROJECT_DIR)	
	echo $(PROJECT_NAME)
	conda create --yes --name $(PROJECT_NAME) python=$(PYTHONVERSION)
else
	@echo "Conda not available!"
	@echo "To install miniconda:"
ifeq (Darwin,$(shell uname -s))
	@echo " - wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh Miniconda3-latest.sh"
endif
ifeq (Linux,$(shell uname -s))
	@echo " - wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh Miniconda3-latest.sh"
endif
	@echo " - chmod 755 Miniconda3-latest.sh"
	@echo " - ./Miniconda3-latest.sh"
endif

## Make environment ready for use
requirements: 
	pip install -r requirements.txt
	make make_jupyter_kernel
	
make_jupyter_kernel: 
	python -m ipykernel install --user --name $(PROJECT_NAME) --display-name $(PROJECT_NAME)

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                                #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#   * save line in hold space
#   * purge line
#   * Loop:
#       * append newline + line to hold space
#       * go to next line
#       * if line starts with doc comment, strip comment character off and loop
#   * remove target prerequisites
#   * append hold space (+ newline) to line
#   * replace newline plus comments by `---`
#   * print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')

