#!/bin/bash

TESTS=*.tst

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

for TEST in $TESTS; do
	./$TEST >& /dev/null
	if [[ $? -eq 0 ]]; then 	# success
		echo -e "${GREEN}Passed ${TEST}${NC}"
	else 						# failure
		echo
		./$TEST
		echo -e "\n${RED}Failed Test ${TEST}${NC}"
		break
	fi
done