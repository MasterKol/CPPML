#!/bin/bash

cd src

# settings
CFLAGS="-std=c++17 -O2 -Xclang -fopenmp -I/usr/local/opt/libomp/include -I${INCLUDE} -I${INCLUDE}/Layers -I${INCLUDE}/Optimizers"
BP="../bin"
CC="clang++"

INCLUDE="../include"
SOURCES=`echo ./*.cpp ./*/*.cpp`
OBJECTS=`for SOURCE in $SOURCES; do echo -n $\{BP}"/$(basename ${SOURCE%.*}.o) "; done`

OUTFILE=Makefile

touch ${OUTFILE}

echo " " > ${OUTFILE}

echo CC=${CC} >> ${OUTFILE}
echo BP=${BP} >> ${OUTFILE}
echo CFLAGS=${CFLAGS} >> ${OUTFILE}
echo INC=${INCLUDE} >> ${OUTFILE}
echo ".PHONY: all" >> ${OUTFILE}
echo "all: \${BP} ${OBJECTS}" >> ${OUTFILE}
echo "" >> ${OUTFILE}
echo -e '${BP}:\n\tmkdir ${BP}\n\tmkdir ${BP}/Layers\n\tmkdir ${BP}/Optimizers\n' >>${OUTFILE}

for SOURCE in $SOURCES
do
	BASEPATH=${SOURCE%/*}
	# get all internal headers from cpp and hpp file
	HEADERS=`awk -F \" -v bp=$BASEPATH '/\#include \"/ {print "${INC}/"bp"/" $2}' $SOURCE ${INCLUDE}/${SOURCE%.*}.hpp`
	# write target and dependencies line to makefile
	echo \${BP}/$(basename ${SOURCE%.*}).o : $SOURCE $HEADERS >> ${OUTFILE}
	# write standard general compile line
	echo -e '\t${CC} ${CFLAGS} -c $< -o $@\n' >> ${OUTFILE}
done

