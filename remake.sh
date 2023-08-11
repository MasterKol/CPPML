#!BASH

cd src

SOURCES=`echo ./*.cpp ./*/*.cpp`
OBJECTS=`for SOURCE in $SOURCES; do echo -n $\{BP}"/$(basename ${SOURCE%.*}.o) "; done`
CFLAGS="-std=c++17 -Xclang -fopenmp -I/usr/local/opt/libomp/include -I../include"
BP="../bin"
CC="clang++"

#`echo *.cpp */*.cpp`

INCLUDE="../include"

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
	HEADERS=`awk -F \" -v bp=$BASEPATH '/\#include \"/ {print "${INC}/"bp"/" $2}' $SOURCE ${INCLUDE}/${SOURCE%.*}.hpp`
	echo \${BP}/$(basename ${SOURCE%.*}).o : $HEADERS >> ${OUTFILE}
	echo -e '\t${CC} ${CFLAGS} -c $< -o $@\n' >> ${OUTFILE}
done

