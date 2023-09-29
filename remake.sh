#!/bin/bash

cd src

include="../include"
# settings
cflags="-std=c++17 -O2 -Xclang -fopenmp -I/usr/local/opt/libomp/include -I${include} -I${include}/Layers -I${include}/Optimizers -Wall -g"
bp="../bin"
cc="clang++"

search_paths=". ${include} ${include}/Layers ${include}/Optimizers ./Layers ./Optimizers"
sources=`echo ./*.cpp ./*/*.cpp`
objects=`for source in $sources; do echo -n $\{bp}"/$(basename ${source%.*}.o) "; done`

outfile=Makefile

function simplifyPath(){
	for file in "$@"; do
		# sed command first removes */.. , then replaces /./ with / , then removes leading ./ , then reduces //+ to /
		# if sed is not installed then file is returned unchanged
		echo "$file" | sed -E 's/[a-zA-Z0-9_]+\/\.{2}\///g; s/\/\.\//\//g; s/^\.?\///g; s/\/{2,}/\//g' 2> /dev/null || echo "$file"
	done
}

function findFile() {
	for file in "$@"; do
		if test -f $file; then
			echo -n "$file "
			continue
		fi

		for spath in $search_paths; do
			if test -f "$spath/$file"; then
				echo -n "$spath/$file "
			fi
		done
	done
	exit 1
}

function getHeaders() {
	file=$(findFile "$1")
	basepath=${file%/*}
	headers=$(awk -F \" -v bp=$basepath '/\#include \"/ {print $2}' $file)
	simplifyPath $(findFile $headers)
}

touch ${outfile}

# write header info to start of outfile
cat > $outfile << EOM
cc=${cc}
bp=${bp}
cflags=${cflags}
inc=${include}
.PHONY: all
all: \${bp} ${objects}

\${bp}:
	mkdir \${bp}
	tmkdir \${bp}/Layers
	mkdir \${bp}/Optimizers

EOM

for source in $sources
do
	basepath=${source%/*}
	# get all internal headers from cpp and hpp file
	headers="$(getHeaders $source) $(getHeaders ${source%.*}.hpp)"
	# write target and dependencies line to makefile
	echo \${bp}/$(basename ${source%.*}).o : $source $headers >> ${outfile}
	# write standard general compile line
	echo -e '\t${cc} ${cflags} -c $< -o $@\n' >> ${outfile}
done