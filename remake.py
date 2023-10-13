#!/usr/bin/env python3

path_to_openmp = "/usr/local/opt/libomp/include"
cflags = "-std=c++17 -O2 -Wall -g"
cc = "g++"

import os
import subprocess
import sys
import re

# move to same directory as this script
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
os.chdir("src")

include = os.path.join("..", "include")
bp = os.path.join("..", "bin")
cflags += f' -Xclang -fopenmp -I{path_to_openmp} -I{include} -I{os.path.join(include, "Layers")} -I{os.path.join(include, "Optimizers")}'

sources = [os.path.join(root, file) for root, dirs, files in os.walk(".") for file in files if file[-3:] == "cpp"]
objects = [os.path.join(bp, os.path.basename(x[:-3] + "o")) for x in sources]

outfile = "Makefile"

def findFile(path):
	if os.path.exists(path):
		return path
	elif os.path.exists(os.path.join(include, path)):
		return os.path.join(include, path)
	else:
		return ""

with open(outfile, 'w') as sys.stdout:
	print(f'''\
cc={cc}
cflags={cflags}
.PHONY: all
all: {bp} {' '.join(objects)}

{bp}:
	mkdir {bp} || true
''')

	for source in sources:
		basepath = os.path.dirname(source)

		headers = [findFile(os.path.join(basepath, line.strip()[10:-1])) for line in open(source) if re.match(r'#include ".*"',line)]
		headers = ' '.join(os.path.normpath(x) for x in headers if x != "")

		print(f"{os.path.normpath(os.path.join(bp, os.path.basename(source[:-4] + '.o')))}: {source} {headers}")
		print("\t${cc} ${cflags} -c $< -o $@\n")