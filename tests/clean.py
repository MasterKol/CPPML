#!/usr/bin/env python3

import os
import re
import subprocess

for root, dirs, files in os.walk("."):
	for dir1 in dirs:
		path = os.path.join(root, dir1)
		if os.path.exists(path) and re.search("\.dSYM$", path):
			subprocess.call(["rm", "-r", path])

for root, dirs, files in os.walk("."):
	for f in files:
		file = os.path.join(root, f)
		if os.path.exists(file) and re.search("\.tst$", file):
			subprocess.call(["rm", file])