#!/usr/bin/env python3

import os
import subprocess
import sys

# split flags and files
arg_set = set(sys.argv[1:])
flags = {x for x in arg_set if x[0] == '-'}
tests = arg_set - flags

verbose = False
if '-v' in flags:
	verbose = True
	flags.remove('-v')

# move to same directory as this scripts
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def printGreen(txt):
	print('\033[0;32m' + str(txt) + '\033[0m')

def printRed(txt):
	print('\033[0;31m' + str(txt) + '\033[0m')

def printProc(proc):
	data = proc.communicate()
	out = data[0].decode()
	err = data[1].decode()

	print(('\n' if len(out) > 0 else '') + out, end='')
	print('\n' if (len(out) > 0 or len(err) > 0) else '', end='')
	print(err + ('\n' if len(err) > 0 else ''), end='')

# run test at specified location
def runSingleTest(test):
	if os.path.isfile(test + ".cpp"):
		test = test + ".tst"
	elif test[-3:] == "cpp" and os.path.isfile(test):
		test = test[:-3] + "tst"
	else:
		print("Test " + test + " does not exist")
		os.exit(1)

	# try and rebuild library if necessary
	if subprocess.call('cd ..; make', shell=True) != 0:
		exit(1)

	# try compiling test
	if subprocess.call(['make', test, f"CXXFLAGS={' '.join(x for x in flags)}"]) != 0:
		exit(1)
	
	# call test
	process = subprocess.Popen("./" + test, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	process.wait()

	if process.returncode == 0: # success
		if verbose: printProc(process)
		printGreen("Passed " + test)
	else: #failure
		printProc(process)

		printRed("\nFailed Test " + test)
		os.exit(1) # exit with code 1 to indicate failure

# run all tests in the same folder and sub-folders of this program
def runAllTests():
	# get all files ending in .cpp
	sources = [os.path.join(root, file) for root, dirs, files in os.walk(".") for file in files if file[-3:] == "cpp"]

	for source in sources:
		runSingleTest(source)

# no files specified, run all tests
if len(tests) == 0:
	tests.add('.')

# some tests were specified, try to run them
for arg in tests:
	if os.path.isfile(arg) or os.path.isfile(arg + ".cpp"):
		runSingleTest(arg)
	else:
		for root, dirs, files in os.walk(arg):
			for file in files:
				if file[-3:] == "cpp":
					runSingleTest(os.path.join(root, file))