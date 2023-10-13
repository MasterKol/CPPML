#!/usr/bin/env python3

import os
import subprocess
import sys

# move to same directory as this scripts
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def printGreen(txt):
	print('\033[0;32m' + str(txt) + '\033[0m')

def printRed(txt):
	print('\033[0;31m' + str(txt) + '\033[0m')

# run all tests in the same folder and sub-folders of this program
def runAllTests():
	# get all files ending in .cpp
	sources = [os.path.join(root, file) for root, dirs, files in os.walk(".") for file in files if file[-3:] == "cpp"]

	# change extensions from .cpp to .tst
	tests = [x[:-3] + "tst" for x in sources]

	for test in tests:
		# make test, exit if build fails
		if subprocess.call(['make', test]) != 0:
			exit(1)

		# call test
		process = subprocess.Popen("./" + test, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		process.wait()

		if process.returncode == 0: # success
			printGreen("Passed " + test)
		else: #failure
			# print output
			print(process.communicate())

			printRed("\nFailed Test " + test)

			os.exit(1) # exit with code 1 to indicate failure

# run test at specified location
def runSingleTest(test):
	if os.path.isfile(test + ".cpp"):
		test = test + ".tst"
	elif test[-3:] == "cpp" and os.path.isfile(test):
		test = test[:-3] + "tst"
	else:
		print("Test " + test + " does not exist")
		os.exit(1)

	# rebuild library if necessary
	subprocess.call('cd ..; make', shell=True)

	# try compiling test
	if subprocess.call(['make', test]) != 0:
		exit(1)
	
	# call test
	process = subprocess.Popen("./" + test)
	process.wait()

	if process.returncode == 0: # success
		printGreen("Passed " + test)
	else: #failure
		printRed("\nFailed Test " + test)

		os.exit(1) # exit with code 1 to indicate failure

if len(sys.argv) > 2:
	print("Invalid usage, runtests expects at most 1 argument")
	os.exit(1)

if len(sys.argv) == 2:
	runSingleTest(sys.argv[1])
else:
	runAllTests()