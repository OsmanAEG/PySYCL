import os
import subprocess
import unittest
import math

def find_python_test_files(directory):
	"""
	Find all the python test files in the given directory.

	:param directory: The directory to search for python test files.
	:type directory: str

	:returns: A list of python test files.
	:rtype: list
	"""
	python_test_files = []
	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith("_Tests.py"):
				python_test_files.append(os.path.join(root, file))
	return python_test_files

def run_python_tests(python_test_files):
	for python_test_file in python_test_files:
		print("Running python tests in: " + python_test_file)
		subprocess.call(["python", python_test_file])

if __name__ == "__main__":
	cmake_source_directory = "."
	python_test_files = find_python_test_files(cmake_source_directory)
	run_python_tests(python_test_files)