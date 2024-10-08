from setuptools import setup, find_packages
import glob

with open("README.md", "r") as fh:
	description = fh.read()

setup(
	name="mesa_helper",
	version="1.0.0",
	author="Henrik Knierim",
	author_email="henrik.knierim@gmail.com",
	packages=find_packages(include=['mesa_helper','mesa_helper.*']),
	description="A simple package for changing MESA inlists and analyzing MESA outputs.",
	long_description=description,
	long_description_content_type="text/markdown",
	url="https://github.com/Henrik-Knierim/py_mesa_helper",
	license='MIT',
	python_requires='>=3.10',
	install_requires=[
		"mesa_reader>=0.3.0",
		'numpy',
		'scipy',
		'matplotlib',
		'pandas',
		]
)
