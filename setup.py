import setuptools

with open("README.md", "r") as fh:
	description = fh.read()

setuptools.setup(
	name="mesa_inlist_manager",
	version="0.0.1",
	author="Henrik Knierim",
	author_email="henrik.knierim@uzh.ch",
	packages=["mesa_inlist_manager"],
	description="A simple package for changing MESA inlists.",
	long_description=description,
	long_description_content_type="text/markdown",
	url="https://github.com/Henrik-Knierim/mesa_inlist_manager",
	license='MIT',
	python_requires='>=3.9',
	install_requires=["os"]
)
