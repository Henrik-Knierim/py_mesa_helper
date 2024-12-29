#!/bin/bash

# Run the unit tests
python -m unittest discover -s tests

# Check if the tests passed
if [ $? -eq 0 ]; then
    # Install the package
    pip install .
else
    echo "Tests failed. Installation aborted."
    exit 1
fi