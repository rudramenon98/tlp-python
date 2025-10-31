# Run this script from the parent directory of the packages

autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables ./packages
isort ./packages
black ./packages
