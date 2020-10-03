[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trevortknguyen/dimensionality)

# Dimensionality

This repository is supposed to investigate dimensionality reduction methods
suitable for extracellular neural recordings.

## Dependencies not in Nixpkgs

This requires the following GitHub repositories:
- Eden-Kramer-Lab/regularized_glm
- Eden-Kramer-Lab/replay_trajectory_classification
- flatironinstitute/mountainsort

They can be cloned and soft linked to be used by their source.

## Directory sturcture
- ./figures: put some figures here
- ./neural_dimensionality_reduction: put code here
- ./: put all notebooks here

## DATA_DIR

This repository is still in a draft state. The data files are specified by an
environmental variable for the directory. It can be useful to use a `.envrc`
file to manage this.
