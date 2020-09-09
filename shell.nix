{ pkgs ? import <nixpkgs> {} }:

let
  # python environment
  my-python-packages = python-packages: with python-packages; [
    numpy
    scipy

    # plotting in notebooks: plotly interactive, seaborn quick
    plotly
    seaborn

    # notebooks
    jupyter
  ]; 
  python-with-my-packages = pkgs.python3.withPackages my-python-packages;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python-with-my-packages
  ];
}

