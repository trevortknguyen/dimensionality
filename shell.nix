{ pkgs ? import <nixpkgs> {} }:

let
  # python environment
  my-python-packages = python-packages: with python-packages; [
    # for pyms
    numpydoc

    # for decoder
    xarray
    networkx
    numba
    dask
    distributed
    patsy
    statsmodels

    # autoencoder stuff
    pytorch

    # reading data
    netcdf4
    pandas
    numpy
    scipy
    scikitlearn

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

