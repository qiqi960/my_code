Benchmarking nearest neighbors
==============================

This project contains tools to benchmark various implementations of approximate nearest neighbor (ANN) search for selected metrics. We have pre-generated datasets (in HDF5 format) and prepared Docker containers for each algorithm, as well as a test suite to verify function integrity.

Data sets
=========

We have a number of precomputed data sets in HDF5 format. All data sets have been pre-split into train/test and include ground truth data for the top-100 nearest neighbors.

Install
=======

The only prerequisite is Python (tested with 3.10.6) and Docker.

1. Run `pip install -r requirements.txt`.
2. Run `python install.py` to build all the libraries inside Docker containers.

Running
=======

1. Run `python run.py` (this can take an extremely long time, potentially days)
2. Run `python plot.py` or `python create_website.py` to plot results.
3. Run `python data_export.py --out res.csv` to export all results into a csv file for additional post-processing.

You can customize the algorithms and datasets as follows:

* Check that `ann_benchmarks/algorithms/{YOUR_IMPLEMENTATION}/config.yml` contains the parameter settings that you want to test
* To run experiments on SIFT, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time. 
* To process the results, either use `python plot.py --dataset glove-100-angular` or `python create_website.py`. An example call: `python create_website.py --plottype recall/time --latex --scatter --outputdir website/`. 
