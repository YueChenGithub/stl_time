This code is responsible for the master thesis "Enhancing Spatial and Temporal Robustness of Signal Temporal Logic for Control Systems" by Minghua Chen.

The code is tested on Ubuntu 22.04 with Python 3.8.

## 1. Create Environment

Fist create a conda environment with the following command:

`conda env create -n stl --file ENV.yml`

and install and activate Gurobi following:

`https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-`


## 2. Run the code

To run the code, first activate the environment:

`conda activate stl`

Then run the code for the first example and using spatial robustness optimization:

`python ./stl/1_ge.py -method s -x0 0`

A list of method options are:

| Method  | Robustness                                                                 |
|---------|----------------------------------------------------------------------------|
| s       | Spatial Robustness                                                         |
| t_min   | (Both Hand) Temporal Robustness                                            |
| t_left  | Left Hand Temporal Robustness                                              |
| t_right | Right Hand Temporal Robustness                                             |
| c_min   | Predicate Level multiplicative Robustness                                  |
| c_sum2  | Predicate Level additive Robustness                                        |
| G_s     | Operator Level Robustness w.r.t. Spatial Robustness                        |
| G_c_min | Operator Level Robustness w.r.t. Predicate Level multiplicative Robustness |
| mul     | Specification Level multiplicative Robustness                              |
| sum     | Specification Level additive Robustness                                    |
| xi      | Specification Level accumulative Robustness                                |
| 0       | Satisfaction Only                                                          |