# SGPR-memristor
Code for "Accelerating Machine Learning-Based Memristor Compact Modeling Using Sparse Gaussian Process"

Modeling memristors using sparse Gaussian process regression.

## Preparation
- Install `Anaconda 4.10.1`
- Build a virtual environment with the following command:
    ```sh
    $ conda env create -f=./env_conda.yml
    ```
- Activate the virtual environment using the following command:
    ```sh
    $ conda activate py36_sgpr
    ```


## Execution Steps
1. Run `$ make train` to generate the model.
2. Run `$ make hspice` to convert the generated model into Verilog and execute it.

## Additional Information
The execution results of Hspice in step 2 are outputted in the `img/` directory.

## Explanation of Each File
```
data/ : The place to save csv data
HSPICE_*/ : The place to save Hspice files
img/ : The place to save figures outputted by matplotlib
save_sgpr/ : The place to save model data executed by Pyro
temp/ : Verilog-A templates files
utils/ : Utilities for the program
verilog/ : The place where Verilog-A files are located
params.py : The program that converts model data saved in save_sgpr/ into necessary data for Verilog-A
read_plot_mosfet.py : The program that reads data executed by hspice and outputs figures
sgpr_train_current.py : The program to generate a current prediction model with Pyro
sgpr_train_state.py : The program to generate a state prediction model with Pyro
sgpr_train.py : The program that runs sgpr_train_current.py and sgpr_train_state.py
```

## Additional experiments
The scripts in `experiments/` train the models, convert them into Verilog-A, run HSPICE, and collect the training time, the HSPICE runtime, and the errors in one csv file. They do not overwrite `save_sgpr/` or `verilog/`; each run is written to `runs/<name>/`.

- Install the packages in `requirements.txt` (the scripts were run with Python 3.12.3, PyTorch 2.15.0, and Pyro 1.9.2; `env_conda.yml` is the environment of the original scripts).
- Run all the steps:
    ```sh
    $ bash experiments/run_all_memristor.sh
    ```
  The results are written to `results/rerun/sweep.csv`. `DRY=1 bash experiments/run_all_memristor.sh` skips HSPICE and uses a Python emulation of `verilog/rram_gp.va` to check the scripts.

```
experiments/run_all_memristor.sh : Runs the steps below in order
experiments/run_memristor_sweep.py : SGPR with the number of inducing points Z = 50, 200, 600 (settings of sgpr_train_*.py),
                                     GPR without sparsity (--model gpr),
                                     Matern 3/2, Matern 5/2, and rational quadratic kernels (--kernels),
                                     and SGPR without optimization of the hyperparameters and inducing points (--stop none)
experiments/run_baseline_lin.py : Evaluates the model of https://github.com/albertlin11/RRAMunif with the same HSPICE decks.
                                  --stored uses the trained model included in that repository; without it, the model is trained with the script of that repository (TensorFlow is required)
experiments/plot_memristor_figs.py : Plots the measured data and the HSPICE results
experiments/results/ : csv files of the results obtained on an AMD Ryzen Threadripper PRO 7995WX CPU (10 threads for the training, HSPICE U-2023.03-SP1)
```

Columns of `sweep.csv`: `train_time` is the sum of the training times of the state model and the current model (median of three trainings), `runtime_test` is the HSPICE runtime for `HSPICE_test/1R.sp` (mean of three runs), and `rmse_*_mA` and `rmsle_*_x1000` are the errors of the current calculated as in `utils/plot_MLmodel.py`.

## Reference
We would greatly appreciate it if you could kindly acknowledge our tool in your research by referencing the following publications:
````
@inproceedings{DATE2024_yshintani,
   title={Accelerating Machine Learning-Based Memristor Compact Modeling Using Sparse Gaussian Process},
   author={Yuta Shintani and Michiko Inoue and Michihiro Shintani},
   booktitle={in Proceedings of the Conference on Design, Automation \& Test in Europe (DATE)},
   year={2024}
}
````
