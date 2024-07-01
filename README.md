# PocketCoffea VHcc setup

1. Install Miniconda
   * At RWTH it should be at your `/net/scratch_cms3a/<username>` area
   * At lxplus CERN, it should be in your `/eos/user/u/username` area
3. Create a dedicated environment for PocketCoffea, install the packages, and compile:
    ```
	conda create -n PocketCoffea python=3.10 -c conda-forge
	conda activate PocketCoffea
    pip install -e .
    ```
	Follow [their installation instructions](https://pocketcoffea.readthedocs.io/en/latest/installation.html) for other options.
        Afterwards please install the needed packages, that enable trainings and running the analysis smoothly. Please keep using conda, since using pip might 
        alter the environment leading to conflicts.
    ```
	conda install conda-forge::xrootd
	conda install -c conda-forge lightgbm
    conda install conda-forge::tensorflow
    ```
5. Checkout *this* repo:
    ```
    git@github.com:cms-rwth/VHccPoCo.git
    ```
3. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```
5. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files:
    ```
    build_datasets --cfg samples_Run2UL_2017.json -o -ws T2_DE_RWTH -ws T2_DE_DESY -ws T1_DE_KIT_Disk -ws T2_CH_CERN -ir
    ```
6. Run with the `futures` executor (test before large submission):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_Test --executor futures -s 10 -lf 1 -lc 1
    ```
7. Run on condor with Parsl executor (only if the previous step was successeful):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_v01 --executor parsl-condor@RWTH -s 60
    ```
8. Make some plots:
   ```
   make_plots output_VHcc_v01
   ```
