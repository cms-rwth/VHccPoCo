# PocketCoffea VHcc setup

1. Install Miniconda or Micromamba
   * At RWTH it should be at your `/net/scratch_cms3a/<username>` area
   * At lxplus CERN, it should be in your `/eos/user/u/username` area
2. Create a dedicated environment for `PocketCoffea`, install the packages, and compile:
    ```
	conda create -n PocketCoffea python=3.10 -c conda-forge
	conda activate PocketCoffea
	# install PocketCoffea
	git clone git@github.com:PocketCoffea/PocketCoffea.git
	cd PocketCoffea
	pip install -e .
    ```
	Follow [their installation instructions](https://pocketcoffea.readthedocs.io/en/latest/installation.html) for other options.
	Afterwards install additional packages needed for BDT/DNN training and evaluation. Keep using conda, since using pip might
	alter the environment, leading to conflicts.
    ```
	conda install conda-forge::xrootd
	conda install conda-forge::lightgbm
	conda install conda-forge::tensorflow
    conda install pytorch::pytorch
    conda install conda-forge:alive-progress
    conda install conda-forge:optuna
    conda install conda-forge:imblearn
    ```
    For brux20 cluster at Brown, you may need `conda install conda-forge::ca-certificates`.
3. Checkout *this* repo:
    ```
	git clone git@github.com:cms-rwth/VHccPoCo.git
    ```
4. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```
5. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files. First activate voms proxy. Then:
    ```
	cd VHccPoCo
	mkdir datasets
	build-datasets --cfg samples_Run3.json -o -ws T2_DE_RWTH -ws T2_DE_DESY -ws T1_DE_KIT_Disk -ws T2_CH_CERN -ir
	cd ../
    ```
    Use `-p 12` with `build-datasets` to parallelizing with 12 cores.
6. Run with the `futures` executor (test before large submission):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_Test --executor futures -s 10 -lf 1 -lc 1
    ```
7. Run on condor with Parsl executor (only if the previous step was successeful):
    ```
    runner --cfg VHccPoCo/cfg_VHcc_ZLL.py -o output_VHcc_v01 --executor parsl-condor@RWTH -s 60
    ```
	Note: use `dask@lxplus` executor if running at CERN.

8. Make some plots:
   ```
   make-plots -inp output_VHcc_v01 -op VHccPoCo/params/plotting.yaml
   ```
   The plot parameters can be changed by editing `VHccPoCo/params/plotting.yaml`.

9. Produce shapes for limit setting with `scripts/convertToRoot.py` script.
   * See details in [scripts/README.md](scripts/README.md)
