# PocketCoffea VHbb setup
**This manual originates from the [VHccPoCo](https://github.com/cms-rwth/VHccPoCo)**

---
**Recommended**

More information can be found in the installation [instruction](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

```
apptainer shell -B /afs -B /cvmfs/cms.cern.ch  -B /tmp  -B /eos/cms/  -B /etc/sysconfig/ngbauth-submit  -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable --no-env

```

For local environment setup:

```
git clone git@github.com:PocketCoffea/PocketCoffea.git
cd PocketCoffea

python -m venv --system-site-packages myPocket
source myPocket/bin/activate
pip install -e .[dev]

pip install lightgbm tensorflow==2.12.0
pip install setuptools==70.*
```

For the later login:

```
# Activate the virtual environment
cd PocketCoffea
source myPocket/bin/activate
```

---
## For Analysis
1. (If your local username is different from your CERN username) Setup your CERN username variable:
    ```
    export CERN_USERNAME="YOURUSERNAME"
    ```

2. Follow [examples](https://pocketcoffea.readthedocs.io/en/latest/analysis_example.html) to create dataset input files (Add `--overwrite` if the outputs are already there.):

   ```
	cd VHccPoCo
	mkdir datasets
	build-datasets --cfg samples_Run3.json -o -rs 'T[123]_(IT|UK|US)_\w+' -ir # or
    ```
    
3. Run with the `futures` executor (test before large submission) in the same dir where the VHbbCoPo is:
    ```
    runner --cfg VHccPoCo/cfg_VHbb_ZLL.py -o output_vhbb_zll_dev_local --executor futures -s 10 -lf 1 -lc 1 2>&1 | tee output_vhbb_zll_local.log  # local test
    ```

4. Run on condor (only if the previous step was successeful):
    ``` 
    runner --cfg VHccPoCo/cfg_VHbb_ZLL.py --executor dask@lxplus --custom-run-options VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_run_options_vhbb_test.yaml  -o output_vhbb_zll_dev_all 2>&1 | tee dask_out.log &

    ```
    
    **you** can now try the new job submission on condor, if you are wasting time with dask failure of random files. Give it a try with `-e condor@lxplus --max-events-per-job N --dry-run`
    

5. Make some plots:
   ```
   cd output_vhbb_zll_dev_all
   make-plots -i output_all.coffea --cfg parameters_dump.yaml -o plots -op ../VHccPoCo/params/skip_plot_opt_vhbb_run3/custom_plot_options_vhbb_ZLL.yaml
   ```
   
6. produce datacards:
     ```
     python VHccPoCo/scripts/convertToRoot.py path/to/your/output/output_all.coffea -c VHccPoCo/params/shapemaker_vhbb_run3/Zll_Shape.yaml
     ```
     
---