## Scripts

1. `convertToRoot.py` - a script to convert .coffea output inot the ROOT histograms for input to Combine. Work in progress...
Usage:
```
python VHccPoCo/scripts/convertToRoot.py  path/to/output_all.coffea -c path/to/params/shape_config.yaml
```

2. `compute_ctagSF_calibration.py` - a script to compute normalisation corrections after c-tagger shape correction scale factors.
   -  `plot_shapes_comparison.py` - a script to make plots and validate the above corrections.
