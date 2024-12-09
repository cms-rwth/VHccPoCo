import os,sys
import yaml
import argparse
import uproot
from coffea.util import load, save

from datacard_plotter import plot_histograms

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def testFileStructure(hists, example_variable, example_data, example_MC, example_subsample, eras, example_category, variations):
    # information printing
    print("The structure of the COFFEA file:")
    print('\t top keys:\n', hists.keys())
    print('\n\t Variables:\n', hists["variables"].keys())
    print('\n\t Columns:\n', hists["columns"].keys())
    print('\n\t Samples:\n', hists["variables"][example_variable].keys())
    print('\n\t Subsamples (for DATA):\n', hists["variables"][example_variable][example_data].keys())
    print(f'\n\t Subsamples (for {example_MC}):\n', hists["variables"][example_variable][example_MC].keys())
    if example_subsample in hists["variables"][example_variable].keys():
        print(f'\n\t Subsamples (for {example_subsample}):\n', hists["variables"][example_variable][example_subsample].keys())
    era0 = eras[0]
    print(f'\n\t A Histogram (for {example_MC}:):\n', hists["variables"][example_variable][example_MC][f'{example_MC}_{era0}'])
    print('\n\t Draw it: \n', hists["variables"][example_variable][example_MC][f'{example_MC}_{era0}'][{'cat':example_category, 'variation': variations}])


def convertCoffeaToRoot(coffea_file_name, config):

    inputfile = coffea_file_name
    Channel = config["input"]["channel"]
    eras = config["input"]["eras"]
    variations = config["input"]["variations"]

    example_variable = config["config"]["example_variable"]
    example_data = config["config"]["example_data"]
    example_MC = config["config"]["example_MC"]
    example_subsample = config["config"]["example_subsample"]
    example_category = config["config"]["example_category"]

    hists = load(inputfile)

    testFileStructure(hists, example_variable, example_data, example_MC, example_subsample, eras, example_category, variations)

    # Here we decide which histograms are used for coffea -> root conversion
    # and, possibly, a NEW name of the category

    categ_to_var = {
        category: [details['observable'], details['new_name']]
        for category, details in config["categories"].items()
    }

    output_dict = {}
    # Which processes to consider:
    samples = hists["variables"][example_variable].keys()

    map_sampleName_to_processName = config.get("sample_to_process_map", {})
    sample_to_merge_list = config.get("sample_to_merge_list", {})

    for era in eras:
        for samp in samples:
            isData=False
            if 'DATA' in samp:
                print(samp)
                isData=True
            if samp not in map_sampleName_to_processName.keys():
                continue
            proc = map_sampleName_to_processName[samp]
            print('Processing sample:', samp, ' assigned process:', proc)

            for cat, var_and_name in categ_to_var.items():
                #print('\t Converting histogram for:', cat, var_and_name)
                variable = var_and_name[0]
                newCatName = var_and_name[1]
                for variation in variations:
                    if isData:
                        if variation!='nominal':
                                continue
                        if ('DATA_DoubleMu' in samp and ('mm' in cat or 'll' in cat)) \
                           or ('DATA_EGamma' in samp and ('ee' in cat or 'll' in cat)) \
                           or ('DATA_SingleMu' in samp and 'mn' in cat) \
                           or ('DATA_EGamma' in samp and 'en' in cat) \
                           or ('DATA_MET' in samp and 'nn' in cat):
                            pass
                        else:
                            # This shall not pass!
                            continue
                    subsamples = [sname for sname in hists['variables'][variable][samp].keys() if era in sname]
                    #print("\t Subsamples:", subsamples)
                    if len(subsamples)==0:
                        print("Something is wrong. Probably ERA is not correct:", era)
                        print("\t list of available subsamples:",hists['variables'][variable][samp].keys())
                        sys.exit(1)
                    elif len(subsamples)==1:
                        # Note: only nominal is done here.
                        # Need to loop over variations to get shape systematics (todo)
                        if isData:
                            myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat}]
                        else:
                            myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': variation}]
                    else:
                        print("\t Subsamples:", subsamples)
                        # We need to add all the histograms for sub-samples
                        if isData:
                            myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat}]
                        else:
                            myHist = hists['variables'][variable][samp][subsamples[0]][{'cat':cat, 'variation': variation}]

                        for i in range(1,len(subsamples)):
                            if isData:
                                # hist_i = myHist = hists['variables'][variable][samp][subsamples[i]][{'cat':cat}]
                                hist_i = hists['variables'][variable][samp][subsamples[i]][{'cat': cat}]
                            else:
                                # hist_i = myHist = hists['variables'][variable][samp][subsamples[i]][{'cat':cat, 'variation': variation}]
                                hist_i = hists['variables'][variable][samp][subsamples[i]][{'cat': cat, 'variation': variation}]
                            myHist = myHist + hist_i

                    output_dict[era+'_'+newCatName+'/'+proc+'_Shape_'+f'{variation}'] = myHist
                    
        # Add merged histograms for categories in sample_to_merge_list
        for merged_name, merge_samples in sample_to_merge_list.items():
            print(f'Creating merged histogram for {merged_name}')
            for cat, var_and_name in categ_to_var.items():
                variable = var_and_name[0]
                newCatName = var_and_name[1]

                for variation in variations:  # Iterate over all variations (nominal, Up, Down)
                    merged_hist = None
                    for samp in merge_samples:
                        proc = map_sampleName_to_processName.get(samp, None)
                        if proc is None:
                            print(f"Sample {samp} not found in sample_to_process_map. Skipping...")
                            continue
                        hist_key = era + '_' + newCatName + '/' + proc + '_Shape_' + f'{variation}'
                        if hist_key not in output_dict:
                            print(f"Histogram for {samp} not found in output_dict. Skipping...")
                            continue
                        if merged_hist is None:
                            merged_hist = output_dict[hist_key]
                        else:
                            merged_hist = merged_hist + output_dict[hist_key]

                    if merged_hist is not None:
                        merged_key = era + '_' + newCatName + '/' + merged_name + '_Shape_' + f'{variation}'
                        output_dict[merged_key] = merged_hist
                        print(f"Merged histogram saved for {merged_key}")

        # Explicitly handle data merging
        if "data_obs" in sample_to_merge_list:
            print(f"Merging data samples into data_obs")
            for cat, var_and_name in categ_to_var.items():
                variable = var_and_name[0]
                newCatName = var_and_name[1]

                merged_data_hist = None
                for data_sample in sample_to_merge_list["data_obs"]:
                    data_key = era + '_' + newCatName + '/' + data_sample + '_Shape_nominal'
                    if data_key not in output_dict:
                        print(f"Data sample {data_sample} not found in output_dict. Skipping...")
                        continue
                    if merged_data_hist is None:
                        merged_data_hist = output_dict[data_key]
                    else:
                        merged_data_hist = merged_data_hist + output_dict[data_key]

                if merged_data_hist is not None:
                    merged_data_key = era + '_' + newCatName + '/data_obs_Shape_nominal'
                    output_dict[merged_data_key] = merged_data_hist
                    print(f"Merged data histogram saved for {merged_data_key}")


        shapes_file_name = config["output"]["shapes_file_name"]+"_"+era+"_"+Channel+".root"        
        with uproot.recreate(shapes_file_name) as root_file:
            for shape, histogram in output_dict.items():
                root_file[shape] = histogram
                
    return shapes_file_name

if __name__ == "__main__":

    print("Hello world")

    parser = argparse.ArgumentParser(description="Run the Coffea to ROOT converter with a specified config file.")
    parser.add_argument( "inputfile", type=str, default="", help="Path to the input .coffea file.")
    parser.add_argument( "-c", "--config", dest="config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    coffea_file_name = args.inputfile
    root_file = convertCoffeaToRoot(coffea_file_name, config)
    
    # Add plotting step
    categ_to_var = {
        category: [details['observable'], details['new_name']]
        for category, details in config["categories"].items()
    }
    plot_histograms(root_file, config, config["input"]["eras"], categ_to_var)

    print("... and goodbye.")
