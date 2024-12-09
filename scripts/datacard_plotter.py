import uproot
import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(root_file_path, config, eras, categ_to_var):
    """
    Generate CMS-style stack plots for histograms directly from a ROOT file.

    Parameters:
    - root_file_path: Path to the ROOT file containing histograms.
    - config: Configuration dictionary from the YAML file.
    - eras: List of eras (e.g., ["2022_preEE"]).
    - categ_to_var: Mapping of categories to variables and names.
    """
    signal_scaling = config["plotting"].get("signal_scaling", 1)
    blinding_config = config["plotting"].get("blinding", {})

    with uproot.open(root_file_path) as root_file:
        for era in eras:
            for cat, var_and_name in categ_to_var.items():
                variable, newCatName = var_and_name
                blinded_region = blinding_config.get(cat, {})

                # Collect histograms
                stack_components = {"TT": None, "VJet": None, "VV": None, "ZH_hbb": None}
                signal_hist = None
                data_hist = None

                # Load histograms for stack components
                for proc in stack_components.keys():
                    hist_name = f"{era}_{newCatName}/{proc}_Shape_nominal"
                    if hist_name in root_file:
                        hist = root_file[hist_name]
                        values, edges = hist.to_numpy(flow=False)
                        stack_components[proc] = (values, edges)

                # Load signal histogram
                signal_name = f"{era}_{newCatName}/ZH_hbb_Shape_nominal"
                if signal_name in root_file:
                    hist = root_file[signal_name]
                    values, edges = hist.to_numpy(flow=False)
                    signal_hist = (values, edges)

                # Load data histogram
                data_name = f"{era}_{newCatName}/data_obs_Shape_nominal"
                if data_name in root_file:
                    hist = root_file[data_name]
                    values, edges = hist.to_numpy(flow=False)
                    data_hist = (values, edges)

                # Prepare plot
                fig = plt.figure(figsize=(6, 6))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_main = fig.add_subplot(gs[0])
                ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

                bottoms = None

                # Plot stack components
                colors = {"TT": "#02baf7", "VJet": "#f5f768", "VV": "#91bfdb", "ZH_hbb": "#ff0000", "ZH_hcc": "#b700ff"}
                labels = {"TT": "TT", "VJet": "VJet", "VV": "VV", "ZH_hbb": "ZH_hbb", "ZH_hcc": "ZH_hcc"}
                bottoms = None
                stack_uncertainty = None  # Initialize stack uncertainty
                for proc, hist_data in stack_components.items():
                    if hist_data is not None:
                        values, edges = hist_data
                        bin_centers = edges[:-1] + np.diff(edges) / 2  # Compute bin centers
                        bin_widths = np.diff(edges)  # Compute bin widths

                        # Blinding logic
                        if blinded_region and variable in blinded_region:
                            lower, upper = blinded_region[variable]
                            lower = float(lower) if isinstance(lower, (int, str)) else lower
                            upper = float(upper) if isinstance(upper, (int, str)) else upper
                            mask = (edges[:-1] > (lower if lower is not None else -float('inf'))) & \
                                   (edges[:-1] < (upper if upper is not None else float('inf')))
                            values[mask] = 0  # Blind the region

                        # Use bin centers for alignment with signal and data
                        ax_main.bar(bin_centers, values, width=bin_widths, bottom=bottoms, color=colors[proc], label=labels[proc])
                        if bottoms is None:
                            bottoms = values
                        else:
                            bottoms += values
                            
                # Calculate stack uncertainties (upper and lower bounds)
                stack_uncertainty = np.sqrt(bottoms)

                # Plot statistical uncertainty band on the upper plot
                ax_main.fill_between(bin_centers, bottoms - stack_uncertainty, bottoms + stack_uncertainty, 
                                     color="gray", alpha=0.3, step="mid", label="Stat. Unc.")

                # Plot signal
                if signal_hist:
                    values, edges = signal_hist
                    values *= signal_scaling
                    ax_main.step(edges, np.append(values, values[-1]), where="post", color="black", linestyle="--", label=f"Signal x{signal_scaling}")

                # Plot data
                if data_hist:
                    values, edges = data_hist
                    bin_centers = edges[:-1] + np.diff(edges) / 2
                    ax_main.errorbar(bin_centers, values, yerr=np.sqrt(values), fmt="o", color="black", label="Data")

                # Ratio plot: Data/MC
                if data_hist and bottoms is not None:
                    data_values, edges = data_hist
                    bin_centers = edges[:-1] + np.diff(edges) / 2

                    # Ensure no division by zero
                    safe_bottoms = np.where(bottoms == 0, 1e-10, bottoms)
                    ratio = data_values / safe_bottoms
                    ratio_unc = np.sqrt(data_values) / safe_bottoms  # Uncertainty in the ratio
                    
                    # Statistical uncertainty band
                    stack_uncertainty = np.sqrt(bottoms)
                    ratio_band = stack_uncertainty / safe_bottoms

                    # Plot the ratio
                    ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_unc, fmt="o", color="black", label="Data/MC")
                    # Plot uncertainty band
                    ax_ratio.fill_between(bin_centers, 1 - ratio_band, 1 + ratio_band, 
                                          color="gray", alpha=0.3, step="mid", label="Stat. Unc.")
                    
                    ax_ratio.axhline(1, color="gray", linestyle="--")  # Reference line at ratio=1
                    ax_ratio.set_ylabel("Data/MC")
                    ax_ratio.set_ylim(0.5, 1.5)  # Adjust y-axis limits for better visualization
                    ax_ratio.grid(axis="y", linestyle="--", alpha=0.5)

                # Formatting for the main plot
                ax_main.set_ylabel("Events")
                ax_main.legend()
                ax_main.grid(axis="y", linestyle="--", alpha=0.5)
                ax_main.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels on the upper plot

                # Formatting for the ratio plot
                ax_ratio.set_xlabel(variable)
                ax_ratio.set_ylabel("Data/MC")
                ax_ratio.grid(axis="y", linestyle="--", alpha=0.5)

                # CMS text
                ax_main.text(0, 1.02, "CMS", fontsize=16, fontweight="bold", transform=ax_main.transAxes)
                ax_main.text(0.2, 1.02, "Private Work", fontsize=12, style="italic", transform=ax_main.transAxes)
                ax_main.text(1, 1.02, f"{era}: {newCatName}", fontsize=12, ha="right", transform=ax_main.transAxes)

                # Save plot
                plot_file = f"plot_datacards/stack_plot_{newCatName}_{era}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"Saved plot: {plot_file}")

                # Set y-axis to log scale
                ax_main.set_yscale("log")
                ax_main.set_ylim(bottom=0.1)  # Ensure the lower limit is positive for log scale
                ax_main.grid(axis="y", linestyle="--", alpha=0.5)  # Update grid for log scale

                # Save log scale plot
                plot_file_log = f"plot_datacards/stack_plot_{newCatName}_{era}_log.png"
                plt.savefig(plot_file_log, dpi=300, bbox_inches="tight")
                print(f"Saved log-scale plot: {plot_file_log}")

                plt.close()
