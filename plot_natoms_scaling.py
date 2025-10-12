import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Set seaborn style for poster
sns.set_style("whitegrid")
# sns.set_palette("husl")
sns.set_context("poster")

base_dir = "plots/natoms_scaling"
os.makedirs(base_dir, exist_ok=True)

force_download = True

# Check if parquet file exists
parquet_file = "wandb_natoms_scaling.parquet"

if os.path.exists(parquet_file) and not force_download:
    print("Loading existing parquet file...")
    runs_df = pd.read_parquet(parquet_file)
else:
    print("Downloading data from wandb...")
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("andreas-burger/hip")

    list_of_runs = []
    for run in runs:
        if (
            "split" not in run.config["training"]
            or run.config["training"]["split"] != "size"
        ):
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_dict = run.summary._json_dict
        # Keep only columns containing "val"
        summary_dict = {k: v for k, v in summary_dict.items() if "val" in k}

        if "eval_8-Loss E" not in summary_dict:
            continue

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_dict = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # Recursively flatten dict
        config_dict = flatten_dict(config_dict)
        # Keep only columns containing "weight" or "split"
        config_dict = {
            k: v for k, v in config_dict.items() if "weight" in k or "split" in k
        }

        # .name is the human-readable name of the run.
        list_of_runs.append(
            {
                "name": run.name,
                **config_dict,
                **summary_dict,
            }
        )

    runs_df = pd.DataFrame(list_of_runs)

    # Save the downloaded data
    runs_df.to_parquet(parquet_file)
    print("Data saved to parquet file.")

# List the columns
print("DataFrame columns:")
print(runs_df.columns.tolist())

# Filter to keep only columns containing loss types
for loss_type in ["Loss E", "Loss F", "MAE Hessian"]:
    df = runs_df.copy()
    matching_cols = [col for col in df.columns if loss_type in col]

    # Keep only loss-related columns plus name
    columns_to_keep = ["name", "training.splitsize"] + matching_cols
    df = df[columns_to_keep]

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Plot evaluation metrics
    """Plot eval-{x}-{loss_type} for x from 0 to 20, with each method as a separate line."""

    plt.figure(figsize=(12, 8))

    # Find all eval columns for this loss type
    eval_cols = [col for col in df.columns if "eval_" in col and loss_type in col]

    if not eval_cols:
        print(f"No eval columns found for {loss_type}")
        print(df.columns)
        continue

    # Extract x values from column names (eval_{x}-{loss_type})
    x_values = []
    for col in eval_cols:
        try:
            # Extract number between eval_x- and -{loss_type}
            x_val = int(col.split("eval_")[1].split("-")[0])
            x_values.append(x_val)
        except (IndexError, ValueError):
            continue

    if not x_values:
        print(f"Could not extract x values for {loss_type}")
        continue

    # Sort columns by x value
    sorted_cols = sorted(zip(x_values, eval_cols))
    x_vals = [x for x, _ in sorted_cols]
    sorted_eval_cols = [col for _, col in sorted_cols]

    # Prepare data for seaborn
    plot_data = []
    for idx, row in df.iterrows():
        splitsize = row.get("training.splitsize", f"Split {idx}")
        for i, col in enumerate(sorted_eval_cols):
            val = row[col]
            if pd.notna(val) and val is not None:
                plot_data.append(
                    {
                        "X Value": x_vals[i],
                        "Value": float(val),
                        "Split Size": splitsize,
                        "Method": row["name"],
                    }
                )

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        # Sort by split size for consistent ordering
        plot_df = plot_df.sort_values("Split Size", ascending=False)
        sns.lineplot(
            data=plot_df,
            x="X Value",
            y="Value",
            # hue='Split Size',
            hue="Method",
            marker="o",
            linewidth=2,
            markersize=4,
            palette="viridis",
        )

    plt.xlabel("X Value")
    plt.ylabel(f"{loss_type}")
    plt.title(f"{loss_type} per atom size")
    plt.legend(title="Training Atoms", loc="upper left", frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.xlim(8, 20)
    plt.tight_layout(pad=0.0)

    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_eval_{loss_type.replace(' ', '_').lower()}_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show()

    ##########################################################
    # Plot "val-{loss_type}" vs "training.splitsize"
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df, x="training.splitsize", y=f"val-{loss_type}", marker="o", linewidth=0
    )
    plt.xlabel("Max mumber of atoms during training")
    plt.ylabel(f"{loss_type}")
    plt.title(f"{loss_type} per atom size")
    # plt.legend(title='Training Atoms', loc='upper left', frameon=True, edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    filename = os.path.join(
        base_dir, f"natoms_val_{loss_type.replace(' ', '_').lower()}_plot.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show()
