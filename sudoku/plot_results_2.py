import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 8

BASE_DIR = f"../sudoku_results_{batch_size}"
METHODS = [
    "cvxpylayer",
    "qpth",
    "lpgd",
    "ffoqp_eq",
    "ffocp_eq",
]
METHODS_LEGEND = {
    "cvxpylayer": "CvxpyLayer",
    "qpth": "qpth",
    "lpgd": "LPGD",
    "ffoqp_eq": "FFOQP",
    "ffocp_eq": "FFOCP",
}

METHODS_STEPS = [method+"_steps" for method in METHODS]

method_order = [METHODS_LEGEND[m] for m in METHODS]

markers = ["o", "s", "D", "^", "v"]
markers_dict = {method: markers[i] for i, method in enumerate(method_order)}

LINEWIDTH = 1.5

def load_results(base_dir=BASE_DIR, methods=METHODS):
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)
            m = m.removesuffix("_steps")
            df["method"] = METHODS_LEGEND[m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            #df["eps"]  = grab(r"eps([0-9eE\.\-]+)", float)
            dfs.append(df)
        
        print("method: ", m)
        # print(df)
        
    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)

    
def plot_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_method = df.groupby('method')[time_names].mean().reset_index()

    # Convert wide → long format so Seaborn can handle grouped bars
    df_long = df_avg_method.melt(id_vars='method', 
                                value_vars=time_names,
                                var_name='Metrics', 
                                value_name='Time')

    plt.figure(figsize=(8,5))
    sns.barplot(data=df_long, x='method', y='Time', hue='Metrics')
    plt.ylabel("Time")
    plt.title("Forward and Backward Time")
    plt.savefig(f"{plot_path}/{plot_name_tag}_time_vs_method.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_epoch = df.groupby(['method', iteration_name])[time_names].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[0], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Forward Time")
    plt.title(f"Forward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_forward_time_vs_{iteration_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Backward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[1], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Backward Time")
    plt.title(f"Backward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_backward_time_vs_{iteration_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag=""):
    # Group by method, average over epochs and seeds
    df_avg_method = df.groupby('method')[time_names].mean().reset_index()

    # --- Stacked Bar Chart ---
    methods = df_avg_method['method']
    forward = df_avg_method[time_names[0]]
    backward = df_avg_method[time_names[1]]

    plt.figure(figsize=(8,5))
    plt.bar(methods, forward, label=time_names[0], color=palette[0])
    plt.bar(methods, backward, bottom=forward, label=time_names[1], color=palette[1])
    plt.ylabel("Time")
    plt.title("Total Time vs Method")
    plt.legend()
    plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_losse_vs_epoch(df, loss_metric_name, iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag="", loss_range=None, stride=50):
    df_avg_epoch = df.groupby(['method', iteration_name])[[loss_metric_name]].mean().reset_index()
    df_avg_epoch = df_avg_epoch[df_avg_epoch[iteration_name] % stride == 0]

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    ax = sns.lineplot(data=df_avg_epoch, x=iteration_name, y=loss_metric_name, hue='method', dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("loss")
    plt.title(f"Loss vs {iteration_name}")
    
    # plt.legend(
    #     title=None,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=(df["method"].nunique()),           
    #     frameon=False
    # )
    
    if loss_range is not None:
        ax.set_ylim(loss_range)
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric_name}_vs_{iteration_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
        

if __name__=="__main__":
    
    
    df = load_results()
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    

    plot_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag="sudoku")
    # plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR)
    plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag="sudoku")
    
    
    
    
    df = load_results(methods=METHODS_STEPS)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    plot_time_vs_method(df, time_names=['iter_forward_time', 'iter_backward_time'], plot_path=BASE_DIR, plot_name_tag="sudoku_steps")
    plot_time_vs_epoch(df, time_names=['iter_forward_time', 'iter_backward_time'], iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="sudoku_steps")
    plot_total_time_vs_method(df, time_names=['iter_forward_time', 'iter_backward_time'], plot_path=BASE_DIR, plot_name_tag="sudoku_steps")

    plot_losse_vs_epoch(df, "train_loss", iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="sudoku_steps", loss_range=(0.07, 0.1))
    plot_losse_vs_epoch(df, "train_error", iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="sudoku_steps")