import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 8
BASE_DIR = f"../synthetic_results_{batch_size}"
# BASE_DIR = f"../synthetic_results_1_compare_SCS_OSQP_dim200_debug"
METHODS = [
    "cvxpylayer",
    "lpgd",
    "bpqp",
    "ffocp_eq",
    "dqp",
    "qpth",
    "ffoqp_eq",   
]
METHODS_LEGEND = {
    "cvxpylayer": "CvxpyLayer",
    "lpgd": "LPGD",
    "bpqp": "BPQP",
    "ffocp_eq": "FFOCP",
    "dqp": "dQP",
    "qpth": "qpth",
    "ffoqp_eq": "FFOQP",
}

METHODS_STEPS = [method+"_steps" for method in METHODS]

method_order = [METHODS_LEGEND[m] for m in METHODS]

markers = ["o", "s", "D", "^", "v", "x", "P", "s"]
markers_dict = {method: markers[i] for i, method in enumerate(method_order)}

LINEWIDTH = 1.5


def load_results(base_dir=BASE_DIR, methods=METHODS, methods_legend=None):
    if methods_legend is None:
        methods_legend = METHODS_LEGEND
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            m = m.removesuffix("_steps")
            if m == "ffocp_eq" and "backwardTol" not in os.path.basename(fp):
                continue
            
            df = pd.read_csv(fp)
            df["method"] = methods_legend[m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["backwardTol"] = grab(r"backwardTol([0-9eE\+\-\.]+?)(?:_|\.csv)", float)
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
    plt.savefig(f"{plot_path}/{plot_name_tag}_time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_epoch = df.groupby(['method', iteration_name])[time_names].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[0], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Forward Time")
    plt.title(f"Forward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_forward_time_vs_{iteration_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Backward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[1], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Backward Time")
    plt.title(f"Backward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_backward_time_vs_{iteration_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# def plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag=""):
#     # Group by method, average over epochs and seeds
#     df_avg_method = df.groupby('method')[time_names].mean().reset_index()

#     # --- Stacked Bar Chart ---
#     methods = df_avg_method['method']
#     forward = df_avg_method[time_names[0]]
#     backward = df_avg_method[time_names[1]]

#     plt.figure(figsize=(8,5))
#     plt.bar(methods, forward, label=time_names[0], color=palette[0])
#     plt.bar(methods, backward, bottom=forward, label=time_names[1], color=palette[1])
#     plt.ylabel("Time")
#     plt.title("Total Time vs Method")
#     plt.legend()
#     plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches='tight')
#     # plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.png", dpi=300, bbox_inches='tight')
#     plt.close()

def plot_total_time_vs_method(
    df,
    time_names=["forward_time", "backward_time"],
    plot_path=BASE_DIR,
    plot_name_tag="",
):
    df_avg_method = df.groupby("method")[time_names].mean().reindex(method_order)

    methods = df_avg_method.index.tolist()
    forward = df_avg_method[time_names[0]].to_numpy()
    backward = df_avg_method[time_names[1]].to_numpy()

    total_times = forward + backward
    max_time = np.nanmax(total_times)
    min_time = 0
    y_max = max_time * 1.05

    dashed_methods = {"CvxpyLayer", "LPGD", "BPQP", "FFOCP"}  # CP
    x = np.arange(len(methods))
    width = 0.75

    fig, ax = plt.subplots(figsize=(10, 5))
    sf = mticker.ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.get_offset_text().set_visible(False)

    for i, m in enumerate(methods):
        is_dashed = (m in dashed_methods)
        ls = (0, (4, 2)) if is_dashed else "solid"
        lw = 2.0 if is_dashed else 1.5

        ax.bar(x[i], forward[i], width=width,
               color=palette[0], edgecolor="black", linewidth=lw, linestyle=ls)
        ax.bar(x[i], backward[i], bottom=forward[i], width=width,
               color=palette[1], edgecolor="black", linewidth=lw, linestyle=ls)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Time")
    ax.set_title("Total Time vs Methods")
    ax.set_ylim(min_time, y_max)

    handles = [
        Line2D([], [], linestyle="none", label="Comp. Time"),
        Patch(facecolor=palette[0], edgecolor="black", label="Forward"),
        Patch(facecolor=palette[1], edgecolor="black", label="Backward"),
        Line2D([], [], linestyle="none", label=""),
        Line2D([], [], linestyle="none", label="Solvers"),
        Line2D([0], [0], color="black", linewidth=2, linestyle=(0, (4, 2)), label="CP methods"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="solid", label="QP methods"),
    ]

    leg = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=0.7,
        labelspacing=0.1,
    )
    leg._legend_box.align = "left"

    for h, t in zip(leg.legend_handles, leg.get_texts()):
        txt = t.get_text()
        if txt in ("Comp. Time", "Solvers"):
            t.set_weight("bold")
            if hasattr(h, "set_visible"):
                h.set_visible(False)
        if txt == "":
            if hasattr(h, "set_visible"):
                h.set_visible(False)
            t.set_color((0, 0, 0, 0))

    fig.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def plot_loss_vs_epoch(df, loss_metric_name, iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag="", loss_range=None, stride=50):
    df_avg_epoch = df.groupby(['method', iteration_name])[[loss_metric_name]].mean().reset_index()
    
    # print(df_avg_epoch)
    # epoch_0_loss = -0.00950829166918993
    # df_avg_epoch.loc[df_avg_epoch[iteration_name] == 0, loss_metric_name] = epoch_0_loss
    # df_avg_epoch = df_avg_epoch[df_avg_epoch[iteration_name] % stride == 0]

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

def plot_total_time_vs_method_by_ydim(df, plot_path, time_names=('forward_time','backward_time'), tag="syn"):
    os.makedirs(plot_path, exist_ok=True)
    ydims = sorted(df["ydim"].dropna().unique().astype(int))
    for y in ydims:
        d = df[df["ydim"] == y]
        plot_total_time_vs_method(
            d,
            time_names=list(time_names),
            plot_path=plot_path,
            plot_name_tag=f"{tag}_ydim{y}"
        )

def plot_total_time_vs_ydim_for_methods(
    df,
    methods_subset,
    markers_dict=markers_dict,
    time_names=('forward_time', 'backward_time'),
    plot_path=BASE_DIR,
    plot_name_tag="syn_methods_vs_ydim"
):
    os.makedirs(plot_path, exist_ok=True)
    df_sub = df[df["method"].isin(methods_subset)].copy()
    df_sub = df_sub.dropna(subset=["ydim"])
    df_sub["ydim"] = df_sub["ydim"].astype(int)

    df_group = (
        df_sub
        .groupby(["method", "ydim"])[list(time_names)]
        .mean()
        .reset_index()
    )

    df_group["total_time"] = df_group[time_names[0]] + df_group[time_names[1]]

    plt.figure(figsize=(8, 5))

    for idx, method in enumerate(methods_subset):
        d = df_group[df_group["method"] == method].copy()
        d = d.sort_values("ydim")
        marker = markers_dict.get(method, markers[idx % len(markers)])
        plt.plot(
            d["ydim"],
            d["total_time"],
            label=method,
            marker=marker,
            linewidth=LINEWIDTH,
        )

    plt.xlabel("y_dim")
    plt.ylabel("Total Time")
    plt.title("Total Time vs y_dim (selected methods)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{plot_path}/{plot_name_tag}_total_time_vs_ydim.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_total_loss_vs_method_by_ydim(df, plot_path):
    os.makedirs(plot_path, exist_ok=True)
    ydims = sorted(df["ydim"].dropna().unique().astype(int))
    for y in ydims:
        d = df[df["ydim"] == y]
        plot_loss_vs_epoch(
            d,
            loss_metric_name="train_df_loss",
            iteration_name="iter",
            plot_path=plot_path,
            plot_name_tag=f"syn_ydim{y}"
        )


def plot_total_time_vs_method_by_backwardTol(df, time_names=['forward_time', 'backward_time'],
                                             plot_path=BASE_DIR, plot_name_tag=""):
    """
    Plots stacked bar chart of total time vs backward tolerance for each method.
    Each method is plotted separately in a for-loop to ensure only one method at a time.
    """
    os.makedirs(plot_path, exist_ok=True)

    df = df.dropna(subset=['method', 'backwardTol']).copy()

    # Get unique methods
    methods = df['method'].unique()
    assert(len(methods)==1)

    for method in methods:
        d = df[df['method'] == method].copy()

        # Remove duplicate backwardTol per method (take mean)
        d = d.groupby('backwardTol')[time_names].mean().reset_index()

        plt.figure(figsize=(8,5))
        plt.bar(d['backwardTol'].astype(str), d[time_names[0]], label=time_names[0], color=palette[0])
        plt.bar(d['backwardTol'].astype(str), d[time_names[1]], bottom=d[time_names[0]],
                label=time_names[1], color=palette[1])
        
        plt.xlabel("backwardTol")
        plt.ylabel("Time")
        plt.title(f"Total Time vs backwardTol ({method})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_backwardTol_{method}.pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_loss_vs_method_by_backwardTol(df, loss_metric_name,
                                       plot_path=BASE_DIR, plot_name_tag="",
                                       loss_range=None):
    """
    Plots line plot of loss vs backwardTol for each method.
    Ensures one method at a time and no duplicate backwardTol.
    """
    os.makedirs(plot_path, exist_ok=True)

    df = df.dropna(subset=['method', 'backwardTol']).copy()
    methods = df['method'].unique()
    assert(len(methods)==1)

    for method in methods:
        d = df[df['method'] == method].copy()
        d = d.groupby('backwardTol')[[loss_metric_name]].mean().reset_index()

        plt.figure(figsize=(8,5))
        plt.plot(d['backwardTol'], d[loss_metric_name], linewidth=LINEWIDTH)
        plt.xlabel("backwardTol")
        plt.ylabel(loss_metric_name)
        plt.title(f"{loss_metric_name} vs backwardTol ({method})")
        if loss_range is not None:
            plt.ylim(loss_range)
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric_name}_vs_backwardTol_{method}.pdf",
                    dpi=300, bbox_inches='tight')
        plt.close()
        
def plot_loss_vs_epoch_method_tol(df, loss_metric='train_df_loss', iteration='epoch',
                                  plot_path=BASE_DIR, plot_name_tag=""):
    """
    Simple: plot loss vs epoch, each line = one (method, backwardTol) tuple.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(plot_path, exist_ok=True)
    df = df.dropna(subset=['method', 'backwardTol', iteration, loss_metric])

    plt.figure(figsize=(8,5))

    for (method, tol), d in df.groupby(['method', 'backwardTol']):
        d = d.sort_values(iteration)
        plt.plot(d[iteration], d[loss_metric], label=f"{method}-tol{tol:.0e}")

    plt.xlabel(iteration)
    plt.ylabel(loss_metric)
    plt.title(f"{loss_metric} vs {iteration}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric}_vs_{iteration}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


        

if __name__=="__main__":
    
    df = load_results()
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    print("loaded df")

    # plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag="syn")
    plot_total_time_vs_method_by_backwardTol(df, time_names=['forward_time', 'backward_time'],
                                             plot_path=BASE_DIR, plot_name_tag="syn")
   
   #########################################
    df = load_results(methods=METHODS_STEPS)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    print("loaded df steps")

    # plot_total_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_solve")
    # plot_total_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_setup")
    
    # plot_loss_vs_epoch(df, "train_df_loss", iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="syn_steps")
    plot_loss_vs_epoch_method_tol(df, loss_metric='train_df_loss', iteration='iter',
                                  plot_path=BASE_DIR, plot_name_tag="syn_steps")
    