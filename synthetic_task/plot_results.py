import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 8
BASE_DIR = f"../synthetic_results_{batch_size}"
# BASE_DIR = f"../synthetic_results_1_compare_SCS_OSQP_dim200_debug"
METHODS = [
    "cvxpylayer",
    "qpth",
    "lpgd",
    "ffoqp_eq_schur",
    "ffocp_eq",
]
METHODS_LEGEND = {
    "cvxpylayer": "CvxpyLayer",
    "qpth": "qpth",
    "lpgd": "LPGD",
    "ffoqp_eq_schur": "FFOQP",
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
            # if m=="ffoqp_eq_schur_steps" or m=="ffoqp_eq_schur":
            #     m = "ffoqp_eq"
            new_m = m.removesuffix("_steps")
            df["method"] = METHODS_LEGEND[new_m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan
            
            seed = grab(r"_seed(\d+)", int)
            # Join with epoch 0 results to get the epoch 0 train df loss
            # epoch_0_loss = EPOCH_0_DF[(EPOCH_0_DF["method"]==m) & (EPOCH_0_DF["seed"]==seed)]["train_df_loss"].values
            # epoch_0_row = [0,0,epoch_0_loss,0,0,0,0,0,0]
            
            

            df["seed"] = grab(r"_seed(\d+)", int)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            # df["eps"]  = grab(r"eps([0-9eE\.\-]+)", float)

            dfs.append(df)

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
    plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_losse_vs_epoch(df, loss_metric_name, iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag="", loss_range=None, stride=50):
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
    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric_name}_vs_{iteration_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
        


        

if __name__=="__main__":
    # df = load_results()
    # df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    #plot_time_vs_ydim(df)
    # plot_time_vs_method(df)
    # plot_time_vs_epoch(df)
    # plot_total_time_vs_method(df)
    # plot_losse_vs_epoch(df)

    # plot_opt_time_vs_epoch(df)
    # plot_opt_time_vs_epoch_per_method(df)
    
    
    df = load_results()
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    print("loaded df")

    # plot_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR)
    # plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR)
    plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag="syn")
    # plot_losse_vs_epoch(df, "train_df_loss", iteration_name='epoch', plot_path=BASE_DIR)
    
    
    
    df = load_results(methods=METHODS_STEPS)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    print("loaded df steps")

    # plot_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="steps_solve")
    # plot_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="steps_setup")
    plot_total_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_solve")
    plot_total_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_setup")
    plot_losse_vs_epoch(df, "train_df_loss", iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="syn_steps")
    