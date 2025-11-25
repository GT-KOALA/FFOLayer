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
    "ffocp_eq",
    # "lpgd",
    "cvxpylayer",
    # "ffoqp_eq_schur",
    # "qpth"
]
METHODS_STEPS = [method+"_steps" for method in METHODS]
LINEWIDTH = 1.5

def load_results(base_dir=BASE_DIR, methods=METHODS):
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)
            if m=="ffoqp_eq_schur_steps" or m=="ffoqp_eq_schur":
                m = "ffoqp_eq"
            df["method"] = m.removesuffix("_steps")

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            # df["eps"]  = grab(r"eps([0-9eE\.\-]+)", float)

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)


# def plot_time_vs_ydim(df):
    
#     print("Columns:", df.columns.tolist())
#     print(df.head())
#     # === 2. Compute averages over epochs and seeds ===
#     df['total_time'] = df['forward_time'] + df['backward_time']
#     grouped = (
#         df.groupby(['method', 'ydim'], as_index=False)
#         .agg({'forward_time': 'mean', 'backward_time': 'mean', 'total_time': 'mean'})
#     )

#     # === 3. Plot ===
#     metrics = ['forward_time', 'backward_time', 'total_time']
#     titles = ['Forward Time', 'Backward Time', 'Total Time']

#     plt.figure(figsize=(8, 6))

#     colors = {
#         'qpth': 'tab:blue',
#         'ffoqp_eq_schur': 'tab:orange',
#         'ffocp_eq': 'tab:green'
#     }

#     for i, metric in enumerate(metrics, start=1):
#         plt.subplot(3, 1, i)
#         for method, color in colors.items():
#             subset = grouped[grouped['method'] == method].sort_values('ydim')
#             plt.plot(subset['ydim'], subset[metric], marker='o', label=method, color=color)
#         plt.title(titles[i-1])
#         plt.xlabel("ydim")
#         plt.ylabel("Average Time")
#         plt.grid(True)
#         if i == 1:
#             plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"{BASE_DIR}/average_times_vs_ydim.png", dpi=300, bbox_inches="tight")
#     plt.close()
    
# def plot_opt_time_vs_epoch(df):
#     df_avg_epoch = df.groupby(['method', 'epoch'])[['forward_opt_time', 'backward_opt_time']].mean().reset_index()

#     # --- Forward Time Figure ---
#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_avg_epoch, x='epoch', y='forward_opt_time', hue='method', marker='o', dashes=False)
#     plt.ylabel("Forward opt Time")
#     plt.title("Forward opt Time vs Epoch")
#     plt.savefig(f"{BASE_DIR}/forward_opt_time_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()

#     # --- Backward Time Figure ---
#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_avg_epoch, x='epoch', y='backward_opt_time', hue='method', marker='o', dashes=False)
#     plt.ylabel("Backward opt Time")
#     plt.title("Backward opt Time vs Epoch")
#     plt.savefig(f"{BASE_DIR}/backward_opt_time_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
# def plot_opt_time_vs_epoch_per_method(df):
#     # Melt the dataframe to long form for Seaborn
#     df_long = df.groupby(['method', 'epoch'])[['forward_opt_time', 'backward_opt_time']].mean().reset_index()
#     df_long = df_long.melt(id_vars=['method', 'epoch'], 
#                            value_vars=['forward_opt_time', 'backward_opt_time'],
#                            var_name='pass_type', value_name='opt_time')

#     method = df["method"].iloc[0]

#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_long, x='epoch', y='opt_time', hue='pass_type', 
#                  style='pass_type', markers=True, dashes=False)

#     plt.xlabel("Epoch")
#     plt.ylabel("Optimization Time (s)")
#     plt.title(f"Optimization Time vs Epoch ({method})")
#     plt.grid(True)
#     plt.legend(title="Pass Type")
#     plt.savefig(f"{BASE_DIR}/{method}_opt_time_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
# def plot_time_vs_method(df):
#     df_avg_method = df.groupby('method')[['forward_time', 'backward_time']].mean().reset_index()

#     # Convert wide → long format so Seaborn can handle grouped bars
#     df_long = df_avg_method.melt(id_vars='method', 
#                                 value_vars=['forward_time', 'backward_time'],
#                                 var_name='Metrics', 
#                                 value_name='Time')

#     plt.figure(figsize=(8,5))
#     sns.barplot(data=df_long, x='method', y='Time', hue='Metrics')
#     plt.ylabel("Time")
#     plt.title("Forward and Backward Time")
#     # plt.legend(title="Metrics")
#     plt.savefig(f"{BASE_DIR}/time_vs_method.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
# def plot_time_vs_epoch(df):
#     df_avg_epoch = df.groupby(['method', 'epoch'])[['forward_time', 'backward_time']].mean().reset_index()

#     # --- Forward Time Figure ---
#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_avg_epoch, x='epoch', y='forward_time', hue='method', marker='o', dashes=False)
#     plt.ylabel("Forward Time")
#     plt.title("Forward Time vs Epoch")
#     plt.savefig(f"{BASE_DIR}/forward_time_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()

#     # --- Backward Time Figure ---
#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_avg_epoch, x='epoch', y='backward_time', hue='method', marker='o', dashes=False)
#     plt.ylabel("Backward Time")
#     plt.title("Backward Time vs Epoch")
#     plt.savefig(f"{BASE_DIR}/backward_time_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
# def plot_total_time_vs_method(df):
#     # Group by method, average over epochs and seeds
#     df_avg_method = df.groupby('method')[['forward_time', 'backward_time']].mean().reset_index()

#     # --- Stacked Bar Chart ---
#     methods = df_avg_method['method']
#     forward = df_avg_method['forward_time']
#     backward = df_avg_method['backward_time']

#     plt.figure(figsize=(8,5))
#     plt.bar(methods, forward, label='forward_time', color=palette[0])
#     plt.bar(methods, backward, bottom=forward, label='backward_time', color=palette[1])
#     plt.ylabel("Time")
#     plt.title("Total Time vs Method")
#     plt.legend()
#     plt.savefig(f"{BASE_DIR}/total_time_vs_method.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
# def plot_losse_vs_epoch(df, loss_metric_name='train_df_loss'):
#     df_avg_epoch = df.groupby(['method', 'epoch'])[[loss_metric_name]].mean().reset_index()

#     # --- Forward Time Figure ---
#     plt.figure(figsize=(8,5))
#     sns.lineplot(data=df_avg_epoch, x='epoch', y=loss_metric_name, hue='method', marker='o', dashes=False)
#     plt.ylabel("loss")
#     plt.title("Loss vs Epoch")
#     plt.savefig(f"{BASE_DIR}/loss_vs_epoch.png", dpi=300, bbox_inches='tight')
#     plt.close()

    
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
    plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_losse_vs_epoch(df, loss_metric_name, iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_epoch = df.groupby(['method', iteration_name])[[loss_metric_name]].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=loss_metric_name, hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("loss")
    plt.title(f"Loss vs {iteration_name}")
    
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
    print("loaded df")

    # plot_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR)
    # plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR)
    plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR)
    # plot_losse_vs_epoch(df, "train_df_loss", iteration_name='epoch', plot_path=BASE_DIR)
    
    
    
    df = load_results(methods=METHODS_STEPS)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    print("loaded df steps")

    # plot_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="steps_solve")
    # plot_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="steps_setup")
    plot_total_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="steps_solve")
    plot_total_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="steps_setup")
    plot_losse_vs_epoch(df, "train_df_loss", iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="steps")
    