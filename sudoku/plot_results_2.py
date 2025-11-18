import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 1
BASE_DIR = f"../sudoku_results_{batch_size}"
METHODS = [
    # "qpth",
    # "ffoqp_eq_schur",
    "ffocp_eq_SCS",
    "ffocp_eq_OSQP"
]

def load_results(base_dir=BASE_DIR, methods=METHODS):
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = pd.read_csv(fp)
            df["method"] = m

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

df = load_results()
df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

# for c in ["epoch","test_ts_loss","test_df_loss","forward_time","backward_time"]:
#     if c not in df.columns:
#         df[c] = np.nan
    
def plot_time_vs_method(df):
    df_avg_method = df.groupby('method')[['forward_time', 'backward_time']].mean().reset_index()

    # Convert wide → long format so Seaborn can handle grouped bars
    df_long = df_avg_method.melt(id_vars='method', 
                                value_vars=['forward_time', 'backward_time'],
                                var_name='Metrics', 
                                value_name='Time')

    plt.figure(figsize=(8,5))
    sns.barplot(data=df_long, x='method', y='Time', hue='Metrics')
    plt.ylabel("Time")
    plt.title("Forward and Backward Time")
    # plt.legend(title="Metrics")
    plt.savefig(f"{BASE_DIR}/time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_time_vs_epoch(df):
    df_avg_epoch = df.groupby(['method', 'epoch'])[['forward_time', 'backward_time']].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x='epoch', y='forward_time', hue='method', marker='o', dashes=False)
    plt.ylabel("Forward Time")
    plt.title("Forward Time vs Epoch")
    plt.savefig(f"{BASE_DIR}/forward_time_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Backward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x='epoch', y='backward_time', hue='method', marker='o', dashes=False)
    plt.ylabel("Backward Time")
    plt.title("Backward Time vs Epoch")
    plt.savefig(f"{BASE_DIR}/backward_time_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_total_time_vs_method(df):
    # Group by method, average over epochs and seeds
    df_avg_method = df.groupby('method')[['forward_time', 'backward_time']].mean().reset_index()

    # --- Stacked Bar Chart ---
    methods = df_avg_method['method']
    forward = df_avg_method['forward_time']
    backward = df_avg_method['backward_time']

    plt.figure(figsize=(8,5))
    plt.bar(methods, forward, label='forward_time', color=palette[0])
    plt.bar(methods, backward, bottom=forward, label='backward_time', color=palette[1])
    plt.ylabel("Time")
    plt.title("Total Time vs Method")
    plt.legend()
    plt.savefig(f"{BASE_DIR}/total_time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_losse_vs_epoch(df, loss_metric_name):
    df_avg_epoch = df.groupby(['method', 'epoch'])[[loss_metric_name]].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x='epoch', y=loss_metric_name, hue='method', marker='o', dashes=False)
    plt.ylabel("loss")
    plt.title("Loss vs Epoch")
    plt.savefig(f"{BASE_DIR}/{loss_metric_name}_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()
        
    
# plot_time_vs_ydim(df)
plot_time_vs_method(df)
plot_time_vs_epoch(df)
plot_total_time_vs_method(df)
plot_losse_vs_epoch(df, "train_loss")
plot_losse_vs_epoch(df, "train_error")