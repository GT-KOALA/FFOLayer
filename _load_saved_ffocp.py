import json, argparse, numpy as np, cvxpy as cp, pathlib, sys

def build_problem(variables, equality_functions, inequality_functions, objective):
    sol = [[] for v in variables]
    sol_numpy = [[] for v in variables]
    equality_dual = [[] for c in equality_functions]
    inequality_dual = [[] for c in inequality_functions]
    ineq_slack_residual = [[] for c in inequality_functions]

    equality_constraints = [equality_function == 0 for equality_function in equality_functions]
    inequality_constraints = [inequality_function <= 0 for inequality_function in inequality_functions]
    # print('equality_constraints', equality_constraints)
    # print('inequality_constraints', inequality_constraints)

    problem = cp.Problem(
        cp.Minimize(objective),
        constraints=equality_constraints + inequality_constraints
    )
    return problem, sol, sol_numpy, equality_dual, inequality_dual, ineq_slack_residual

def kkt_residuals(eq_vals, ineq_vals, masks):
    import numpy as np
    eq_norm = sum(np.linalg.norm(v) for v in eq_vals)
    act_norm = 0.0
    inact_violation = 0.0
    for v, m in zip(ineq_vals, masks):
        act_norm += np.linalg.norm(m * v)
        if m.shape != v.shape:
            m = np.ones_like(v) * float(m)
        inact_violation = max(inact_violation, float(np.maximum(v[m==0], 0).max()) if np.any(m==0) else 0.0)
    return eq_norm, act_norm, inact_violation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="directory like ./cvxpy_logs/ffocp_eq_b0")
    ap.add_argument("--assume_y_old_is_new", action="store_true",
                    help="if your dump writes new solution to y_old_*, turn this switch on")
    args = ap.parse_args()

    bundle = pathlib.Path(args.dump_dir)
    meta = json.loads((bundle / "meta.json").read_text())
    arrs = np.load(bundle / "issue_min.npz", allow_pickle=True)

    alpha = float(meta["alpha"])
    dual_cutoff = float(meta["dual_cutoff"])
    solver_used = meta.get("solver_used", "GUROBI")

    objective, eq_funcs, ineq_funcs, param_order, variables = build_problem()

    for k, p in enumerate(param_order):
        p.value = arrs[f"param_{k}"]

    dvar_params = [cp.Parameter(shape=v.shape) for v in variables]
    ineq_dual_params = [cp.Parameter(shape=f.shape) for f in ineq_funcs]
    eq_dual_params   = [cp.Parameter(shape=f.shape) for f in eq_funcs]

    vars_dvars_product = cp.sum([cp.sum(cp.multiply(dp, v))
                                 for dp, v in zip(dvar_params, variables)])
    ineq_dual_product = cp.sum([cp.sum(cp.multiply(lm, f))
                                for lm, f in zip(ineq_dual_params, ineq_funcs)])
    new_objective = (1.0 / alpha) * vars_dvars_product + objective + ineq_dual_product

    active_mask_params = [cp.Parameter(shape=f.shape) for f in ineq_funcs]
    active_equalities = [cp.multiply(m, f) == 0 for m, f in zip(active_mask_params, ineq_funcs)]

    # original equality constraints
    eq_constraints = [f == 0 for f in eq_funcs]

    prob = cp.Problem(cp.Minimize(new_objective), constraints=eq_constraints + active_equalities)

    have_dvars = any(n.startswith("dvars_") for n in arrs.files)
    for j, v in enumerate(variables):
        if have_dvars and f"dvars_{j}" in arrs:
            dvar_params[j].value = arrs[f"dvars_{j}"]
        else:
            dvar_params[j].value = np.zeros(v.shape)

    for l, f in enumerate(eq_funcs):
        key = f"dual_eq_old_{l}"
        if key in arrs:
            eq_dual_params[l].value = arrs[key]
        else:
            eq_dual_params[l].value = np.zeros(f.shape)

    for m, f in enumerate(ineq_funcs):
        ineq_dual_params[m].value = arrs[f"dual_ineq_old_{m}"]
        active_mask_params[m].value = arrs[f"active_mask_used_{m}"]

    if solver_used.upper() == "GUROBI":
        prob.solve(solver=cp.GUROBI, Threads=os.cpu_count(), OutputFlag=0)
    else:
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False)

    print(f"[status] {prob.status}")

    eq_vals = [f.value for f in eq_funcs]
    ineq_vals = [f.value for f in ineq_funcs]
    masks = [active_mask_params[m].value for m in range(len(ineq_funcs))]
    eq_r, act_r, inact_violate = kkt_residuals(eq_vals, ineq_vals, masks)
    print(f"[residuals] eq_norm={eq_r:.3e}, active_eq_norm={act_r:.3e}, inactive_violation={inact_violate:.3e}")

    for j, v in enumerate(variables):
        sol_now = v.value
        key = f"y_old_{j}"
        if key in arrs:
            sol_dump = arrs[key]
            diff = float(np.linalg.norm(sol_now - sol_dump))
            print(f"[var {j}] ||y_now - y_dump|| = {diff:.6e}")
        else:
            print(f"[var {j}] (no y_old_{j} in dump)")

if __name__ == "__main__":
    import os
    main()
