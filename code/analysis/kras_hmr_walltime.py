"""
Wall-clock time per trajectory piece, from the NAMD logs.

Run from KRAS_unbound_basarim/ (paths below point into KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/):

    python kras_hmr_walltime.py

For each replicate it reads the chained step5_production*.log files and reports,
per piece, the steps run, the simulated time covered, and the wall-clock time
elapsed, as d:hh:mm. Totals per replicate and overall are printed at the end.

Wall time is taken from NAMD's own Wall: accumulator in the last TIMING record
of each log, which counts from the start of that run. If a log has no TIMING
records the piece is reported as incomplete.

Nothing is written to the replicate directories.
"""

import os
import re
import glob

# ---- settings ---------------------------------------------------------------
DIRS = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3": "rep1", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_2": "rep2", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_3": "rep3"}
LOG_GLOB = "step5_production*.log"
TIMESTEP_FS = 4.0        # HMR production timestep

# TIMING: 60000  CPU: 50.53, 0.000828/step  Wall: 50.19, 0.000833/step, ...
RE_TIMING = re.compile(
    r"^TIMING:\s+(\d+)\s+CPU:\s+([\d.eE+-]+),.*?Wall:\s+([\d.eE+-]+),", re.M)
# WallClock: 1234.5  CPUTime: ...  Memory: ...   (printed at clean exit)
RE_WALLCLOCK = re.compile(r"^WallClock:\s+([\d.eE+-]+)", re.M)


def order(path):
    m = re.search(r"_(\d+)\.log$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def dhm(seconds):
    """Seconds -> d:hh:mm."""
    if seconds is None:
        return "     --"
    m_total = int(round(seconds / 60.0))
    d, rem = divmod(m_total, 1440)
    h, m = divmod(rem, 60)
    return f"{d}:{h:02d}:{m:02d}"


def read_piece(path):
    text = open(path, errors="replace").read()
    hits = RE_TIMING.findall(text)
    wc = RE_WALLCLOCK.findall(text)

    steps = int(hits[-1][0]) if hits else None
    # prefer the final WallClock line (exact, printed on exit); otherwise the
    # last TIMING accumulator, which is a lower bound on the run's wall time
    if wc:
        wall = float(wc[-1])
        exact = True
    elif hits:
        wall = float(hits[-1][2])
        exact = False
    else:
        wall, exact = None, False

    return {"name": os.path.basename(path), "steps": steps,
            "wall": wall, "exact": exact, "records": len(hits)}


if __name__ == "__main__":
    grand_wall, grand_steps = 0.0, 0

    for d, lab in DIRS.items():
        files = sorted(glob.glob(os.path.join(d, LOG_GLOB)), key=order)
        if not files:
            print(f"{lab} ({d}): no logs matched {LOG_GLOB}\n")
            continue

        print(f"=== {lab}  ({d})")
        print(f"{'piece':<28}{'steps':>12}{'sim (ns)':>11}"
              f"{'wall d:hh:mm':>14}{'ns/day':>9}")

        rep_wall, rep_steps = 0.0, 0
        for f in files:
            p = read_piece(f)
            if p["steps"] is None:
                print(f"{p['name']:<28}{'--':>12}{'--':>11}"
                      f"{'--':>14}{'--':>9}   no timing records")
                continue

            sim_ns = p["steps"] * TIMESTEP_FS / 1e6
            rate = (sim_ns / (p["wall"] / 86400.0)
                    if p["wall"] and p["wall"] > 0 else float("nan"))
            flag = "" if p["exact"] else "  *"
            print(f"{p['name']:<28}{p['steps']:>12,}{sim_ns:>11.1f}"
                  f"{dhm(p['wall']):>14}{rate:>9.1f}{flag}")

            rep_wall += p["wall"] or 0.0
            rep_steps += p["steps"]

        rep_ns = rep_steps * TIMESTEP_FS / 1e6
        rep_rate = rep_ns / (rep_wall / 86400.0) if rep_wall else float("nan")
        print(f"{'TOTAL':<28}{rep_steps:>12,}{rep_ns:>11.1f}"
              f"{dhm(rep_wall):>14}{rep_rate:>9.1f}")
        print()

        grand_wall += rep_wall
        grand_steps += rep_steps

    if grand_steps:
        g_ns = grand_steps * TIMESTEP_FS / 1e6
        print("=== all replicates")
        print(f"  simulated {g_ns / 1000:.2f} \u00b5s over {dhm(grand_wall)} "
              f"of GPU wall time")
        print(f"  effective {g_ns / (grand_wall / 86400.0):.1f} ns/day "
              f"including restart overhead")
        print("\n  * wall time taken from the last TIMING record "
              "(no WallClock line);\n    this is a lower bound on the piece's "
              "true wall time.")
