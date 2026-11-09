#!/usr/bin/env python3
"""Load each generated model alone in an empty world under the MuJoCo plugin
(headless server), run a few seconds and check it loads cleanly and stays put.

    python3 components/simulator/assets_pipeline/load_test.py [--models-dir assets/generated] [--seconds 5] \
        [--snapshot] NAME [NAME ...]

Run inside the container with the Gazebo workspace sourced (make_replicacad.sh
does this). Pass: gz exits 0, no [Err] lines or MuJoCo warnings, no NaN, every
link found, peak link speed < 5 m/s, every link within 5 cm of its start, live
joints within their limits (+-0.05). --snapshot adds a camera (needs a GPU for
EGL rendering) and keeps one PNG per model for a visual check.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SPEED, MAX_DRIFT, LIMIT_TOL = 5.0, 0.05, 0.05


def camera_sdf(aabb, out_dir):
    lo, hi = aabb
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    size = max(b - a for a, b in zip(lo, hi))
    d = [2.0 * size + 0.3, 1.2 * size + 0.2, 0.9 * size + 0.1]
    pos = [c[i] + d[i] for i in range(3)]
    v = [-x for x in d]
    yaw = math.atan2(v[1], v[0])
    pitch = -math.atan2(v[2], math.hypot(v[0], v[1]))
    return f"""
    <model name="snapshot_cam">
      <static>true</static>
      <pose>{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} 0 {pitch:.4f} {yaw:.4f}</pose>
      <link name="link">
        <sensor name="cam" type="camera">
          <update_rate>1</update_rate>
          <topic>snapshot</topic>
          <camera>
            <horizontal_fov>0.9</horizontal_fov>
            <image><width>960</width><height>720</height></image>
            <clip><near>0.05</near><far>50</far></clip>
            <save enabled="true"><path>{out_dir}</path></save>
          </camera>
        </sensor>
      </link>
    </model>"""


def world_sdf(name, fixed, snapshot_xml):
    z = 0.0 if fixed else 0.002
    sensors = ""
    if snapshot_xml:
        sensors = """
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""
    return f"""<?xml version="1.0"?>
<sdf version="1.11">
  <world name="load_test">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics">
      <engine><filename>gz-physics-mujoco-plugin</filename></engine>
    </plugin>
    <plugin filename="gz-sim-python-system-loader-system"
            name="gz::sim::systems::PythonSystemLoader">
      <module_name>load_check_system</module_name>
    </plugin>{sensors}
    <gravity>0 0 -9.81</gravity>
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.8 0.85 0.9 1</background></scene>
    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.5 0.3 -0.8</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual>
      </link>
    </model>
    <include>
      <uri>model://{name}</uri>
      <pose>0 0 {z} 0 0 0</pose>
    </include>{snapshot_xml}
  </world>
</sdf>
"""


def test_model(name, models_dir, seconds, snapshot, out_root):
    mdir = os.path.join(models_dir, name)
    with open(os.path.join(mdir, "manifest.json")) as f:
        manifest = json.load(f)
    sdf_links = [l.get("name") for l in ET.parse(os.path.join(mdir, "model.sdf")).iter("link")]
    out = os.path.join(out_root, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    snap_dir = os.path.abspath(os.path.join(out, "snapshot"))
    cam = camera_sdf(manifest["aabb_model_frame"], snap_dir) if snapshot else ""
    world = os.path.abspath(os.path.join(out, "world.sdf"))
    with open(world, "w") as f:
        f.write(world_sdf(name, manifest["fix_to_world"], cam))
    result_json = os.path.abspath(os.path.join(out, "check.json"))
    env = dict(os.environ)
    env.update({
        "GZ_SIM_RESOURCE_PATH": os.path.abspath(models_dir),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": HERE,
        "LOAD_CHECK_OUT": result_json,
        "GZ_PARTITION": f"load_test_{name}_{os.getpid()}",
        "GZ_MUJOCO_NOSLIP_ITERATIONS": env.get("GZ_MUJOCO_NOSLIP_ITERATIONS", "10"),
    })
    cmd = ["gz", "sim", "-s", "-r", "-v", "3", "--iterations", str(int(seconds * 1000)), world]
    if snapshot:
        cmd.insert(3, "--headless-rendering")
    log_path = os.path.join(out, "gz_sim.log")
    with open(log_path, "w") as log:
        sim = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        sub = None
        if snapshot:
            # gz-sensors only renders (and saves) a camera that has a subscriber
            sub = subprocess.Popen(["gz", "topic", "-e", "-t", "/snapshot", "-n", "2"], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            rc = sim.wait(timeout=90 + 3 * seconds)
        except subprocess.TimeoutExpired:
            sim.kill()
            sim.wait()
            rc = "timeout"
        if sub is not None and sub.poll() is None:
            sub.kill()
            sub.wait()
    problems = []
    if rc != 0:
        problems.append(f"gz exit {rc}")
    with open(log_path, errors="replace") as f:
        lines = f.read().splitlines()
    errs = [l for l in lines if "[Err]" in l or "Nan, Inf or huge" in l]
    problems += [f"log: {l.strip()[:160]}" for l in errs[:5]]
    mj_log = os.path.join(out, "MUJOCO_LOG.TXT")
    if os.path.exists(mj_log):
        with open(mj_log) as f:
            problems.append("MUJOCO_LOG: " + f.read().strip().splitlines()[-1][:160])
    summary = {"name": name, "rc": rc, "warnings": sum("[Wrn]" in l for l in lines)}
    if not os.path.exists(result_json):
        problems.append("no check.json (system did not run)")
    else:
        with open(result_json) as f:
            data = json.load(f)
        summary["sim_time"] = data["t"]
        if data["t"] < seconds - 0.6:
            problems.append(f"sim stopped at t={data['t']:.2f}")
        if data["nan"]:
            problems.append("NaN pose/velocity")
        d = data["models"].get(name)
        if d is None:
            problems.append("model not found in sim")
        else:
            missing = set(sdf_links) - set(d["links"])
            if missing:
                problems.append(f"links missing: {sorted(missing)}")
            vmax = max((r["vmax"] for r in d["links"].values()), default=0.0)
            drift = max((math.dist(r["p0"], r["p"]) for r in d["links"].values()), default=0.0)
            summary.update(vmax=round(vmax, 4), drift_m=round(drift, 5))
            if vmax > MAX_SPEED:
                problems.append(f"peak link speed {vmax:.2f} m/s")
            if drift > MAX_DRIFT:
                problems.append(f"link moved {drift:.3f} m")
            summary["joints"] = {}
            for jn, jm in manifest["joints"].items():
                if jm["mode"] != "live":
                    continue
                r = d["joints"].get(jn)
                if r is None:
                    problems.append(f"joint {jn} not found")
                    continue
                summary["joints"][jn] = [round(r["qmin"], 5), round(r["qmax"], 5)]
                lo = jm["lower"] if jm["lower"] is not None else -1e9
                hi = jm["upper"] if jm["upper"] is not None else 1e9
                if r["qmin"] < lo - LIMIT_TOL or r["qmax"] > hi + LIMIT_TOL:
                    problems.append(f"joint {jn} left its limits: [{r['qmin']:.3f}, {r['qmax']:.3f}]")
    if snapshot:
        pngs = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []
        if pngs:
            keep = os.path.join(out, "snapshot.png")
            shutil.copyfile(os.path.join(snap_dir, pngs[-1]), keep)
            shutil.rmtree(snap_dir)
            summary["snapshot"] = keep
        else:
            problems.append("no snapshot image")
    summary["problems"] = problems
    summary["pass"] = not problems
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--models-dir", default="assets/generated")
    ap.add_argument("--out-dir", default=None, help="default: <models-dir>/_load_test")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()
    out_root = a.out_dir or os.path.join(a.models_dir, "_load_test")
    os.makedirs(out_root, exist_ok=True)
    results = []
    for n in a.names:
        r = test_model(n, a.models_dir, a.seconds, a.snapshot, out_root)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"LOAD_TEST {status} {n}: t={r.get('sim_time')} vmax={r.get('vmax')} m/s "
              f"drift={r.get('drift_m')} m joints={r.get('joints')} wrn={r['warnings']}"
              + ("" if r["pass"] else f" problems={r['problems']}"), flush=True)
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    ok = all(r["pass"] for r in results)
    print(f"LOAD_TEST {'ALL PASS' if ok else 'FAILURES'} ({sum(r['pass'] for r in results)}/{len(results)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
