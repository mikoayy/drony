"""Podgląd i ocena polityki w TubeAviary.

    # podgląd wytrenowanego modelu w GUI
    python play_tube.py --model results/tube-.../best_model.zip

    # ocena bez GUI: skuteczność na 100 epizodach
    python play_tube.py --model results/tube-.../best_model.zip --episodes 100 --gui false

    # bez modelu: ręczny sterownik (sprawdza, czy zadanie jest wykonalne)
    python play_tube.py --episodes 20 --gui false
"""
import time
import argparse
from collections import Counter

import numpy as np

from gym_pybullet_drones.envs.TubeAviary import TubeAviary
from gym_pybullet_drones.utils.enums import Physics
from gym_pybullet_drones.utils.utils import sync, str2bool


LANE_MARGIN = 0.03


def scripted_policy(env):
    """Ręczny sterownik (punkt odniesienia): każdy dron trzyma się prawej strony tuby
    (y' = -lane w swoim układzie) i leci prosto; drony mijają się obok siebie.
    Też zdecentralizowany: dron używa tylko własnej pozycji."""
    lane = max(0.0, min(env.TUBE_R - env.DRONE_R - LANE_MARGIN, 0.08))
    act = np.zeros((2, 3))
    for i in range(2):
        u, y, z = env.toLocal(i, env.pos[i], is_pos=True)
        err = np.array([0.0, -lane - y, -z])
        aligned = np.linalg.norm(err) < 0.03
        if u > env.U_GOAL:
            v = np.array([env.U_GOAL - u, err[1], err[2]])
        else:
            v = np.array([1.0 if (aligned or u > -0.15) else 0.0, 0, 0]) + 4 * err
        act[i] = v / max(1.0, np.linalg.norm(v))   # |akcja| <= 1, czyli prędkość <= speed_limit
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--gui", type=str2bool, default=True)
    ap.add_argument("--downwash", action="store_true")
    ap.add_argument("--sensing_range", type=float, default=float("inf"))
    ap.add_argument("--obs_noise", type=float, default=0.0)
    ap.add_argument("--lane_margin", type=float, default=0.03, help="zapas od ścianki dla ręcznego sterownika [m]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global LANE_MARGIN
    LANE_MARGIN = args.lane_margin
    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    env = TubeAviary(tube_radius=args.radius, gui=args.gui,
                     sensing_range=args.sensing_range, obs_noise=args.obs_noise,
                     physics=Physics.PYB_DW if args.downwash else Physics.PYB)
    outcomes, min_dists, returns = Counter(), [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        start, ret, k, min_d = time.time(), 0.0, 0, np.inf
        while True:
            if model:   # każdy dron decyduje sam, tylko na podstawie własnej obserwacji
                act = np.stack([model.predict(obs[i], deterministic=True)[0] for i in range(2)])
            else:
                act = scripted_policy(env)
            obs, r, term, trunc, info = env.step(act)
            ret += r
            min_d = min(min_d, info["min_dist"])
            if args.gui:
                sync(k, start, env.CTRL_TIMESTEP)
            k += 1
            if term or trunc:
                break
        if info["success"]:
            res = "sukces"
        elif info["collisions"]:
            res = ("minięcie poza tubą" if "pass_outside_tube" in info["collisions"]
                   else "kolizja: " + ",".join(info["collisions"]))
        elif info["out_of_bounds"]:
            res = "poza obszarem"
        else:
            res = "koniec czasu"
        outcomes["sukces" if info["success"] else res.split(":")[0]] += 1
        min_dists.append(min_d)
        returns.append(ret)
        cx = "—" if info["cross_x"] is None else f"{info['cross_x']:.2f}"
        print(f"epizod {ep:3d}: {res:30s} nagroda {ret:7.1f}  czas {k / env.CTRL_FREQ:5.1f}s  "
              f"min. odl. dronów {min_d:.3f} m  minięcie w x={cx}")
    env.close()
    n = args.episodes
    print(f"\nSkuteczność: {outcomes['sukces']}/{n} = {100 * outcomes['sukces'] / n:.0f}%   {dict(outcomes)}")
    print(f"Średnia nagroda {np.mean(returns):.1f}, najmniejsza odległość dronów {np.min(min_dists):.3f} m")


if __name__ == "__main__":
    main()
