"""Trening PPO: dwa drony lecą naprzeciw siebie przez okrągłą tubę (TubeAviary).

Sterowanie zdecentralizowane: każdy dron działa tylko na swojej obserwacji
i dostaje własną nagrodę (MultiAgentVecEnv). Sieć jest wspólna podczas
treningu (parameter sharing), ale w locie każdy dron wykonuje ją osobno.

Przykłady (z katalogu gym_pybullet_drones/examples):

    # domyślnie: curriculum 0.20-0.26 -> 0.17-0.23 -> 0.14-0.20 m (promień losowany co epizod);
    # awans na kolejny poziom, gdy skuteczność na treningu >= 70%
    python learn_tube.py --steps 1e7

    # stały promień (podaj 0, żeby wyłączyć losowanie)
    python learn_tube.py --radius 0.15 --radius_min 0 --radius_max 0

    # trudniej: downwash, mniejszy zasięg widzenia, szum pomiaru
    python learn_tube.py --downwash --sensing_range 1.0 --obs_noise 0.01

    # dotrenowanie istniejącego modelu (curriculum)
    python learn_tube.py --init_model results/tube-.../best_model.zip --radius_min 0.14 --radius_max 0.16

Postęp: tensorboard --logdir results
"""
import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from gym_pybullet_drones.envs.TubeAviary import TubeAviary
from gym_pybullet_drones.utils.enums import Physics
from gym_pybullet_drones.utils.ma_vec_env import MultiAgentVecEnv


class CurriculumCallback(BaseCallback):
    """Zwęża tubę, gdy skuteczność na treningu przekroczy próg.

    Poziomy to zakresy losowania promienia; ostatni to zakres docelowy.
    Po awansie zapisuje model danego poziomu.
    """

    def __init__(self, levels, envs, run_dir, threshold=0.7, min_episodes=200):
        super().__init__()
        self.levels, self.envs, self.run_dir = levels, envs, run_dir
        self.threshold, self.min_episodes = threshold, min_episodes
        self.level = 0

    def _apply(self):
        lo, hi = self.levels[self.level]
        for env in self.envs:
            env.env_method("set_radius_range", lo, hi)
        print(f"[CURRICULUM] poziom {self.level}: promień tuby {lo:.2f}-{hi:.2f} m", flush=True)

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        buf = self.model.ep_success_buffer
        self.logger.record("curriculum/level", self.level)
        if self.level < len(self.levels) - 1 and len(buf) >= min(self.min_episodes, buf.maxlen):
            rate = sum(buf) / len(buf)
            if rate >= self.threshold:
                self.model.save(f"{self.run_dir}/level{self.level}_model.zip")
                self.level += 1
                buf.clear()
                self._apply()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.15, help="stały promień tuby [m] (gdy nie podano --radius_min/max)")
    ap.add_argument("--radius_min", type=float, default=0.14, help="losowy promień: dolna granica")
    ap.add_argument("--radius_max", type=float, default=0.20, help="losowy promień: górna granica")
    ap.add_argument("--steps", type=float, default=5e6, help="liczba kroków treningu")
    ap.add_argument("--n_envs", type=int, default=8, help="liczba równoległych symulacji")
    ap.add_argument("--sensing_range", type=float, default=float("inf"), help="zasięg widzenia drugiego drona [m] (domyślnie bez limitu)")
    ap.add_argument("--obs_noise", type=float, default=0.0, help="szum pomiaru pozycji drugiego drona [m]")
    ap.add_argument("--downwash", action="store_true", help="fizyka z downwashem (Physics.PYB_DW)")
    ap.add_argument("--curriculum_start", type=float, default=0.06,
                    help="o ile szersza jest tuba na pierwszym poziomie curriculum [m]")
    ap.add_argument("--curriculum_step", type=float, default=0.03, help="zwężenie tuby na każdym poziomie [m]")
    ap.add_argument("--no_curriculum", action="store_true",
                    help="bez curriculum: od razu zakres --radius_min/--radius_max")
    ap.add_argument("--init_model", type=str, default=None, help="model do dotrenowania (curriculum)")
    ap.add_argument("--output", type=str, default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    radius = (args.radius_min, args.radius_max) if args.radius_min and args.radius_max else args.radius
    env_kwargs = dict(tube_radius=radius, sensing_range=args.sensing_range, obs_noise=args.obs_noise,
                      physics=Physics.PYB_DW if args.downwash else Physics.PYB)

    run_dir = os.path.join(args.output, "tube-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_env = MultiAgentVecEnv(make_vec_env(TubeAviary, n_envs=args.n_envs, seed=args.seed,
                                              env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv))
    eval_env = MultiAgentVecEnv(make_vec_env(TubeAviary, n_envs=1, seed=args.seed + 1000,
                                             env_kwargs=env_kwargs))
    print("[INFO] agentów:", train_env.num_envs)
    print("[INFO] action space (1 dron):", train_env.action_space)
    print("[INFO] observation space (1 dron):", train_env.observation_space)

    if args.init_model:
        model = PPO.load(args.init_model, env=train_env, tensorboard_log=args.output)
    else:
        model = PPO("MlpPolicy", train_env,
                    n_steps=1024, batch_size=2048, n_epochs=10,
                    learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
                    tensorboard_log=args.output, seed=args.seed, verbose=1, device="cpu")

    eval_cb = EvalCallback(eval_env, n_eval_episodes=10,
                           eval_freq=max(20_000 // train_env.num_envs, 1),
                           best_model_save_path=run_dir, log_path=run_dir,
                           deterministic=True, verbose=1)
    callbacks = [eval_cb]
    if not args.no_curriculum and isinstance(radius, tuple):
        lo, hi = radius
        n = int(round(args.curriculum_start / args.curriculum_step))
        levels = [(lo + k * args.curriculum_step, hi + k * args.curriculum_step) for k in range(n, -1, -1)]
        callbacks.append(CurriculumCallback(levels, [train_env, eval_env], run_dir))
    model.learn(total_timesteps=int(args.steps), callback=callbacks,
                tb_log_name=os.path.basename(run_dir), reset_num_timesteps=args.init_model is None)
    model.save(os.path.join(run_dir, "final_model.zip"))
    print("[INFO] zapisano w", run_dir)


if __name__ == "__main__":
    main()
