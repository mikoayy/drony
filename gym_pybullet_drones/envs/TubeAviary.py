"""Dwa drony lecą naprzeciw siebie przez wąską, okrągłą tubę i nie mogą się zderzyć.

Układ sceny (oś tuby = oś X, środek osi na wysokości Z0):

    cel D1 (x=-0.5)  start D0 (x=-0.9)   tuba x in [0, L]   start D1 (x=L+0.9)  cel D0 (x=L+0.5)
                         D0 o  ---->   ((=================))   <----  o D1

Drony MUSZĄ minąć się wewnątrz tuby: miejsce, w którym ich pozycje wzdłuż osi X
się zrównują, musi leżeć w [0, L]. Czekanie przed wlotem, aż drugi przeleci,
kończy epizod porażką.

Sterowanie zdecentralizowane: każdy dron ma własną obserwację (tylko to, co
sam "wie" i widzi) i własną nagrodę. Interfejs gymnasium zwraca je razem jako
tablice (2, ...), a `MultiAgentVecEnv` (utils/ma_vec_env.py) rozdziela je na
dwóch niezależnych agentów dla stable-baselines3.

Każdy dron widzi świat we własnym układzie: "do przodu" to zawsze +x, wlot
tuby jest w x=0, wylot w x=L. Dla drona 1 układ jest obrócony o 180° wokół
osi Z. Dzięki temu oba drony mają ten sam problem do rozwiązania.

Akcja na drona: zadana prędkość [vx, vy, vz] w zakresie [-1, 1] (razy speed_limit),
w układzie własnym drona; ciągiem silników steruje PID (DSLPIDControl).

Obserwacja na drona (24 liczby, układ własny drona):
    pozycja względem wlotu tuby (3), roll/pitch/yaw (3), prędkość (3),
    prędkość kątowa (3), pozycja i prędkość drugiego drona względem
    tego drona (6, zera gdy jest poza zasięgiem czujnika), flaga
    "widzę drugiego" (1), ostatnia akcja (3), promień tuby (1),
    pozostały czas epizodu jako ułamek 1 -> 0 (1).

Koniec czasu to porażka (kara jak za zderzenie), więc zawis przed tubą się nie opłaca.
"""
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

MIRROR = np.diag([-1.0, -1.0, 1.0])   # obrót o 180° wokół Z


class TubeAviary(BaseRLAviary):
    """Zadanie: drony z przeciwnych końców przelatują przez tubę i mijają się bez kolizji."""

    ################################################################################

    def __init__(self,
                 tube_radius=0.18,
                 tube_length: float = 1.5,
                 tube_height: float = 1.0,
                 speed_limit: float = 0.6,
                 episode_len_sec: float = 15.0,
                 randomize_start: bool = True,
                 sensing_range: float = np.inf,
                 obs_noise: float = 0.0,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 ):
        """
        Parameters
        ----------
        tube_radius : float | tuple(float, float)
            Promień wewnętrzny tuby [m]. Krotka (min, max) losuje promień
            w każdym epizodzie (curriculum / uogólnianie).
            Dron CF2X ma promień kolizji 0.06 m. Żeby drony minęły się
            obok siebie, potrzeba R > ~0.13 m; przy R = 0.18 m mijają się
            na styk (odległość środków do ~0.22 m przy limicie kolizji 0.12 m).
            Przy mniejszym R jeden dron musi poczekać przed tubą.
        tube_length : float
            Długość tuby wzdłuż osi X [m].
        tube_height : float
            Wysokość osi tuby nad ziemią [m].
        speed_limit : float
            Maksymalna zadawana prędkość [m/s].
        episode_len_sec : float
            Maksymalny czas epizodu [s].
        randomize_start : bool
            Czy losować pozycje startowe dronów.
        sensing_range : float
            Zasięg, z jakiego dron "widzi" drugiego [m] (np.inf = zawsze).
        obs_noise : float
            Odchylenie standardowe szumu pomiaru pozycji drugiego drona [m].
        physics : Physics
            Physics.PYB_DW dodaje downwash (ważny, gdy drony mijają się jeden nad drugim).
        """
        #### Parametry sceny (muszą być przed super().__init__, bo ta buduje tubę)
        self.TUBE_RADIUS_RANGE = tube_radius if isinstance(tube_radius, (tuple, list)) else (tube_radius, tube_radius)
        self.TUBE_R = float(np.mean(self.TUBE_RADIUS_RANGE))
        self.TUBE_L = tube_length
        self.Z0 = tube_height
        self.TUBE_WALL = 0.01            # grubość ścianki [m]
        self.TUBE_SEGMENTS = 36          # z ilu płytek składa się obwód tuby
        self.U_START = -0.9              # start i cel we współrzędnej "postępu" u (układ drona)
        self.U_GOAL = self.TUBE_L + 0.5
        self.EPISODE_LEN_SEC = episode_len_sec
        self.RANDOMIZE_START = randomize_start
        self.SENSING_RANGE = sensing_range
        self.OBS_NOISE = obs_noise
        self.OBS_DIM = 24
        self.tube_ids = []
        self._rng = np.random.default_rng()
        self._prev_phi = None
        self._eval = None
        self._crossed = False
        self.cross_x = None
        #### Początek układu każdego drona: wlot "jego" końca tuby
        self.FRAME_ORIGIN = np.array([[0.0, 0.0, self.Z0], [self.TUBE_L, 0.0, self.Z0]])
        self.FRAME_ROT = np.array([np.eye(3), MIRROR])

        #### Wagi nagrody
        self.W_PROGRESS = 10.0           # za postęp w stronę celu (shaping potencjałowy)
        self.W_AXIS = 1.0                # kara za odsunięcie od osi przed wlotem (w potencjale)
        self.DX_NEAR = 1.0               # drony są "blisko" wzdłuż tuby, gdy |x0 - x1| < DX_NEAR
        self.LAT_SAFE = 0.14             # wtedy w przekroju tuby powinny być rozsunięte o tyle [m]
        self.W_SEPARATION = 1.0          # waga członu rozsunięcia w potencjale
        self.R_PASS = 10.0               # dla obu w chwili udanego minięcia w tubie
        self.TIME_PENALTY = 0.05         # co krok; wiszenie kosztuje od razu, a nie dopiero na końcu
        self.R_SUCCESS = 50.0            # dla obu, gdy oba doleciały
        self.R_CRASH = -20.0             # dla winnego (ściana/ziemia/poza obszarem) lub obu (zderzenie,
                                         # minięcie poza tubą); plus koszt czasu, który jeszcze został,
                                         # żeby rozbicie się nie było "tańsze" niż dalszy lot
        self.R_TIMEOUT = -20.0           # dla obu

        super().__init__(drone_model=drone_model,
                         num_drones=2,
                         initial_xyzs=self._startPositions(randomize=False),
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL,
                         )
        self.SPEED_LIMIT = speed_limit
        self.DRONE_R = self.COLLISION_R  # 0.06 m dla CF2X

    ################################################################################
    #### Układy współrzędnych #######################################################

    def toLocal(self, i, v, is_pos=False):
        """Wektor ze świata do układu drona i (pozycje: względem jego wlotu)."""
        v = np.asarray(v, dtype=float)
        if is_pos:
            v = v - self.FRAME_ORIGIN[i]
        return self.FRAME_ROT[i] @ v

    def toWorld(self, i, v):
        """Kierunek z układu drona i do świata (MIRROR jest swoją odwrotnością)."""
        return self.FRAME_ROT[i] @ np.asarray(v, dtype=float)

    def progress(self, i):
        """u: postęp drona i wzdłuż tuby w jego układzie (0 = wlot, L = wylot)."""
        return self.toLocal(i, self.pos[i], is_pos=True)[0]

    ################################################################################
    #### Scena ######################################################################

    def _startPositions(self, randomize):
        xyz = np.zeros((2, 3))
        for i in range(2):
            local = np.array([self.U_START, 0.0, 0.0])
            if randomize:
                local += self._rng.uniform([-0.15, -0.25, -0.2], [0.15, 0.25, 0.2])
            xyz[i] = self.FRAME_ORIGIN[i] + self.FRAME_ROT[i] @ local
        return xyz

    def set_radius_range(self, lo, hi):
        """Zmienia zakres losowania promienia (używane przez curriculum); działa od następnego epizodu."""
        self.TUBE_RADIUS_RANGE = (float(lo), float(hi))

    def _timeLeft(self):
        return max(0.0, 1.0 - self.step_counter / (self.PYB_FREQ * self.EPISODE_LEN_SEC))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        lo, hi = self.TUBE_RADIUS_RANGE
        self.TUBE_R = float(self._rng.uniform(lo, hi))
        self.INIT_XYZS = self._startPositions(self.RANDOMIZE_START)
        self.INIT_RPYS = np.array([[0, 0, 0], [0, 0, np.pi]])   # drony "patrzą" w stronę tuby
        for c in self.ctrl:
            c.reset()
        self._prev_phi = None
        self._eval = None
        self._crossed = False
        self.cross_x = None
        obs, info = super().reset(seed=seed, options=options)
        self._prev_phi = self._potential()
        return obs, info

    def _addObstacles(self):
        """Buduje okrągłą tubę z płytek ułożonych po obwodzie (statyczne, masa 0)."""
        R, L, t, n = self.TUBE_R, self.TUBE_L, self.TUBE_WALL, self.TUBE_SEGMENTS
        width = 2 * (R + t) * np.tan(np.pi / n) * 1.1   # lekka zakładka, żeby nie było szpar
        half = [L / 2, t / 2, width / 2]                  # lokalnie: x wzdłuż tuby, y promieniowo
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.CLIENT)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.3, 0.6, 1.0, 0.35],
                                  physicsClientId=self.CLIENT)
        self.tube_ids = []
        for k in range(n):
            th = 2 * np.pi * k / n
            rc = R + t / 2
            self.tube_ids.append(p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[L / 2, rc * np.cos(th), self.Z0 + rc * np.sin(th)],
                baseOrientation=p.getQuaternionFromEuler([th, 0, 0]),
                physicsClientId=self.CLIENT))
        if self.GUI:
            colors = [[1, 0, 0], [0, 0, 1]]
            for i in range(2):
                goal = self.FRAME_ORIGIN[i] + self.FRAME_ROT[i] @ np.array([self.U_GOAL, 0, 0])
                p.addUserDebugPoints([goal.tolist()], [colors[i]], pointSize=10,
                                     physicsClientId=self.CLIENT)

    ################################################################################
    #### Akcje i obserwacje #########################################################

    def _actionSpace(self):
        """Akcja na drona: zadana prędkość [vx, vy, vz] w [-1, 1] (razy SPEED_LIMIT), układ drona."""
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, 3)))
        return spaces.Box(low=-np.ones((self.NUM_DRONES, 3)), high=np.ones((self.NUM_DRONES, 3)), dtype=np.float32)

    def _preprocessAction(self, action):
        """Polityka podaje prędkość w układzie drona; PID dostaje ją w układzie świata."""
        action = np.clip(np.array(action, dtype=float).reshape(self.NUM_DRONES, 3), -1, 1)
        self.action_buffer.append(action)   # w obserwacji trzymamy akcję w układzie drona
        rpm = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            v_world = self.toWorld(i, action[i]) * self.SPEED_LIMIT
            rpm[i], _, _ = self.ctrl[i].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                       cur_pos=state[0:3],
                                                       cur_quat=state[3:7],
                                                       cur_vel=state[10:13],
                                                       cur_ang_vel=state[13:16],
                                                       target_pos=state[0:3],
                                                       target_rpy=np.array([0, 0, state[9]]),
                                                       target_vel=v_world)
        return rpm

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, self.OBS_DIM), dtype=np.float32)

    def _computeObs(self):
        """Obserwacja lokalna każdego drona (wiersz i = to, co wie dron i)."""
        last_act = self.action_buffer[-1]
        obs = np.zeros((self.NUM_DRONES, self.OBS_DIM), dtype=np.float32)
        for i in range(self.NUM_DRONES):
            j = 1 - i
            rpy = self.rpy[i].copy()
            if i == 1:   # obrót o 180° wokół Z: roll i pitch zmieniają znak, yaw +pi
                rpy = np.array([-rpy[0], -rpy[1], (rpy[2] + 2 * np.pi) % (2 * np.pi) - np.pi])
            rel_p = self.pos[j] - self.pos[i]
            rel_v = self.vel[j] - self.vel[i]
            visible = np.linalg.norm(rel_p) <= self.SENSING_RANGE
            if visible:
                if self.OBS_NOISE > 0:
                    rel_p = rel_p + self._rng.normal(0, self.OBS_NOISE, 3)
                    rel_v = rel_v + self._rng.normal(0, 2 * self.OBS_NOISE, 3)
                other = np.hstack([self.toLocal(i, rel_p), self.toLocal(i, rel_v)])
            else:
                other = np.zeros(6)
            obs[i] = np.hstack([self.toLocal(i, self.pos[i], is_pos=True),
                                rpy,
                                self.toLocal(i, self.vel[i]),
                                self.toLocal(i, self.ang_v[i]),
                                other,
                                [float(visible)],
                                last_act[i],
                                [self.TUBE_R],
                                [self._timeLeft()]])
        return obs

    ################################################################################
    #### Nagroda i warunki końca ####################################################

    def _radial(self, i):
        return np.hypot(self.pos[i, 1], self.pos[i, 2] - self.Z0)

    def _potentials(self):
        """Potencjał każdego drona (shaping): postęp u minus odsunięcie od osi przed wylotem.

        Za celem potencjał maleje, więc dron, który już przeleciał, ma zostać przy celu.
        Wewnątrz tuby kara działa dopiero blisko ścianki, więc drony mogą się wyminąć.
        """
        r_ok = self.TUBE_R - self.DRONE_R - 0.02   # tyle wolno odjechać od osi bez kary
        phi = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            u = self.progress(i)
            phi[i] = u if u <= self.U_GOAL else self.U_GOAL - (u - self.U_GOAL)
            if u < self.TUBE_L:
                phi[i] -= self.W_AXIS * max(0.0, self._radial(i) - r_ok)
        #### Rozsunięcie w przekroju tuby: im bliżej siebie wzdłuż osi, tym bardziej się liczy.
        #### Jako potencjał: odsunięcie się w bok daje nagrodę od razu, jazda na wprost - karę.
        dx = abs(self.pos[0, 0] - self.pos[1, 0])
        lat = float(np.linalg.norm(self.pos[0, 1:] - self.pos[1, 1:]))
        closeness = max(0.0, 1.0 - dx / self.DX_NEAR)
        phi -= self.W_SEPARATION * max(0.0, self.LAT_SAFE - lat) / self.LAT_SAFE * closeness
        return phi

    def _potential(self):
        return self._potentials()

    def _collisions(self):
        """Zwraca listę opisów kolizji (pusta = brak)."""
        events = []
        rd = self.DRONE_R
        for i in range(self.NUM_DRONES):
            x, r = self.pos[i, 0], self._radial(i)
            # Geometria: w obszarze tuby dron musi być cały w środku
            if -rd <= x <= self.TUBE_L + rd and r + rd >= self.TUBE_R:
                events.append(f"drone{i}_wall")
            # Kontakty z fizyki (ściany, krawędź wlotu, ziemia)
            for c in p.getContactPoints(bodyA=int(self.DRONE_IDS[i]), physicsClientId=self.CLIENT):
                other = c[2]
                if other in self.tube_ids:
                    events.append(f"drone{i}_wall")
                elif other == self.PLANE_ID:
                    events.append(f"drone{i}_ground")
            if self.pos[i, 2] < 0.05:
                events.append(f"drone{i}_ground")
        # Dron-dron: przybliżenie sferą o promieniu kolizji drona
        if np.linalg.norm(self.pos[0] - self.pos[1]) < 2 * rd:
            events.append("drone_drone")
        if p.getContactPoints(bodyA=int(self.DRONE_IDS[0]), bodyB=int(self.DRONE_IDS[1]),
                              physicsClientId=self.CLIENT):
            events.append("drone_drone")
        return sorted(set(events))

    def _outOfBounds(self):
        """Lista dronów, które wyleciały poza obszar albo za mocno się przechyliły."""
        out = []
        for i in range(self.NUM_DRONES):
            x, y, z = self.pos[i]
            if (x < -2.0 or x > self.TUBE_L + 2.0 or abs(y) > 1.2
                    or abs(z - self.Z0) > 0.9 or abs(self.rpy[i, 0]) > 0.8 or abs(self.rpy[i, 1]) > 0.8):
                out.append(i)
        return out

    def _checkCrossing(self):
        """Wykrywa moment minięcia (dron 0 startuje z lewej, więc x0 < x1 aż do minięcia).

        Zwraca True, jeśli minięcie właśnie nastąpiło POZA tubą albo jeden dron wyleciał
        z tuby, zanim minął drugiego (= porażka, drugi czekał na zewnątrz).
        """
        if self._crossed:
            return False
        if self.pos[0, 0] < self.pos[1, 0]:
            if any(self.progress(i) > self.TUBE_L + self.DRONE_R for i in range(2)):
                self._crossed = True
                return True
            return False
        self._crossed = True
        self.cross_x = float(self.pos[0, 0] + self.pos[1, 0]) / 2
        return not (0.0 <= self.cross_x <= self.TUBE_L)

    def _evaluate(self):
        """Liczy wszystko raz na krok; reward/terminated/truncated/info czytają wynik."""
        if self._eval is not None and self._eval["step"] == self.step_counter:
            return self._eval
        phi = self._potentials()
        rewards = self.W_PROGRESS * (phi - self._prev_phi) if self._prev_phi is not None else np.zeros(2)
        self._prev_phi = phi

        dist = float(np.linalg.norm(self.pos[0] - self.pos[1]))
        rewards -= self.TIME_PENALTY
        was_crossed = self._crossed

        collisions = self._collisions()
        out = self._outOfBounds()
        passed_outside = self._checkCrossing()
        if passed_outside:
            collisions.append("pass_outside_tube")
        failure = bool(collisions or out)
        if self._crossed and not was_crossed and not failure:
            rewards += self.R_PASS     # minęły się w tubie bez kolizji
        #### Kary: winny dron za ścianę/ziemię/wylot, oba za zderzenie i minięcie poza tubą
        for i in range(self.NUM_DRONES):
            mine = any(e.startswith(f"drone{i}_") for e in collisions) or i in out
            shared = "drone_drone" in collisions or passed_outside
            if mine or shared:
                steps_left = self._timeLeft() * self.EPISODE_LEN_SEC * self.CTRL_FREQ
                rewards[i] += self.R_CRASH - self.TIME_PENALTY * steps_left
        success = (not failure and self._crossed
                   and all(self.progress(i) > self.U_GOAL - 0.1 for i in range(2)))
        if success:
            rewards += self.R_SUCCESS
        #### Koniec czasu = porażka obu (czas jest w obserwacji, więc to stan końcowy, nie ucięcie)
        timeout = (not (failure or success)
                   and (self.step_counter + self.PYB_STEPS_PER_CTRL) / self.PYB_FREQ >= self.EPISODE_LEN_SEC)
        if timeout:
            rewards += self.R_TIMEOUT

        self._eval = dict(step=self.step_counter, rewards=rewards.astype(float),
                          terminated=bool(failure or success or timeout),
                          truncated=False,
                          info=dict(is_success=success, success=success, collisions=collisions, timeout=timeout,
                                    out_of_bounds=bool(out), min_dist=dist, cross_x=self.cross_x,
                                    tube_radius=self.TUBE_R, agent_rewards=rewards.astype(float)))
        return self._eval

    def _computeReward(self):
        """Suma nagród obu dronów (dla zgodności z gymnasium); osobne są w info['agent_rewards']."""
        return float(self._evaluate()["rewards"].sum())

    def _computeTerminated(self):
        return self._evaluate()["terminated"]

    def _computeTruncated(self):
        return self._evaluate()["truncated"]

    def _computeInfo(self):
        if self._prev_phi is None:   # wywołanie z reset()
            return dict(is_success=False, success=False, collisions=[], out_of_bounds=False, timeout=False,
                        min_dist=float(np.linalg.norm(self.pos[0] - self.pos[1])), cross_x=None,
                        tube_radius=self.TUBE_R, agent_rewards=np.zeros(2))
        return self._evaluate()["info"]
