# Przewodnik po projekcie: krok po kroku, od podstaw do szczegółów

Ten dokument prowadzi przez projekt w kolejności, w jakiej najłatwiej go zrozumieć:
najpierw pojęcia, potem symulator, potem nasze środowisko linijka po linijce,
na końcu uczenie. Każdy etap kończy się **ćwiczeniem** (coś do uruchomienia
albo zmienienia) i **pytaniami kontrolnymi** (odpowiedzi są pod spoilerem).

Wyniki i historia treningów są w [RAPORT.md](RAPORT.md), a komendy w [README.md](README.md).
Tutaj chodzi o to, **jak to działa i dlaczego**.

Wszystkie ścieżki są podane względem katalogu głównego repozytorium. Przed ćwiczeniami
aktywuj środowisko:

```sh
cd ~/Documents/RISA_2/drony/gym-pybullet-drones
source .venv/bin/activate.fish        # bash: source .venv/bin/activate
```

---

## Mapa drogi

| Etap | Temat | Główne pliki |
|---|---|---|
| 1 | Pojęcia RL: agent, środowisko, obserwacja, akcja, nagroda | — |
| 2 | Interfejs Gymnasium: `reset()` i `step()` | — |
| 3 | Symulator `gym-pybullet-drones`: jak liczy fizykę drona | `envs/BaseAviary.py`, `envs/BaseRLAviary.py` |
| 4 | Regulator PID: jak zadana prędkość staje się obrotami silników | `control/DSLPIDControl.py` |
| 5 | Nasze środowisko `TubeAviary`: scena i układy współrzędnych | `envs/TubeAviary.py` |
| 6 | `TubeAviary`: akcja i obserwacja | `envs/TubeAviary.py` |
| 7 | `TubeAviary`: nagroda, kolizje, koniec epizodu | `envs/TubeAviary.py` |
| 8 | Dodatkowa fizyka: downwash i turbulencje | `envs/TubeAviary.py` |
| 9 | Wielu agentów: jak dwa drony stają się dwoma agentami | `utils/ma_vec_env.py` |
| 10 | Algorytm PPO i sieć neuronowa | Stable-Baselines3 |
| 11 | Skrypt treningu: curriculum, ewaluacja, zapisywanie | `examples/learn_tube.py` |
| 12 | Podgląd, ocena i ręczny sterownik | `examples/play_tube.py` |
| 13 | Czego nauczyły nas nieudane treningi | `tube_project/logs/` |
| 14 | Eksperymenty do samodzielnego zrobienia | — |

Ścieżki `envs/…`, `control/…`, `utils/…`, `examples/…` leżą w `gym_pybullet_drones/`.

---

## Etap 1. Pojęcia uczenia ze wzmocnieniem

Uczenie ze wzmocnieniem (RL, *reinforcement learning*) to uczenie przez próby i błędy.
Nikt nie pokazuje agentowi, jak ma lecieć. Agent sam próbuje i dostaje liczbę, która
mówi, czy to było dobre.

```
            akcja a_t
   ┌───────────────────────────┐
   │                           ▼
┌──────┐                 ┌─────────────┐
│AGENT │                 │ ŚRODOWISKO  │
│(sieć)│ ◄────────────── │ (symulator) │
└──────┘  obserwacja o_t+1, nagroda r_t+1
```

| Pojęcie | Co to jest | W naszym projekcie |
|---|---|---|
| **Agent** | ten, kto podejmuje decyzje | sieć neuronowa sterująca jednym dronem |
| **Środowisko** | świat, na który agent działa | symulacja PyBullet z tubą i dwoma dronami |
| **Obserwacja** | to, co agent wie o świecie w danej chwili | 24 liczby: pozycja, prędkość, drugi dron, promień tuby… |
| **Akcja** | decyzja agenta | 3 liczby: zadana prędkość w osiach x, y, z |
| **Nagroda** | liczba oceniająca ostatni krok | np. +10·postęp, −0,05 za upływ czasu, −20 za zderzenie |
| **Epizod** | jedna próba od startu do końca | jeden przelot: od startu do sukcesu, kolizji lub upływu 15 s |
| **Polityka** π(a\|o) | reguła wyboru akcji na podstawie obserwacji | to, czego uczy się sieć |
| **Zwrot** (return) | suma nagród w epizodzie, z dyskontem | to, co agent stara się maksymalizować |

**Dyskonto γ (gamma).** Nagroda za 100 kroków jest mniej warta niż teraz:
liczy się jako γ¹⁰⁰·r. U nas γ = 0,995, więc nagroda za 100 kroków (3,3 s) waży 0,6,
a za 450 kroków (cały epizod) 0,1. Dobór γ okazał się w tym projekcie ważny (etap 13).

**Krok czasowy.** Agent decyduje 30 razy na sekundę, więc jeden krok to 1/30 s lotu.

### Ćwiczenie 1
Wypisz sobie na kartce, co **Ty** chciałbyś wiedzieć, gdybyś miał przelecieć dronem przez
tubę naprzeciw drugiego drona z zawiązanymi oczami i dostawał tylko liczby. Porównaj potem
z obserwacją w etapie 6.

<details><summary>Pytania kontrolne</summary>

1. *Czym różni się obserwacja od stanu?* Stan to pełny opis świata (np. wszystkie pozycje,
   prędkości, siły). Obserwacja to tylko to, co dostaje agent. U nas dron nie zna np. akcji
   drugiego drona, więc jego obserwacja to nie cały stan.
2. *Dlaczego γ < 1?* Żeby zwrot był skończony i żeby szybciej zdobyta nagroda była lepsza.
</details>

---

## Etap 2. Interfejs Gymnasium

Każde środowisko RL w Pythonie (Gymnasium) ma dwie główne metody:

```python
obs, info = env.reset(seed=0)            # nowy epizod, zwraca pierwszą obserwację
obs, reward, terminated, truncated, info = env.step(action)   # jeden krok
```

- `terminated = True` – epizod skończył się „naturalnie” (sukces, kolizja, u nas też koniec czasu).
- `truncated = True` – epizod ucięto z zewnątrz (np. limit kroków). U nas zawsze `False`, wyjaśnienie w etapie 7.
- `info` – słownik z dodatkowymi danymi (u nas m.in. `success`, `collisions`, `agent_rewards`).

Środowisko deklaruje też **przestrzenie**: `observation_space` (kształt i zakres obserwacji)
oraz `action_space` (kształt i zakres akcji). Dzięki nim biblioteka uczenia wie, jak zbudować sieć.

### Ćwiczenie 2: pierwszy kontakt ze środowiskiem

```python
# uruchom: python  (w katalogu repozytorium, z aktywnym .venv)
import numpy as np
from gym_pybullet_drones.envs.TubeAviary import TubeAviary

env = TubeAviary(tube_radius=0.18, gui=True, randomize_start=False)   # start dokładnie na osi
obs, info = env.reset(seed=0)
print(env.observation_space, env.action_space)    # Box(2, 24) i Box(2, 3)
print(obs.shape)                                  # (2, 24): wiersz na drona

for k in range(300):
    action = np.array([[1, 0, 0], [1, 0, 0]])     # obaj lecą "prosto" z pełną prędkością
    obs, reward, term, trunc, info = env.step(action)
    if term:
        print(k, info["collisions"], info["success"]); break
env.close()
```

Co się stanie? Drony polecą prosto po osi tuby i po ok. 2,8 s zderzą się czołowo w środku
(`drone_drone`). Spróbuj dać drugiemu dronowi `[1, -0.5, 0]`, czyli lot na ukos w prawo w jego
układzie: tym razem uderzy w krawędź wlotu (`drone1_wall`), bo zjeżdża w bok, zanim wleci do tuby.
Ręcznie dobrać dobry tor nie jest łatwo, i właśnie tego ma się nauczyć sieć.

Bez `randomize_start=False` start jest losowo przesunięty i dron lecący „na ślepo” prosto
zwykle zahacza o wlot.

<details><summary>Pytania kontrolne</summary>

1. *Dlaczego `obs` ma kształt (2, 24), a nie (24,)?* Środowisko zwraca obserwacje obu dronów
   naraz. Na dwóch niezależnych agentów rozdziela je dopiero `MultiAgentVecEnv` (etap 9).
2. *Dlaczego obaj dostają akcję `[1, 0, 0]`, a lecą w przeciwne strony?* Akcja jest w układzie
   drona, a układ drona 1 jest obrócony o 180° (etap 5).
</details>

---

## Etap 3. Symulator `gym-pybullet-drones`

Biblioteka składa się z warstw. Nasza klasa dziedziczy po nich i nadpisuje wybrane metody:

```
gym.Env
  └── BaseAviary        (envs/BaseAviary.py)    fizyka, pętla kroku, drony z pliku URDF
        └── BaseRLAviary   (envs/BaseRLAviary.py)  typowe akcje/obserwacje do RL, kontrolery PID
              └── TubeAviary   (envs/TubeAviary.py)   NASZE: tuba, nagroda, kolizje, obserwacje
```

### 3.1 Dron w PyBullet

PyBullet to silnik fizyki brył sztywnych: liczy ruch obiektów pod wpływem sił i kontaktów.
**Nie liczy przepływu powietrza.** Dron jest bryłą o masie 27 g wczytaną z pliku
`gym_pybullet_drones/assets/cf2x.urdf` (Crazyflie 2.x). Ciąg śmigieł biblioteka dodaje
sama, jako siły zewnętrzne, w `BaseAviary._physics` (`envs/BaseAviary.py:676`):

```python
forces = np.array(rpm**2) * self.KF          # ciąg każdego z 4 silników  F = kf · ω²
torques = np.array(rpm**2) * self.KM         # moment oporu śmigła       M = km · ω²
z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])   # obrót wokół osi pionowej
for i in range(4):
    p.applyExternalForce(DRONE_ID, i, forceObj=[0, 0, forces[i]], ...)   # siła w górę na ramieniu i
p.applyExternalTorque(DRONE_ID, 4, torqueObj=[0, 0, z_torque], ...)
```

Czyli: **dron = bryła + 4 siły w górę na końcach ramion + moment obrotowy**. Różnica ciągów
między silnikami przechyla drona, a przechylony ciąg pcha go w bok. Tak lata każdy quadrocopter.

### 3.2 Pętla kroku

`BaseAviary.step()` (`envs/BaseAviary.py:258`) w uproszczeniu:

```python
def step(self, action):
    rpm = self._preprocessAction(action)          # akcja agenta -> obroty 4 silników (u nas: przez PID)
    for _ in range(self.PYB_STEPS_PER_CTRL):      # 240 Hz / 30 Hz = 8 kroków fizyki na 1 decyzję
        for i in range(NUM_DRONES):
            self._physics(rpm[i], i)              # siły śmigieł
            # (+ opcjonalnie _downwash, _groundEffect, _drag – zależnie od trybu Physics)
        p.stepSimulation()                        # PyBullet przesuwa świat o 1/240 s
    self._updateAndStoreKinematicInformation()    # zapisuje self.pos, self.vel, self.rpy, self.ang_v
    obs = self._computeObs()
    reward = self._computeReward()
    terminated = self._computeTerminated()
    truncated = self._computeTruncated()
    info = self._computeInfo()
    return obs, reward, terminated, truncated, info
```

**Ważne:** wszystkie metody z podkreśleniem (`_computeObs`, `_computeReward`, `_preprocessAction`,
`_addObstacles`, `_physics`, …) to „haczyki”. Klasa pochodna podmienia je, żeby zdefiniować
własne zadanie. `TubeAviary` właśnie to robi.

Druga ważna rzecz: `self.pos`, `self.vel` itd. odświeżają się **raz na krok sterowania**,
a nie po każdym z 8 kroków fizyki. To miało znaczenie przy przykładaniu sił (etap 8).

### 3.3 Tryby fizyki (`Physics`)

| Tryb | Co liczy |
|---|---|
| `PYB` (domyślny) | tylko ciąg śmigieł + grawitacja + kontakty |
| `PYB_DW` | + downwash (strumień powietrza z drona powyżej) |
| `PYB_GND`, `PYB_DRAG`, … | + efekt ziemi, opór powietrza |
| `DYN` | własne równania ruchu zamiast PyBullet |

### Ćwiczenie 3
Otwórz `gym_pybullet_drones/envs/BaseAviary.py`, znajdź `step` (linia ok. 258) i porównaj
z uproszczeniem wyżej. Znajdź, gdzie wywoływane jest `self._downwash(i)`.

<details><summary>Pytania kontrolne</summary>

1. *Ile razy PyBullet liczy fizykę między dwiema decyzjami agenta?* 8 razy (240/30).
2. *Dlaczego dron przy zawisie ma wszystkie 4 silniki na tych samych obrotach?* Bo wtedy suma
   ciągów równoważy ciężar, nie ma przechyłu, a momenty oporu par śmigieł się znoszą.
</details>

---

## Etap 4. Regulator PID: od zadanej prędkości do obrotów silników

Agent **nie** steruje silnikami bezpośrednio. Podaje zadaną prędkość, a obroty liczy regulator
`DSLPIDControl` (`control/DSLPIDControl.py`). Ma on dwie pętle:

```
zadana prędkość ──► [pętla położenia/prędkości] ──► żądany wektor ciągu ──► [pętla orientacji] ──► obroty 4 silników
                    _dslPIDPositionControl            (kierunek = żądany przechył)  _dslPIDAttitudeControl
```

Pętla zewnętrzna (`DSLPIDControl.py:156`):

```python
pos_e = target_pos - cur_pos          # u nas target_pos = cur_pos, więc pos_e = 0
vel_e = target_vel - cur_vel          # błąd prędkości
target_thrust = P·pos_e + I·∫pos_e + D·vel_e + [0, 0, m·g]
```

Ponieważ podajemy `target_pos = cur_pos` (aktualną pozycję), działa tylko człon prędkościowy:
regulator dąży do zadanej prędkości i nie trzyma żadnego punktu. Jeśli coś zepchnie drona
(np. podmuch), regulator **nie wraca** do poprzedniego miejsca. Korygowanie toru należy do agenta.
Wektor `target_thrust` mówi, w którą stronę ma być skierowany ciąg, czyli jak przechylić drona.
Pętla wewnętrzna przechyla drona do tej orientacji, a „mikser” rozdziela to na 4 silniki.

**Dlaczego tak, a nie RL na silnikach?** Sterowanie 4 silnikami wymaga utrzymywania drona
w powietrzu w każdej milisekundzie. Sieć musiałaby się tego najpierw nauczyć, co kosztuje
miliony kroków. Podział „RL planuje prędkość, PID stabilizuje” jest szybszy w uczeniu
i bliższy prawdziwemu Crazyflie, którego firmware przyjmuje zadaną prędkość.

### Ćwiczenie 4
W ćwiczeniu 2 zmień akcję na `[[0, 0, 0], [0, 0, 0]]` i obserwuj w GUI: drony powinny wisieć.
Potem `[[0, 0, 1], [0, 0, 0]]`: dron 0 wznosi się z prędkością ok. 0,6 m/s (po 2 s jest ok. 1 m wyżej),
aż wyleci poza obszar (`out_of_bounds`).

<details><summary>Pytanie kontrolne</summary>

*Dlaczego po podmuchu dron nie wraca na swoje miejsce?* Bo regulator śledzi tylko prędkość
(`target_pos = cur_pos`). Powrót na tor musi „wymyślić” polityka.
</details>

---

## Etap 5. `TubeAviary`: scena i układy współrzędnych

Plik: `gym_pybullet_drones/envs/TubeAviary.py`. Czytaj go równolegle z tym etapem.

### 5.1 Konstruktor (`__init__`, linia 48)

Najpierw ustawia parametry sceny (promień, długość, wysokość tuby, wagi nagrody),
**dopiero potem** woła `super().__init__(...)`. Kolejność jest ważna: konstruktor klasy bazowej
buduje świat i woła `_addObstacles`, który potrzebuje promienia tuby.

```python
super().__init__(drone_model=DroneModel.CF2X, num_drones=2, ...,
                 obs=ObservationType.KIN, act=ActionType.VEL)
```

`num_drones=2`, a typy obserwacji i akcji z biblioteki są i tak nadpisane przez nasze metody.

### 5.2 Budowa tuby (`_addObstacles`, linia 213)

PyBullet nie ma gotowej „rury”, więc tuba to 36 płaskich płytek ułożonych w okrąg:

```python
for k in range(36):
    th = 2π·k/36                                     # kąt płytki
    rc = R + t/2                                     # promień środka płytki (R + pół grubości ścianki)
    pozycja = [L/2, rc·cos(th), Z0 + rc·sin(th)]     # płytki leżą wzdłuż osi X
    orientacja = obrót o th wokół osi X
    p.createMultiBody(baseMass=0, ...)               # masa 0 = obiekt nieruchomy
```

Szerokość płytki to `2·(R+t)·tan(π/36)·1.1`, z 10% zakładką, żeby między płytkami nie było szpar.

### 5.3 Układy współrzędnych drona (linie 157–173)

To kluczowy pomysł projektu. Każdy dron ma **własny układ**:

```
dron 0:  początek w (0, 0, 1)    = lewy wlot,  oś x' w prawo   (macierz I)
dron 1:  początek w (1.5, 0, 1)  = prawy wlot, oś x' w lewo    (macierz MIRROR = diag(-1, -1, 1))
```

```python
def toLocal(self, i, v, is_pos=False):     # świat -> układ drona i
    if is_pos: v = v - self.FRAME_ORIGIN[i]
    return self.FRAME_ROT[i] @ v

def progress(self, i):                      # u: postęp wzdłuż tuby (0 = wlot, 1.5 = wylot, 2.0 = cel)
    return self.toLocal(i, self.pos[i], is_pos=True)[0]
```

Po co? **Oba drony widzą wtedy identyczne zadanie**: „jestem przed wlotem, lecę w +x', z naprzeciwka
leci ktoś”. Ta sama sieć może więc sterować każdym z nich. Bez tego sieć musiałaby się nauczyć
dwóch lustrzanych zachowań.

`MIRROR` to obrót o 180° wokół osi pionowej: zmienia znak x i y, z zostaje. Jest swoją
własną odwrotnością (MIRROR·MIRROR = I), dlatego `toWorld` używa tej samej macierzy.

### 5.4 Reset (linia 195)

```python
self.TUBE_R = losowo z [lo, hi]                     # nowy promień co epizod
self.INIT_XYZS = self._startPositions(randomize)    # start 0,9 m przed wlotem ± losowe przesunięcie
self.INIT_RPYS = [[0,0,0], [0,0,π]]                 # dron 1 obrócony "przodem" do tuby
for c in self.ctrl: c.reset()                       # wyzeruj całki w regulatorach PID
super().reset()                                     # biblioteka przebudowuje świat (i tubę!)
self._prev_phi = self._potential()                  # punkt odniesienia dla nagrody (etap 7)
```

Tuba jest budowana od nowa przy każdym resecie, więc może mieć inny promień w każdym epizodzie.

### Ćwiczenie 5
```python
env = TubeAviary(tube_radius=(0.14, 0.20))
for s in range(3):
    env.reset(seed=s); print(round(env.TUBE_R, 3), env.pos.round(2), env.progress(0), env.progress(1))
```
Sprawdź, że `progress` obu dronów jest na starcie ok. −0,9 (± losowe przesunięcie 0,15), mimo że ich
pozycje x w świecie to ok. −0,9 i 2,4.

<details><summary>Pytania kontrolne</summary>

1. *Co by się stało, gdyby oba drony miały ten sam układ (bez MIRROR)?* Dla drona 1 „do przodu”
   znaczyłoby −x. Wspólna sieć dostawałaby sprzeczne przykłady i uczyłaby się dużo gorzej.
2. *Dlaczego parametry ustawia się przed `super().__init__`?* Bo konstruktor bazowy woła
   `_addObstacles`, który potrzebuje `TUBE_R`, `TUBE_L` i `Z0`.
</details>

---

## Etap 6. `TubeAviary`: akcja i obserwacja

### 6.1 Akcja (`_actionSpace`, `_preprocessAction`, linie 316–339)

```python
spaces.Box(low=-1, high=1, shape=(2, 3))       # 2 drony × [vx, vy, vz]

def _preprocessAction(self, action):
    action = clip(action, -1, 1)
    self.action_buffer.append(action)          # trafi do obserwacji w następnym kroku
    for i in range(2):
        v_world = self.toWorld(i, action[i]) * 0.6           # układ drona -> świat, skala 0,6 m/s
        rpm[i] = self.ctrl[i].computeControl(..., target_pos=cur_pos,
                                             target_rpy=[0, 0, aktualny_yaw], target_vel=v_world)
    return rpm
```

**Dlaczego taka postać akcji?** Pierwsze wersje używały akcji z biblioteki: 4 liczby, czyli kierunek
i osobno wartość prędkości. Kierunek był normalizowany, więc szum eksploracji przy akcji bliskiej
zera dawał ruch w **losowym kierunku** z pełną prędkością. Drony szarpały się i nie trafiały nawet
w szeroką tubę. Bezpośrednia zadana prędkość sprawia, że mały szum daje mały ruch.

### 6.2 Obserwacja (`_computeObs`, linia 343)

Dla każdego drona i składamy 24 liczby, wszystkie w **jego** układzie:

```python
obs[i] = [ toLocal(i, pos[i], is_pos=True),     # 3: gdzie jestem względem mojego wlotu
           rpy_lokalne,                          # 3: przechył i obrót (dla drona 1 odbite)
           toLocal(i, vel[i]),                   # 3: moja prędkość
           toLocal(i, ang_v[i]),                 # 3: moja prędkość kątowa
           toLocal(i, pos[j] - pos[i]),          # 3: gdzie jest drugi dron względem mnie
           toLocal(i, vel[j] - vel[i]),          # 3: jak szybko się zbliża
           widoczny,                             # 1: czy go widzę (zasięg sensing_range)
           ostatnia_akcja[i],                    # 3: co ostatnio zrobiłem
           TUBE_R,                               # 1: jak szeroka jest tuba
           pozostały_czas ]                      # 1: od 1 do 0
```

Wszystko pochodzi wprost z symulatora: **nie ma czujników** (kamery, lidaru), tylko dokładny stan.
Opcjonalnie do danych o drugim dronie dodaje się szum (`obs_noise`), a poza zasięgiem
`sensing_range` te liczby są zerami.

**Jak dron „wie”, gdzie są ścianki?** Nie widzi ich. Zna swoją odległość od osi (√(y'²+z'²)
z pierwszych trzech liczb) i promień tuby, więc luz do ścianki to R − √(y'²+z'²).
Sieć musi to policzyć sama, w swoich warstwach.

**Po co każdy element?**
- *pozostały czas*: bez niego koniec epizodu byłby dla agenta nieprzewidywalny, a kara za niego wyglądałaby na losową (etap 7);
- *ostatnia akcja*: pomaga wygładzić sterowanie (sieć wie, co robiła przed chwilą);
- *promień tuby*: promień zmienia się co epizod, a w wąskiej tubie trzeba lecieć inaczej;
- *flaga widoczności*: odróżnia „nie widzę drugiego” od „drugi jest dokładnie w moim środku” (oba dają zera).

### Ćwiczenie 6
```python
env = TubeAviary(tube_radius=0.18); obs, _ = env.reset(seed=0)
np.set_printoptions(precision=2, suppress=True)
print(obs[0]); print(obs[1])
```
Porównaj wiersze: pozycja drugiego drona (indeksy 12–14) ma w obu x ≈ +3,3 (drugi jest
przed każdym z nich), choć w świecie leżą po przeciwnych stronach. Składowa z ma przeciwne
znaki (+0,24 i −0,24): jeśli dron 1 jest wyżej, to dron 0 widzi go nad sobą, a dron 1 widzi drona 0 pod sobą.

<details><summary>Pytania kontrolne</summary>

1. *Dlaczego pozycja drugiego drona jest względna, a nie bezwzględna?* Bo dla decyzji liczy się,
   gdzie on jest **względem mnie**. Wtedy ta sama sytuacja ma ten sam zapis niezależnie od miejsca.
2. *Czy dron zna akcję drugiego drona?* Nie. Zna tylko jego pozycję i prędkość, a o zamiarach
   może wnioskować wyłącznie z ruchu.
</details>

---

## Etap 7. `TubeAviary`: nagroda, kolizje, koniec epizodu

To najważniejsza część projektu, bo **nagroda definiuje zadanie**. Agent nauczy się tego,
za co dostaje punkty, a nie tego, co mieliśmy na myśli.

### 7.1 Jedno miejsce na wszystko: `_evaluate` (linia 456)

Gymnasium pyta o nagrodę, koniec epizodu i info osobnymi metodami. Żeby nie liczyć kolizji
cztery razy, `_evaluate` liczy wszystko raz na krok i zapamiętuje wynik:

```python
if self._eval is not None and self._eval["step"] == self.step_counter:
    return self._eval                             # już policzone w tym kroku
```

`_computeReward`, `_computeTerminated`, `_computeTruncated`, `_computeInfo` tylko czytają ten wynik.
Gymnasium oczekuje **jednej** liczby nagrody, więc `_computeReward` zwraca sumę obu dronów,
a nagrody poszczególnych dronów idą w `info["agent_rewards"]`.

### 7.2 Shaping potencjałowy (`_potentials`, linia 379)

Gdyby nagroda była tylko za sukces (+50 na końcu), agent na początku nigdy by jej nie zobaczył:
losowe ruchy nie przelecą przez tubę. Potrzebny jest sygnał w każdym kroku. Bezpieczny sposób
to **shaping potencjałowy**: definiujemy funkcję Φ(stan) („jak blisko celu jestem”), a nagroda
za krok to

```
r_shaping = 10 · (Φ(teraz) − Φ(poprzednio))
```

Taka nagroda ma ważną własność (twierdzenie Ng i in., 1999): **nie zmienia, która polityka jest
optymalna**. Dron, który poleci do przodu i wróci, nie zarobi nic, bo zyski i straty się znoszą.

Nasz potencjał:

```python
phi[i] = u                                  # postęp wzdłuż tuby (za celem maleje, żeby nie przelatywać)
if u < L:                                   # przed wylotem:
    phi[i] -= max(0, r_i - r_ok)            #   kara za odjazd od osi bliżej ścianki niż r_ok = R − 6 cm − 2 cm
closeness = max(0, 1 − |x0 − x1| / 1.0)     # 1, gdy drony są na tej samej wysokości tuby; 0, gdy ≥ 1 m
phi -= max(0, 0.14 − lat)/0.14 · closeness  # kara, gdy są blisko, a nie rozsunięte w przekroju o 14 cm
```

Ostatni człon uczy wymijania: gdy drugi dron się zbliża, **odsunięcie się w bok od razu podnosi
potencjał**, czyli daje nagrodę.

### 7.3 Nagrody jednorazowe

```python
rewards -= 0.05                                    # każdy krok kosztuje: wiszenie się nie opłaca
if minęły_się_w_tubie: rewards += 10               # obaj
if winny_kolizji: rewards[i] += -20 - 0.05·pozostałe_kroki
if sukces: rewards += 50                           # obaj
if koniec_czasu: rewards += -20                    # obaj
```

Dlaczego kolizja kosztuje też „pozostały czas”? Bez tego rozbicie się na początku byłoby
**tańsze** niż lot: płaci się −20 i nie płaci kosztu czasu za resztę epizodu. Agent szybko by
to odkrył.

### 7.4 Kolizje (`_collisions`, linia 403)

Dwa niezależne testy, wystarczy jeden:

| Rodzaj | Test geometryczny | Test fizyczny |
|---|---|---|
| ścianka | w obszarze tuby: odległość od osi + 6 cm ≥ R | `getContactPoints` z płytką tuby |
| ziemia | z < 5 cm | kontakt z podłożem |
| dron–dron | odległość środków < 12 cm | kontakt brył dronów |

Test geometryczny traktuje drona jako walec o promieniu 6 cm, czyli ostrzej niż fizyka.
Sukces w symulacji oznacza więc realny zapas.

### 7.5 Wymuszenie minięcia w tubie (`_checkCrossing`, linia 439)

Dron 0 startuje z lewej (x mniejsze), więc dopóki x₀ < x₁, drony się nie minęły. Moment,
w którym x₀ ≥ x₁, to minięcie. Jego miejsce musi leżeć w [0; 1,5]. Jeśli jeden dron wyleci
z tuby, zanim się minęły, znaczy to, że drugi czekał na zewnątrz, więc to też porażka obu.

### 7.6 Koniec czasu jako stan terminalny

Upływ 15 s to `terminated = True` (porażka), a nie `truncated`. Różnica jest ważna w PPO:
przy `truncated` algorytm zakłada, że gra trwałaby dalej, i „dolicza” wartość przyszłości,
więc kara za czekanie by się rozmyła. U nas pozostały czas jest w obserwacji, więc koniec
jest przewidywalny i poprawnie jest go traktować jako prawdziwy koniec.

### Ćwiczenie 7
Uruchom ręczny sterownik i obejrzyj nagrody:
```sh
python gym_pybullet_drones/examples/play_tube.py --radius 0.18 --episodes 3 --gui false
```
Udany przelot daje ok. +160 (suma obu dronów). Policz z grubsza: postęp 2,9 m × 10 = 29,
minus 5 s × 30 × 0,05 = 7,5, plus 10 + 50, razem ok. 81 na drona.

<details><summary>Pytania kontrolne</summary>

1. *Dlaczego kara za bliskość (pierwsze wersje) nauczyła drony stawać, a potencjał rozsunięcia nie?*
   Kara za bliskość dało się ominąć, zatrzymując się przed strefą kary. Potencjał nagradza
   **ruch w bok**, więc jedyny sposób, żeby go zebrać, to faktycznie się rozsunąć.
2. *Co by się stało bez kosztu czasu w każdym kroku?* Patrz etap 13: drony uczyły się wisieć.
</details>

---

## Etap 8. Dodatkowa fizyka: downwash i turbulencje

Obie rzeczy są **wyłączone domyślnie** i włączane flagami `--downwash` i `--turbulence`.

### 8.1 Jak dodać własną siłę w PyBullet

PyBullet zeruje siły zewnętrzne po każdym kroku fizyki, więc siłę trzeba przykładać w każdym
z 8 kroków. Dlatego nadpisujemy `_physics` (linia 265), który biblioteka woła w każdym kroku fizyki:

```python
def _physics(self, rpm, nth_drone):
    super()._physics(rpm, nth_drone)          # ciąg śmigieł jak w bibliotece
    if self.TURBULENCE > 0:
        self._turbulence(nth_drone)           # + nasze zakłócenia
```

Siłę przykładamy przez `_applyForceAtCom` (linia 301). Bierze ona orientację prosto z PyBullet
i przykłada siłę w środku masy drona. Pierwsza wersja używała `self.pos` jako punktu przyłożenia,
a ta pozycja jest nieaktualna między odświeżeniami (etap 3.2). Siła działała więc na ramieniu
i dawała moment, który obracał drona.

### 8.2 Downwash (`_downwash`, linia 242)

Model z biblioteki dawał siłę ∝ 1/Δz², czyli tysiące niutonów, gdy drony były na prawie tej samej
wysokości (dron waży 0,27 N). Nasz model to struga powietrza pod dronem:

```
v_i = √(T / (2ρA)) ≈ 4 m/s          prędkość powietrza tuż pod śmigłami (teoria pędu)
r_jet = 6,3 cm + 0,2·Δz              struga rozszerza się w dół
v = v_i · (6,3 cm / r_jet) · exp(−½·(Δxy / (r_jet/2))²)   słabnie w dół i w bok
F = ½·ρ·C_dA·v²  (w dół, maks. ciężar drona)              opór dolnego drona w strudze
```

Daje to do ok. 0,1 N dokładnie pod drugim dronem i praktycznie zero przy przesunięciu ≥ 12 cm.

### 8.3 Turbulencje (`_turbulence`, linia 271)

Model zastępczy, bo PyBullet nie liczy przepływu:

- **Podmuchy**: proces Ornsteina-Uhlenbecka, czyli szum „z pamięcią”, który zmienia się płynnie
  (czas korelacji 0,3 s), a nie skacze co krok:
  ```python
  f += -f/τ·dt + σ·√(2dt/τ)·N(0,1)          # w długim czasie odchylenie f ≈ σ
  σ = siła · ciężar · (0.25 poza tubą … 1 w tubie) · (1 … 2, gdy drugi dron < 0,5 m)
  ```
- **Efekt ścianki**: przyciąganie do ścianki, od zera przy luzie 5 cm do `2·siła·ciężar` przy ściance.

### Ćwiczenie 8
```sh
python gym_pybullet_drones/examples/play_tube.py --model tube_project/models/FINAL_model.zip --radius 0.16 --turbulence 0.03
```
Obserwuj w GUI, jak drony „pływają” w tubie. Porównaj z uruchomieniem bez flagi.

<details><summary>Pytanie kontrolne</summary>

*Dlaczego podmuchy są procesem OU, a nie niezależnym szumem w każdym kroku?* Niezależny szum
przy 240 Hz uśrednia się prawie do zera, bo dron jest bezwładny i go „nie czuje”. Prawdziwe
podmuchy trwają dziesiątki-setki milisekund, a proces OU to odwzorowuje.
</details>

---

## Etap 9. Wielu agentów: `MultiAgentVecEnv`

Plik: `gym_pybullet_drones/utils/ma_vec_env.py`.

### 9.1 Problem

Stable-Baselines3 (SB3) zakłada **jednego** agenta na środowisko: jedna obserwacja,
jedna akcja, jedna nagroda. Nasze środowisko ma dwa drony.

Najprościej byłoby skleić obserwacje obu dronów w 48 liczb i akcje w 6. To jednak byłby
**wspólny mózg**: jedna sieć widziałaby wszystko i sterowała oboma, a tego zakazuje zadanie.

### 9.2 Rozwiązanie

Opakowanie, które z K symulacji po 2 drony robi 2K „osobnych agentów”:

```python
# K = 8 symulacji, w każdej obs (2, 24)  ->  16 agentów, każdy obs (24,)
def _split_obs(self, obs):
    return obs.reshape(self.num_envs, -1)       # (8, 2, 24) -> (16, 24)

def step_async(self, actions):                  # (16, 3) -> (8, 2, 3)
    self.venv.step_async(actions.reshape(K, 2, -1))

def step_wait(self):
    obs, _, dones, infos = self.venv.step_wait()
    rewards[k*2 + i] = infos[k]["agent_rewards"][i]     # każdy dostaje SWOJĄ nagrodę
    dones = np.repeat(dones, 2)                          # epizod kończy się dla obu naraz
```

Dla SB3 wygląda to jak 16 niezależnych środowisk jednoagentowych. Każdy „agent” dostaje
tylko obserwację swojego drona.

### 9.3 Parameter sharing (IPPO)

SB3 trenuje **jedną** sieć na danych ze wszystkich 16 agentów. W locie każdy dron uruchamia tę
sieć osobno, na **własnej** obserwacji (`play_tube.py`):

```python
act = np.stack([model.predict(obs[i])[0] for i in range(2)])   # osobno dla każdego drona
```

Analogia: dwóch kierowców po tym samym kursie prawa jazdy. Mają te same nawyki („trzymaj się
prawej”), ale każdy patrzy własnymi oczami i decyduje sam. Wspólne nawyki ułatwiają
koordynację bez rozmowy.

<details><summary>Pytania kontrolne</summary>

1. *Czy parameter sharing to „wspólny mózg”?* Nie w czasie lotu: każdy dron liczy akcję tylko
   ze swoich danych. Wspólne są tylko wagi, czyli „wykształcenie”.
2. *Dlaczego `dones` powtarzamy dla obu agentów?* Bo symulacja kończy się dla obu dronów naraz
   (np. zderzenie dotyczy obu).
</details>

---

## Etap 10. Algorytm PPO i sieć neuronowa

### 10.1 Aktor i krytyk

PPO (*Proximal Policy Optimization*) używa dwóch sieci:

- **Aktor** (polityka): obserwacja → akcja. Dokładniej: średnia μ rozkładu normalnego akcji.
  W treningu akcja jest losowana z N(μ, σ), co daje eksplorację. W pokazie bierzemy samo μ.
- **Krytyk** (funkcja wartości V): obserwacja → „ile nagrody spodziewam się do końca epizodu”.

```
obserwacja (24) ─┬─ Linear 24→256 ─ tanh ─ Linear 256→256 ─ tanh ─ Linear 256→3 ─► μ   (+ σ: 3 parametry)
                 └─ Linear 24→256 ─ tanh ─ Linear 256→256 ─ tanh ─ Linear 256→1 ─► V
```

Razem 145 415 parametrów. Obie sieci są osobne (nie dzielą warstw).

### 10.2 Jeden cykl uczenia

1. **Zbieranie:** 16 agentów robi po 1024 kroki. To 16 384 przejść (obserwacja, akcja, nagroda).
2. **Przewaga (advantage):** dla każdego kroku liczymy, o ile lepiej poszło, niż przewidywał krytyk:
   A = (rzeczywisty zwrot) − V(obserwacja). Używamy GAE (λ = 0,95), które łączy przewidywania
   krytyka z rzeczywistymi nagrodami.
3. **Aktualizacja aktora:** podnieś prawdopodobieństwo akcji z A > 0, obniż z A < 0, ale
   **z ograniczeniem** (clip 0,2): nowa polityka nie może się zbyt różnić od starej. To jest „proximal”
   w nazwie i powód, dla którego PPO jest stabilne.
4. **Aktualizacja krytyka:** dopasuj V do rzeczywistych zwrotów.
5. Kroki 3–4 powtarza się 10 razy (`n_epochs`) na minibatchach po 2048. Potem zbieramy nowe dane.

### 10.3 Hiperparametry i dlaczego takie

| Parametr | Wartość | Powód |
|---|---|---|
| `gamma` | 0,995 | przy 0,99 kara za koniec czasu po 15 s była prawie niewidoczna |
| `log_std_init` | −1 (σ ≈ 0,37) | przy σ = 1 dron w wąskiej tubie od razu uderzał w ścianę |
| `ent_coef` | 0 | dodatkowa nagroda za losowość dawała chaotyczny ruch |
| `learning_rate` | 3·10⁻⁴ (dotrenowanie: 1·10⁻⁴) | mniejszy przy dotrenowaniu, bo duże kroki psują dobrą politykę |
| `net_arch` | 256-256 | standard dla obserwacji wektorowych |

<details><summary>Pytania kontrolne</summary>

1. *Po co krytyk, skoro działa tylko aktor?* Krytyk jest potrzebny tylko w treningu. Mówi, czy akcja
   była lepsza od przeciętnej w danym stanie, co mocno zmniejsza szum w uczeniu.
2. *Co się stanie, gdy σ spadnie prawie do zera?* Polityka przestaje eksplorować. Model końcowy ma
   σ ≈ 0,1–0,2, czyli jest już bardzo „pewny”, przez co dotrenowanie łatwo go psuje.
</details>

---

## Etap 11. Skrypt treningu `learn_tube.py`

Plik: `gym_pybullet_drones/examples/learn_tube.py`. Przebieg `main()` (linia 81):

```python
env_kwargs = dict(tube_radius=(0.14, 0.20), turbulence=..., physics=...)
train_env = MultiAgentVecEnv(make_vec_env(TubeAviary, n_envs=8, vec_env_cls=SubprocVecEnv))  # 8 procesów
eval_env  = MultiAgentVecEnv(make_vec_env(TubeAviary, n_envs=1))                             # do ewaluacji

model = PPO("MlpPolicy", train_env, ...)            # albo PPO.load(init_model) przy dotrenowaniu
callbacks = [EvalCallback(...), CurriculumCallback(...)]
model.learn(total_timesteps=..., callback=callbacks)
```

- `SubprocVecEnv` uruchamia każdą symulację w osobnym procesie, czyli na osobnym rdzeniu CPU.
- **`EvalCallback`**: co ok. 20 tys. kroków puszcza 10 epizodów agentów (= 5 przelotów) bez losowości
  i zapisuje `best_model.zip`, jeśli wynik jest najlepszy.
- **`CurriculumCallback`** (linia 41): patrzy na `ep_success_buffer`, czyli skuteczność ostatnich 100
  epizodów agentów. Gdy przekroczy 70%, przechodzi na kolejny poziom: zwęża tubę albo zwiększa
  turbulencje. Przed awansem zapisuje `level{k}_model.zip`.

Poziomy curriculum to słowniki, np.:
```python
[dict(radius=(0.29, 0.35)), dict(radius=(0.26, 0.32)), ..., dict(radius=(0.14, 0.20))]          # zwężanie tuby
[dict(radius=(0.14, 0.20), turbulence=0.01), dict(..., turbulence=0.02), dict(..., turbulence=0.03)]  # turbulencje
```
Zmiana idzie do wszystkich symulacji przez `env_method("set_radius_range", ...)`, czyli wywołanie
metody w każdym procesie.

**Dlaczego curriculum?** Na samym początku sieć nie umie latać. W wąskiej tubie każda próba kończy
się uderzeniem w ścianę, więc nie ma z czego się uczyć. W szerokiej tubie sukcesy przychodzą
szybciej, a umiejętność przenosi się na coraz węższe.

### Ćwiczenie 11
Krótki trening „na próbę” (kilka minut, do katalogu tymczasowego):
```sh
cd gym_pybullet_drones/examples
python learn_tube.py --steps 200000 --n_envs 4 --output /tmp/proba
tensorboard --logdir /tmp/proba        # w drugim terminalu, potem http://localhost:6006
```
Po 200 tys. kroków drony jeszcze nic nie umieją, ale zobaczysz, jak rośnie `ep_rew_mean`.

---

## Etap 12. Podgląd i ocena: `play_tube.py`

Plik: `gym_pybullet_drones/examples/play_tube.py`.

- Z `--model`: każdy dron osobno woła `model.predict(obs[i], deterministic=True)`.
- Bez `--model`: `scripted_policy` (linia 26), ręczny sterownik do porównania. Każdy dron
  ustawia się na pasie po prawej (w swoim układzie) w odległości `min(R − 6 cm − 3 cm, 8 cm)`
  od osi i leci prosto. Też jest zdecentralizowany: używa tylko własnej pozycji.
- Po każdym epizodzie wypisuje: wynik, nagrodę, czas, najmniejszą odległość dronów i miejsce minięcia.

Ręczny sterownik pokazuje, czy zadanie w ogóle jest wykonalne, i daje punkt odniesienia:
w szerokiej tubie działa świetnie, a w wąskiej (0,14 m) prawie nigdy. W wąskiej tubie trzeba
lepiej dobrać moment i wielkość odsunięcia, i tego uczy się RL.

### Ćwiczenie 12
Porównaj ręczny sterownik i model na tej samej serii prób:
```sh
python gym_pybullet_drones/examples/play_tube.py --radius 0.15 --episodes 20 --gui false --seed 100
python gym_pybullet_drones/examples/play_tube.py --radius 0.15 --episodes 20 --gui false --seed 100 --model tube_project/models/FINAL_model.zip
```

---

## Etap 13. Czego nauczyły nas nieudane treningi

Pełna tabela jest w [RAPORT.md, rozdział 8](RAPORT.md). Najważniejsze lekcje:

1. **Agent znajdzie najtańszą drogę, nie tę, którą mieliśmy na myśli.** Gdy wiszenie było tańsze
   od ryzyka, drony wisiały. Gdy kara za bliskość działała od 0,4 m, drony stawały na 0,4 m.
2. **Dyskonto zmienia to, co agent „widzi”.** Przy γ = 0,99 kara za 10 s waży ok. 1/20 swojej
   wartości. Rozwiązanie: koszt czasu w każdym kroku, zamiast jednej kary na końcu.
3. **Sposób zapisu akcji ma ogromne znaczenie.** Ta sama fizyka, inna parametryzacja akcji,
   i trening z niemożliwego staje się wykonalny.
4. **Błędy projektu środowiska ujawniają się jako „dziwne zachowanie”.** Drony stawały przy wlocie,
   bo akurat tam zaczynały się widzieć (zasięg 1,5 m = długość tuby).
5. **Długi trening PPO potrafi się załamać.** Zawsze trzeba zapisywać najlepszy model, a przy
   dotrenowaniu zmniejszać learning rate.

Dobre ćwiczenie: otwórz `tube_project/logs/train_run2.log` i `train_run10.log` i porównaj `ep_rew_mean`
i `ep_len_mean`. Długość epizodu bliska 450 (pełne 15 s) oznacza, że drony wisiały do końca czasu.

---

## Etap 14. Eksperymenty do samodzielnego zrobienia

Od najprostszych:

1. **Zmień promień w podglądzie** (`--radius 0.13`) i sprawdź, gdzie model przestaje działać.
2. **Ogranicz widzenie** (`--sensing_range 0.8`) w podglądzie. Model trenowany z pełnym widzeniem
   nagle „nie widzi” drugiego drona do ostatniej chwili.
3. **Usuń promień tuby z obserwacji** (wpisz stałą zamiast `self.TUBE_R` w `_computeObs`)
   i wytrenuj od zera. Czy model sam rozpozna szerokość tuby?
4. **Zmień wagę rozsunięcia** `W_SEPARATION` na 0 i wytrenuj. Czy drony wciąż nauczą się wymijać?
5. **Osobne sieci dla dronów:** zamiast jednej sieci dwie, np. dwa modele PPO trenowane
   naprzemiennie. Trudniejsze, bo drony muszą się „dogadać” bez wspólnych nawyków.
6. **Czujniki odległości:** dodaj do obserwacji 8 promieni `p.rayTestBatch` wokół drona (jak prosty
   lidar) i usuń pozycję względem tuby. Wtedy dron musi „widzieć” ścianki zamiast znać mapę.

---

## Słowniczek

| Termin | Znaczenie |
|---|---|
| advantage (przewaga) | o ile akcja okazała się lepsza niż przeciętnie w danym stanie |
| curriculum | uczenie od łatwych wersji zadania do trudnych |
| downwash | strumień powietrza od śmigieł skierowany w dół |
| epizod | jedna próba od startu do końca |
| GAE | sposób liczenia przewagi łączący przewidywania krytyka z rzeczywistymi nagrodami |
| IPPO | PPO, w którym każdy agent uczy się na własnych obserwacjach i nagrodach |
| parameter sharing | wiele agentów korzysta z tych samych wag sieci |
| PID | regulator: reaguje na błąd (P), jego sumę w czasie (I) i zmianę (D) |
| potencjał (shaping) | funkcja „jak blisko celu”, której przyrost dodaje się do nagrody |
| proces OU | losowy proces zmieniający się płynnie, z „pamięcią” |
| SB3 | Stable-Baselines3, biblioteka z gotowymi algorytmami RL |
| URDF | plik z opisem robota (bryły, masy, połączenia) |
| VecEnv | wiele kopii środowiska uruchomionych naraz |
