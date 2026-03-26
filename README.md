# MPC Path following 
Repozytorium zawiera implementację modelu matematycznego modelu wraz ze sterowaniem MPC w path following.

## Paczki python
```
pip install numpy scipy matplotlib casadi pygame
```

## Uruchomienie

### Statyczna symulacja modelu
Pliki PowerBoat_Model_Python.py i dwa pozostałe z implementacją silnika zawierają symulację do sprawdzenia zachowania modelu.

```
python ./PowerBoat_Model_Python_2silnik.py 
```

### Path following
Plik casadi_model.py zawiera implementację dynamiki modelu zaimplementowaną za pomocą biblioteki CasADi.

Plik casadi_mpc.py zawiera implementację sterowania predykcyjnego.

Plik casadi_path_following.py zawiera symulację path following na podstawie MPC.

```
python ./casadi_path_following.py
```