# hidden_osc

## Coupled Harmonic Oscillators

First I generate data files for some coupled harmonic oscillators.
```bash
python data_gen/harmonic_osc.py
```
I simulate the trajectories and save the time series of the coordinates.
Although I hide data for one of the oscillators.

Now the problem is to learn from the incomplete time series data to
1. make forecasts learning from incomplete time series data, and
1. guess the coupling of the oscillators.
