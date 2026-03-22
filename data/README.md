# Data Directory

Place the preprocessed `.npy` files here:

- `train_data.npy` — Training inputs, shape: `(N_train, seq_len, n_features)`
- `train_label.npy` — Training targets, shape: `(N_train,)`
- `val_data.npy` — Validation inputs
- `val_label.npy` — Validation targets
- `test_data.npy` — Test inputs
- `test_label.npy` — Test targets

## Dataset Sources

- **GEFCom2014**: Download from the [original competition page](https://www.sciencedirect.com/science/article/pii/S0169207016000133) or [IEEE DataPort](https://ieee-dataport.org/).
- **ISO-NE COVID-19**: Download from [IEEE DataPort](https://ieee-dataport.org/competitions/day-ahead-electricity-demand-forecasting-post-covid-paradigm).

## Preprocessing

The raw datasets should be preprocessed into sliding window format:
1. Extract hourly load values and temperature features.
2. Create input windows of length `seq_len` (default: 168 hours = 1 week).
3. The label for each window is the load value at the next time step (1-hour-ahead forecast).
4. Split chronologically: training / validation / test.
5. Save as `.npy` files.
