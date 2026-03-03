import pandas as pd
import numpy as np
import torch

class SensorDataset:
    def __init__(self, meta_path, ts_path, id_col, time_col, value_col, device):
        self.meta = pd.read_csv(meta_path)
        self.ts = pd.read_csv(ts_path)

        self.id_col = id_col
        self.time_col = time_col
        self.value_col = value_col
        self.device = device

        self.sensor_ids = self.meta[id_col].tolist()
        self.id_to_idx = {sid: i for i, sid in enumerate(self.sensor_ids)}

        # positions
        self.pos = {
            sid: (
                float(self.meta.loc[self.meta[id_col] == sid, "r"].iloc[0]),
                float(self.meta.loc[self.meta[id_col] == sid, "z"].iloc[0]),
            )
            for sid in self.sensor_ids
        }

        # group time series by sensor id
        self.groups = {
            sid: g[[time_col, value_col]].to_numpy()
            for sid, g in self.ts.groupby(id_col)
        }

    def num_sensors(self):
        return len(self.sensor_ids)

    def sample_batch(self, n_sensors, n_time_per_sensor, max_tries: int = 5):
        """
        Returns (r,z,t,y,sidx). If dataset has missing sensors, we retry.
        """
        for _ in range(max_tries):
            chosen = np.random.choice(
                self.sensor_ids,
                size=min(n_sensors, len(self.sensor_ids)),
                replace=False,
            )

            rr, zz, tt, yy, idxs = [], [], [], [], []
            for sid in chosen:
                arr = self.groups.get(sid, None)
                if arr is None or len(arr) == 0:
                    continue

                pick = np.random.choice(len(arr), size=min(n_time_per_sensor, len(arr)), replace=False)
                r, z = self.pos[sid]
                for j in pick:
                    tval, yval = arr[j, 0], arr[j, 1]
                    rr.append(r); zz.append(z); tt.append(tval); yy.append(yval)
                    idxs.append(self.id_to_idx[sid])

            if len(rr) > 0:
                r = torch.tensor(rr, dtype=torch.float32, device=self.device).reshape(-1, 1)
                z = torch.tensor(zz, dtype=torch.float32, device=self.device).reshape(-1, 1)
                t = torch.tensor(tt, dtype=torch.float32, device=self.device).reshape(-1, 1)
                y = torch.tensor(yy, dtype=torch.float32, device=self.device).reshape(-1, 1)
                sidx = torch.tensor(idxs, dtype=torch.long, device=self.device)
                return r, z, t, y, sidx

        # fallback: empty batch (caller will handle)
        empty = torch.empty((0,1), dtype=torch.float32, device=self.device)
        return empty, empty, empty, empty, torch.empty((0,), dtype=torch.long, device=self.device)