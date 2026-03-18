import os
import argparse
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import io

def to_rgb_uint8(x):
    import numpy as np
    from PIL import Image
    import io, os

    # dict: {'bytes': ..., 'path': ...}
    if isinstance(x, dict):
        if x.get("bytes", None) is not None:
            x = np.array(Image.open(io.BytesIO(x["bytes"])).convert("RGB"))
        elif x.get("path", None) is not None:
            if not os.path.exists(x["path"]):
                raise FileNotFoundError(f"image path not found: {x['path']}")
            x = np.array(Image.open(x["path"]).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image dict keys: {list(x.keys())}")

    # bytes
    if isinstance(x, (bytes, bytearray)):
        x = np.array(Image.open(io.BytesIO(x)).convert("RGB"))

    # PIL
    if hasattr(x, "mode") and hasattr(x, "size"):
        x = np.array(x.convert("RGB"))

    arr = np.asarray(x)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # 灰度 -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    assert arr.ndim == 3 and arr.shape[-1] == 3, f"invalid image shape: {arr.shape}"
    return arr

def to_float1d(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="demo.hdf5")  # 你要求的文件名
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)
    required_cols = ["image", "wrist_image", "actions", "state", "episode_index"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    with h5py.File(out_path, "w") as f:
        g_data = f.create_group("data")

        # episode_index 分组
        for epi in sorted(df["episode_index"].unique()):
            sub = df[df["episode_index"] == epi].reset_index(drop=True)
            g_demo = g_data.create_group(f"demo_{int(epi)}")
            g_obs = g_demo.create_group("obs")

            front_list, wrist_list, act_list, ee_list, grip_list = [], [], [], [], []

            for i in range(len(sub)):
                front = to_rgb_uint8(sub.loc[i, "image"])
                wrist = to_rgb_uint8(sub.loc[i, "wrist_image"])
                action = to_float1d(sub.loc[i, "actions"])
                state = to_float1d(sub.loc[i, "state"])

                # state 单列拆分：最后1维作为 gripper，其余作为 ee
                if state.shape[0] < 2:
                    raise ValueError(f"state dim too small at row {i}, got {state.shape}")
                ee = state[:-1]
                grip = state[-1:]

                front_list.append(front)
                wrist_list.append(wrist)
                act_list.append(action)
                ee_list.append(ee)
                grip_list.append(grip)

            g_demo.create_dataset("actions", data=np.stack(act_list).astype(np.float32))
            g_obs.create_dataset("agentview_rgb", data=np.stack(front_list).astype(np.uint8))
            g_obs.create_dataset("eye_in_hand_rgb", data=np.stack(wrist_list).astype(np.uint8))
            g_obs.create_dataset("ee_states", data=np.stack(ee_list).astype(np.float32))
            g_obs.create_dataset("gripper_states", data=np.stack(grip_list).astype(np.float32))

    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()