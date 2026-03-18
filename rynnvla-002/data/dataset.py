import copy
import json
import logging
import os
from pathlib import Path
import pickle
from time import sleep
import traceback
import warnings
import math
import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml

try:
    from libero.libero import benchmark
except ImportError:
    benchmark = None
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class LiberoFinetuneConversation(Dataset):
    def __init__(self, config_path, resolution, with_state=True, with_wrist=True, with_action=True, with_world_model=True):
        logger.info(f"read dataset config from {config_path}")
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.task_suite_name = self.config["META"]["libero_task_suite"]
        self.use_custom_task_files = self.task_suite_name == "custom_files"
        self.task_entries = self._build_task_entries()
        self.num_tasks_in_suite = len(self.task_entries)

        if self.config["META"]["split"]=="all":
            self.split_training_set = False
        else:
            self.split_training_set = True
        self.with_state = with_state
        self.with_wrist = with_wrist
        self.with_action = with_action
        self.with_world_model = with_world_model
        self.get_annotation_data(split=self.config["META"]["split"])

    def _build_task_entries(self):
        if self.use_custom_task_files:
            return self._build_custom_task_entries()

        if benchmark is None:
            raise ImportError(
                "libero is required for built-in task suites. "
                "Use libero_task_suite: custom_files to provide explicit HDF5 paths."
            )

        benchmark_dict = benchmark.get_benchmark_dict()
        logger.info(benchmark_dict)
        logger.info(self.task_suite_name)
        self.task_suite = benchmark_dict[self.task_suite_name]()

        task_dic = {}
        for task_id in range(self.task_suite.n_tasks):
            task = self.task_suite.get_task(task_id).name
            task_dic[task] = task_id

        task_entries = []
        for task_name in sorted(task_dic.keys()):
            task_entries.append(
                {
                    "task_id": task_dic[task_name],
                    "task_name": task_name,
                    "task_name_readable": task_name.replace("_", " "),
                    "data_path": os.path.join(self.config["META"]["raw_data_dir"], f"{task_name}_demo.hdf5"),
                }
            )
        return task_entries

    def _build_custom_task_entries(self):
        raw_task_files = self.config["META"].get("task_files", [])
        if not raw_task_files:
            raise ValueError(
                "custom_files suite requires META.task_files with at least one HDF5 path."
            )

        task_entries = []
        for task_id, task_file in enumerate(raw_task_files):
            if isinstance(task_file, str):
                path = task_file
                task_name = Path(path).stem
                prompt_name = task_name.replace("_", " ")
            elif isinstance(task_file, dict):
                path = task_file["path"]
                task_name = task_file.get("task_name") or Path(path).stem
                prompt_name = task_file.get("prompt_name") or task_file.get("task_name") or task_name.replace("_", " ")
            else:
                raise TypeError("Each META.task_files entry must be a string path or a mapping.")

            if task_name.endswith("_demo"):
                task_name = task_name[:-5]

            resolved_path = self._resolve_task_file_path(path)
            if not Path(resolved_path).exists():
                raise FileNotFoundError(
                    "custom_files task path not found.\n"
                    f"config: {self.config_path}\n"
                    f"task_name: {task_name}\n"
                    f"original path: {path}\n"
                    f"resolved path: {resolved_path}"
                )

            task_entries.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_name_readable": prompt_name.replace("_", " "),
                    "data_path": resolved_path,
                }
            )

        logger.info("Using custom task files: %s", task_entries)
        return task_entries

    def _resolve_task_file_path(self, path):
        path_obj = Path(path)
        if path_obj.is_absolute() or path_obj.exists():
            return str(path_obj)

        config_relative_path = self.config_dir / path_obj
        if config_relative_path.exists():
            return str(config_relative_path)

        return str(path_obj)

    def _open_hdf5(self, data_path, task_name):
        if not Path(data_path).exists():
            raise FileNotFoundError(
                "dataset HDF5 file not found.\n"
                f"config: {self.config_path}\n"
                f"task_name: {task_name}\n"
                f"data_path: {data_path}"
            )
        try:
            return h5py.File(data_path, "r")
        except OSError as exc:
            raise OSError(
                "failed to open dataset HDF5 file.\n"
                f"config: {self.config_path}\n"
                f"task_name: {task_name}\n"
                f"data_path: {data_path}\n"
                f"original error: {exc}"
            ) from exc

    def get_annotation_data(self, split='train'):
        self.data_list = []
        split_index_ood = math.ceil(self.num_tasks_in_suite * 0.9)

        for task_id_new, task_entry in enumerate(self.task_entries):
            task_name = task_entry["task_name"]
            task_id = task_entry["task_id"]
            orig_data_path = task_entry["data_path"]
            orig_data_file = self._open_hdf5(orig_data_path, task_name)
            orig_data = orig_data_file["data"]
        
            trj_list = []
            len_trj_list = 0
            for i in range(50):
                # Get demo data
                if f"demo_{i}" in orig_data:
                    len_trj_list+=1
                    trj_list.append(i)
            split_index_ind = math.ceil(len_trj_list * 0.9)
            for i in range(len(trj_list)):
                trj_id = trj_list[i]
                if self.split_training_set:
                    if task_id_new<split_index_ood:
                        if i<split_index_ind:
                            cur_split = 'train'
                        else:
                            cur_split = 'val'
                    else:
                        cur_split = 'val_ood'
                    if split!=cur_split:
                        continue
                demo_data = orig_data[f"demo_{trj_id}"]
                orig_actions = demo_data["actions"][()]
                for j in range(orig_actions.shape[0]):
                    if self.with_action:
                        action_data = self.get_action_data(
                            j,
                            orig_actions.shape[0],
                            orig_actions,
                            trj=trj_id,
                            task_name=task_name,
                            task_id=task_id,
                            task_name_readable=task_entry["task_name_readable"],
                            data_path=orig_data_path,
                        )
                        if action_data is not None:
                            self.data_list.append(action_data)
                    if self.with_world_model:
                        world_data = self.get_world_model_data(
                            j,
                            orig_actions.shape[0],
                            orig_actions,
                            trj=trj_id,
                            task_name=task_name,
                            task_id=task_id,
                            task_name_readable=task_entry["task_name_readable"],
                            data_path=orig_data_path,
                        )
                        if world_data is not None:
                            self.data_list.append(world_data)

    def get_action_data(self, action_idx, action_sum, orig_actions, trj, task_name, task_id, task_name_readable, data_path):
        len_action = self.config["action_model"]["len_action"]
        his = self.config["action_model"]["his"]
        if action_idx>action_sum-len_action:
            return None
        
        data = {}

        img_history_start_idx = max(0, action_idx - his + 1)
        data['image_idx'] = list(range(action_sum)[img_history_start_idx:action_idx+1])
        data['action_ids'] = list(range(action_idx, action_idx + len_action))

        data['task_name'] = task_name
        data['task_name_readable'] = task_name_readable
        data['trj'] = trj
        data['task_type'] = 'action'
        data['task_id'] = task_id
        data['state_id'] = action_idx
        data['data_path'] = data_path
        return data

    def get_world_model_data(self, action_idx, action_sum, orig_actions, trj, task_name, task_id, task_name_readable, data_path):
        his = self.config["world_model"]["his"]
        if action_idx>action_sum-his-1:
            return None
        data = {}

        historical_images_idx = list(range(max(action_idx - his + 1, 0), action_idx + 1))
        future_images_idx = list(range(action_idx + 1, action_idx + 2))
        data['image_idx'] = historical_images_idx+future_images_idx
        data['action_ids'] = historical_images_idx
        data['task_name'] = task_name
        data['task_name_readable'] = task_name_readable
        data['trj'] = trj
        data['task_type'] = 'world'
        data['task_id'] = task_id
        data['data_path'] = data_path
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        action_ids = self.data_list[idx]['action_ids']
        trj = self.data_list[idx]['trj']
        task_name = self.data_list[idx]['task_name']
        task_name_readable = self.data_list[idx]['task_name_readable']
        orig_data_path = self.data_list[idx]['data_path']
        orig_data_file = self._open_hdf5(orig_data_path, task_name)
        orig_data = orig_data_file["data"]
        demo_data = orig_data[f"demo_{trj}"]
        orig_rgb = demo_data['obs']['agentview_rgb'][()]
        orig_rgb_wrist = demo_data['obs']['eye_in_hand_rgb'][()]
        orig_actions = demo_data["actions"][()]
        orig_ee_states = demo_data['obs']["ee_states"][()]
        orig_gripper_states = demo_data['obs']["gripper_states"][()]
        action = [copy.deepcopy(orig_actions[idx]) for idx in action_ids]

        images = []
        for image_idx in self.data_list[idx]['image_idx']:
            images.append(Image.fromarray(orig_rgb[image_idx][::-1, ::-1].astype(np.uint8)))
            if self.with_wrist:
                images.append(Image.fromarray(orig_rgb_wrist[image_idx][::-1, ::-1].astype(np.uint8)))

        combined_state = []
        if self.data_list[idx]['task_type']=='action':
            if self.with_state:
                state_id = self.data_list[idx]['state_id']
                ee_state = orig_ee_states[state_id]
                gripper_state = orig_gripper_states[state_id]
                combined_state = np.concatenate([ee_state, gripper_state])

                conversations =[
                    {
                        "from": "human",
                        "value": f"What action should the robot take to {task_name_readable}?" + "<|state|>" + "<|image|>" * len(images)
                    },
                    {
                        "from": "gpt",
                        "value": "<|action|>" * len(action)
                    },
                ]
            else:    
              conversations =[
                  {
                      "from": "human",
                      "value": f"What action should the robot take to {task_name_readable}?" + "<|image|>" * len(images)
                  },
                  {
                      "from": "gpt",
                      "value": "<|action|>" * len(action)
                  },
              ]
                  
        elif self.data_list[idx]['task_type']=='world':
            if self.with_wrist:
                conversations = [
                    {
                        "from": "human",
                        # Revised prompt to accurately reflect variable 'his' length
                        "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|image|><|action|>" * len(action)
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|><|image|>" # The model generates a single image
                    },
                ]
            else:
                conversations = [
                    {
                        "from": "human",
                        # Revised prompt to accurately reflect variable 'his' length
                        "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>" * len(action)
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" # The model generates a single image
                    },
                ]
        # print(conversations)
        # print('***********')
        # tokens, labels = self.item_processor.process_item(conv, training_mode=True)
        return conversations, images, action, combined_state

if __name__=='__main__':
    data = LiberoFinetuneConversation('/mnt/damorobot/yuanyq/code/WorldVLA-main/worldvla/configs/libero_256_all/debug.yaml', 256, True)
    print(data.__len__())
    import pdb 
    pdb.set_trace()
