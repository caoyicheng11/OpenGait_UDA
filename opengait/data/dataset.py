import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
from sklearn.cluster import DBSCAN
import pandas as pd
from collections import defaultdict
import numpy as np
import torch


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.eps = data_cfg['eps']
        self.min_samples = data_cfg['min_samples']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}

        self.ground_truth = self.label_list
        self.ground_set = sorted(list(set(self.label_list)))

        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
    
    def __calculate_accuracy(self):
        msg_mgr = get_msg_mgr()

        msg_mgr.log_info(f'Number of cluster: {len(self.label_set)}')
        msg_mgr.log_info(f'Number of noise: {np.sum(self.label_list == -1)}')

        df = pd.DataFrame({"label": self.ground_truth, "pseudo_label": self.label_list})
        df = df[df["pseudo_label"] != -1]

        pl_to_main_label = {}
        for pl in df["pseudo_label"].unique():
            sub_df = df[df["pseudo_label"] == pl]
            main_label = sub_df["label"].mode()[0]
            pl_to_main_label[pl] = main_label

        main_label_to_pl = defaultdict(set)
        for pl, main_label in pl_to_main_label.items():
            main_label_to_pl[main_label].add(pl)

        valid_pseudo_labels = set()
        for main_label, pls in main_label_to_pl.items():
            valid_pseudo_labels.update(pls)

        valid_samples = df[df["pseudo_label"].isin(valid_pseudo_labels)]
        valid_count = len(valid_samples)

        total_count = len(df)

        unique_accuracy = valid_count / total_count

        correct_count = 0
        for idx, row in valid_samples.iterrows():
            if pl_to_main_label[row["pseudo_label"]] == row["label"]:
                correct_count += 1

        final_accuracy = correct_count / total_count
        msg_mgr.log_info(f'Cluster accuracy: {final_accuracy}')

        type_stats = defaultdict(lambda: {"total": 0, "valid": 0})
    
        for label, type_ in zip(self.label_list, self.types_list):
            type_stats[type_[:2]]["total"] += 1
            if label != -1:
                type_stats[type_[:2]]["valid"] += 1
        for type_, stats in type_stats.items():
            ratio = stats["valid"] / stats["total"]
            print(f"{type_}: {ratio:.2%} ({stats['valid']}/{stats['total']})")

        return

    def __get_cluster_centers(self, embeddings, labels):
        valid_mask = (self.label_list != -1)
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]

        ref_embed = []
        ref_label = []

        for label in self.label_set:
            class_embeddings = embeddings[labels == label]
            class_mean = np.mean(class_embeddings, axis=0)
            ref_embed.append(class_mean)
            ref_label.append(label)

        ref_embed = torch.tensor(np.array(ref_embed), dtype=torch.float16).to('cuda')
        ref_label = torch.tensor(np.array(ref_label), dtype=torch.int64).to('cuda')

        return ref_embed, ref_label

    def cluster(self, embeddings):
        features = embeddings.reshape(embeddings.shape[0], -1)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)

        labels = dbscan.fit_predict(features)
        self.label_list = labels
        self.label_set = sorted([label for label in set(labels) if label != -1])
        self.indices_dict = {label: [] for label in self.label_set}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            self.indices_dict[label].append(i)

        for i, seq_info in enumerate(self.seqs_info):
            seq_info[0] = labels[i]

        self.__calculate_accuracy()
        return self.__get_cluster_centers(embeddings, labels)