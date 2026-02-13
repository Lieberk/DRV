import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1]) * 10000
    elif '亿' in str_x:
        return float(str_x[:-1]) * 100000000
    else:
        print("error")
        print(str_x)


class DRV_Dataset(Dataset):
    def __init__(self, path_vid, config):
        with open(os.path.join(config['data_path'], 'data.json'), 'r', encoding='utf-8') as f:
            self.data_complete = json.load(f)

        self.news_id = []
        with open(os.path.join(config['data_path'], 'data-split/', path_vid), "r") as fr:
            for line in fr.readlines():
                self.news_id.append(line.strip())
        self.data = [item for item in self.data_complete if str(item['video_id']) in self.news_id]
        self.text_fea_path = os.path.join(config['data_path'], 'preprocess/pre_DRV/bart_text_feature/')
        self.prior_llm_path = os.path.join(config['data_path'], 'preprocess/pre_DRV/llm_bart_feature/')
        self.frame_fea_path = os.path.join(config['data_path'], 'clip_vit_feature/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item['video_id']

        title_embed = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['title_embed']
        title_mask = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['title_mask']
        title_embed = torch.FloatTensor(title_embed)
        title_mask = torch.FloatTensor(title_mask)

        audio_transcript_embed = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['audio_transcript_embed']
        audio_transcript_mask = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['audio_transcript_mask']
        audio_transcript_embed = torch.FloatTensor(audio_transcript_embed)
        audio_transcript_mask = torch.FloatTensor(audio_transcript_mask)

        frames = pickle.load(open(os.path.join(self.frame_fea_path, vid + '.pkl'), 'rb'))
        frames = torch.FloatTensor(frames)

        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label)

        prior_explanation = pickle.load(open(os.path.join(self.prior_llm_path, vid + '.pkl'), 'rb'))['prior_explanation']
        prior_explanation_mask = pickle.load(open(os.path.join(self.prior_llm_path, vid + '.pkl'), 'rb'))['prior_explanation_mask']
        prior_explanation = torch.FloatTensor(prior_explanation)
        prior_explanation_mask = torch.FloatTensor(prior_explanation_mask)

        prior_video_evidence = pickle.load(open(os.path.join(self.prior_llm_path, vid + '.pkl'), 'rb'))['prior_video_evidence']
        prior_video_evidence_mask = pickle.load(open(os.path.join(self.prior_llm_path, vid + '.pkl'), 'rb'))['prior_video_evidence_mask']
        prior_video_evidence = torch.FloatTensor(prior_video_evidence)
        prior_video_evidence_mask = torch.FloatTensor(prior_video_evidence_mask)

        return {
            'label': label,
            'title': title_embed,
            'title_mask': title_mask,
            'audio_transcript': audio_transcript_embed,
            'audio_transcript_mask': audio_transcript_mask,
            'prior_explanation': prior_explanation,
            'prior_explanation_mask': prior_explanation_mask,
            'prior_video_evidence': prior_video_evidence,
            'prior_video_evidence_mask': prior_video_evidence_mask,
            'frames': frames,
        }


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            video = video[:seq_len]
            mask = np.ones(seq_len)
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.FloatTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def DRD_collate_fn(batch):
    title = [item['title'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    audio_transcript = [item['audio_transcript'] for item in batch]
    audio_transcript_mask = [item['audio_transcript_mask'] for item in batch]

    prior_explanation = [item['prior_explanation'] for item in batch]
    prior_explanation_mask = [item['prior_explanation_mask'] for item in batch]

    prior_video_evidence = [item['prior_video_evidence'] for item in batch]
    prior_video_evidence_mask = [item['prior_video_evidence_mask'] for item in batch]

    label = [item['label'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(55, frames)

    return {
        'label': torch.stack(label),
        'title': torch.stack(title),
        'title_mask': torch.stack(title_mask),
        'audio_transcript': torch.stack(audio_transcript),
        'audio_transcript_mask': torch.stack(audio_transcript_mask),
        'prior_explanation': torch.stack(prior_explanation),
        'prior_explanation_mask': torch.stack(prior_explanation_mask),
        'prior_video_evidence': torch.stack(prior_video_evidence),
        'prior_video_evidence_mask': torch.stack(prior_video_evidence_mask),
        'frames': frames,
        'frames_masks': frames_masks,
    }