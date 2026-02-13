import os
import pickle
from transformers.models.bart.modeling_bart import *
import json
from transformers import BartTokenizer
import torch

data_path = '.\dataset\FakeTT'
pretrain_path = '.\dataset/Pretrain'

gpu_id = "cuda:0"
device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')

bart_model_path = os.path.join(pretrain_path, 'bart-base')
tkr = BartTokenizer.from_pretrained(bart_model_path)
model_gen = BartForConditionalGeneration.from_pretrained(bart_model_path).to(device)


def text_feature_extraction():
    with open(os.path.join(data_path, 'data.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    bert_text_path = os.path.join(data_path, 'preprocess/pre_DRV/bart_text_feature')
    if not os.path.exists(bert_text_path):
        os.makedirs(bert_text_path)

    max_title_l = 32
    max_audio_l = 64

    for i, item in enumerate(data):
        video_id = item['video_id']

        title_text = item['description']
        title_tokens = tkr(title_text,
                           is_split_into_words=True,
                           max_length=max_title_l,
                           padding='max_length',
                           truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids']).unsqueeze(0).to(device)
        title_mask = torch.LongTensor(title_tokens['attention_mask']).unsqueeze(0).to(device)
        title_embed = model_gen.get_input_embeddings()(title_inputid).to(device)

        audio_transcript = item['audio_transcript']
        audio_transcript_tokens = tkr(audio_transcript,
                                      is_split_into_words=True,
                                      max_length=max_audio_l,
                                      padding='max_length',
                                      truncation=True)
        audio_transcript_inputid = torch.LongTensor(audio_transcript_tokens['input_ids']).unsqueeze(0).to(device)
        audio_transcript_mask = torch.LongTensor(audio_transcript_tokens['attention_mask']).unsqueeze(0).to(device)
        audio_transcript_embed = model_gen.get_input_embeddings()(audio_transcript_inputid).to(device)

        data_dict = {'title_embed': title_embed.squeeze(0).detach().cpu().numpy(),
                     'title_mask': title_mask.squeeze(0).detach().cpu().numpy(),
                     'audio_transcript_embed': audio_transcript_embed.squeeze(0).detach().cpu().numpy(),
                     'audio_transcript_mask': audio_transcript_mask.squeeze(0).detach().cpu().numpy(),
                     }

        with open(os.path.join(bert_text_path, '{}.pkl'.format(video_id)), 'wb') as f:
            pickle.dump(data_dict, f)
        print('text processing %s' % video_id)


def llm_data_extraction():
    with open(os.path.join(data_path, 'llm_data.json'), 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    max_explanation_l = 64
    max_evidence_l = 80
    prior_llm_data_path = os.path.join(data_path, 'preprocess/pre_DRV/llm_bart_feature')
    if not os.path.exists(prior_llm_data_path):
        os.makedirs(prior_llm_data_path)

    for i, item in enumerate(llm_data):
        video_id = item['video_id']

        prior_explanation = llm_data[i]['reasoning_rationales']
        prior_explanation_tokens = tkr(prior_explanation,
                                       is_split_into_words=True,
                                       max_length=max_explanation_l,
                                       padding='max_length',
                                       truncation=True)
        prior_explanation_inputid = torch.LongTensor(prior_explanation_tokens['input_ids']).unsqueeze(0).to(device)
        prior_explanation_mask = torch.LongTensor(prior_explanation_tokens['attention_mask']).unsqueeze(0).to(device)
        prior_explanation_embed = model_gen.get_input_embeddings()(prior_explanation_inputid).to(device)
        prior_explanation_out = model_gen.get_encoder()(inputs_embeds=prior_explanation_embed, attention_mask=prior_explanation_mask)
        prior_explanation_embed = prior_explanation_out.last_hidden_state

        prior_video_evidence = llm_data[i]['semantic_rationales']
        prior_video_evidence_tokens = tkr(prior_video_evidence,
                                          is_split_into_words=True,
                                          max_length=max_evidence_l,
                                          padding='max_length',
                                          truncation=True)
        prior_video_evidence_inputid = torch.LongTensor(prior_video_evidence_tokens['input_ids']).unsqueeze(0).to(device)
        prior_video_evidence_mask = torch.LongTensor(prior_video_evidence_tokens['attention_mask']).unsqueeze(0).to(device)
        prior_video_evidence_embed = model_gen.get_input_embeddings()(prior_video_evidence_inputid).to(device)
        prior_video_evidence_enc_out = model_gen.get_encoder()(inputs_embeds=prior_video_evidence_embed, attention_mask=prior_video_evidence_mask)
        prior_video_evidence_embed = prior_video_evidence_enc_out.last_hidden_state

        data_dict = {
            'prior_video_evidence': prior_video_evidence_embed.squeeze(0).detach().cpu().numpy(),
            'prior_video_evidence_mask': prior_video_evidence_mask.squeeze(0).detach().cpu().numpy(),
            'prior_explanation': prior_explanation_embed.squeeze(0).detach().cpu().numpy(),
            'prior_explanation_mask': prior_explanation_mask.squeeze(0).detach().cpu().numpy(),
        }
        with open(os.path.join(prior_llm_data_path, '{}.pkl'.format(video_id)), 'wb') as f:
            pickle.dump(data_dict, f)
        print('text processing %s' % video_id)


if __name__ == '__main__':
    text_feature_extraction()
    llm_data_extraction()
