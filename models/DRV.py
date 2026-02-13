from transformers.models.bart.modeling_bart import *
import torch.nn as nn
import torch
import torch.nn.functional as F




class CriterionBatchCrossPair(nn.Module):
    def __init__(self, temperature):
        super(CriterionBatchCrossPair, self).__init__()
        self.temperature = temperature

    def forward(self, feat_S, feat_T):
        feat_S = F.normalize(feat_S, p=2, dim=-1)
        feat_T = F.normalize(feat_T, p=2, dim=-1)

        s_sim_map = torch.mm(feat_S, feat_S.transpose(0, 1))
        t_sim_map = torch.mm(feat_T, feat_T.transpose(0, 1))

        p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_t = F.softmax(t_sim_map / self.temperature, dim=1)

        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
        return sim_dis


class DRV(nn.Module):
    def __init__(self, cfg):
        super(DRV, self).__init__()
        self.cfg = cfg
        self.kd_temperature = 1.0
        self.temperature = 0.05
        self.dim = 768
        self.model_gen = BartForConditionalGeneration.from_pretrained(".\dataset/Pretrain/bart-base")
        self.classifier = nn.Sequential(nn.Linear(self.dim, 128),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(128),
                                        nn.Linear(128, 2)
                                        )
        self.linear_frames = nn.Sequential(torch.nn.Linear(512, self.dim), torch.nn.ReLU(), nn.Dropout(p=0.1))
        self.BatchCrossPair = CriterionBatchCrossPair(temperature=self.temperature)
        self.p_explanation = nn.Linear(self.dim, self.dim)
        self.p_evidence = nn.Linear(self.dim, self.dim)

    def kl_divergence(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        kl_div = F.kl_div(p.log(), q, reduction='batchmean') * self.kd_temperature**2
        return kl_div

    def forward(self, mode='train', **kwargs):
        fea_title = kwargs['title']
        title_mask = kwargs['title_mask']

        frames_masks = kwargs['frames_masks']
        frames = kwargs['frames']
        fea_img = self.linear_frames(frames)

        audio_transcript = kwargs['audio_transcript']
        audio_transcript_mask = kwargs['audio_transcript_mask']

        concat_feat = torch.cat([fea_title, fea_img, audio_transcript], dim=1)
        concat_mask = torch.cat([title_mask, frames_masks, audio_transcript_mask], dim=1)

        context_enc_out = self.model_gen.get_encoder()(inputs_embeds=concat_feat, attention_mask=concat_mask)
        context_enc_feat = context_enc_out.last_hidden_state
        context_out = torch.mean(context_enc_feat, 1)

        context_explanation = self.p_explanation(context_out)
        context_evidence = self.p_evidence(context_out)

        if mode == 'train':
            prior_explanation = kwargs['prior_explanation']
            prior_video_evidence = kwargs['prior_video_evidence']

            context_out_explanation = torch.mean(prior_explanation, 1)
            s_feat = torch.div(context_explanation, self.temperature)
            t_feat = torch.div(context_out_explanation, self.temperature)
            kd_explanation_loss = self.kl_divergence(s_feat, t_feat)

            context_out_evidence = torch.mean(prior_video_evidence, 1)
            s_feat = torch.div(context_evidence, self.temperature)
            t_feat = torch.div(context_out_evidence, self.temperature)
            kd_evidence_loss = self.kl_divergence(s_feat, t_feat)

            kd_batch = self.BatchCrossPair(context_out, context_out_evidence + context_out_explanation)

            out_x = self.classifier(context_out)
            return out_x, kd_explanation_loss, kd_evidence_loss, kd_batch
        elif mode == 'eval':
            with torch.no_grad():
                out_x = self.classifier(context_out)
                return out_x
        else:
            raise ValueError('Mode should be among [train, eval].')
