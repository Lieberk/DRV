## Abstract
In this work, we propose a novel Dual-Rationales Verification (DRV) framework that achieves robust detection performance through generating semantic and reasoning rationales. The framework employs semantic rationale to precisely identify key visual elements and potential manipulation traces, while utilizing reasoning rationale to quantify cross-modal logical contradictions, thereby emulating human fact-checking cognition. Through veracity-supervised signals, we effectively mitigate hallucinations in MLLM-generated content and transfer these high-quality rationales to lightweight student models via knowledge distillation. 

## Preparation

Start the scripts\pro_text_extract.py to process the dual-rationales features. Replace the llm_data.json file with llm_data\llm_fakett.json and llm_data\llm_fmnv.json.


## Dataset
We conduct experiments on two datasets: FMNV and FakeTT. 
- **FMNV**: 
Each sample in FMNV contains the video itself, its title, audio_transcript, subject, and false category. For the details, please refer to https://github.com/DennisIW/FMNV. 
- **FakeTT**:
The dataset is hosted on Baidu Netdisk. Link: https://pan.baidu.com/s/1uVZXbv3rHAQ9VuxjyfOAag?pwd=dqja  code: dqja


## Training the model
```
python train.py --batch_size 64 --max_epoch 20--seed 2026 --data_path dataset\FakeTT --lr 0.0005
```

## Citation
If you find this repo useful in your research works, please consider citing.
