This is implementation for the paper "Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus" accepted by NAACL 2019.

* Add folder
Create folder data/, dump/, model/ and pretrained_model/ in the same level of src/


* Data prep
Put data to data_folder
- data_folder: ../data/[data_type]/
- data_type: yelp/gyafc_family
- Put train/dev/test corpus in original/target style as corpus.(train/dev/test).(orig/tsf)
Put pretrained embedding in text format to the path of embed_fn


* Running instructions
1. Data processing
yelp data:
python3 corpus_helper.py --data DATA --vec_dim 100 --embed_fn
gyafc_family data:
python3 corpus_helper.py --data DATA --tokenize --vec_dim 100 --embed_fn 

- DATA: gyafc_family / yelp
- "--tokenize" only for gyafc_family data
- data path: ../data/DATA/corpus.(train/test).(orig/tsf)
- save path: pkl is saved to ../dump/DATA/, pkl files are (train/test)_(orig/tsf).pkl, tuned embedding is saved to tune_vec.txt

2. python3 style_transfer_rl.py -- data DATA
- DATA: gyafc_family / yelp

yelp data pretrain
CUDA_VISIBLE_DEVICES=0,1 python3 style_transfer_RL.py --data_type yelp --max_sent_len 18 --lm_seq_length 18 --lm_epochs 5 --style_epochs 1 --pretrain_epochs 2 --beam_width 1
--pretrained_model_path best_pretrained_model 
--batch_size 32

yelp data RL
CUDA_VISIBLE_DEVICES=0,1 python3 style_transfer_RL.py --data_type yelp
--max_sent_len 18 --lm_seq_length 18 
--use_pretrained_model --pretrained_model_path best_pretrained_model
--rollout_num 2 --beam_width 1 
--rl_learning_rate 1e-6 --batch_size 16 --epochs 1 


gyafc_family pretrain
CUDA_VISIBLE_DEVICES=0,1 python3 style_transfer_RL.py --data_type gyafc_family --max_sent_len 30  --lm_seq_length 30 --lm_epochs 8 --style_epochs 3 --pretrain_epochs 4 --beam_width 1
--pretrained_model_path best_pretrained_model 
--batch_size 32 

gyafc_family RL
CUDA_VISIBLE_DEVICES=2,3 python3 style_transfer_RL.py --data_type yelp
--max_sent_len 30 --lm_seq_length 30
--use_pretrained_model --pretrained_model_path best_pretrained_model
--rollout_num 2 --beam_width 1 
--rl_learning_rate 1e-6 --batch_size 16 --epochs 1 

3. Test
yelp data
CUDA_VISIBLE_DEVICES=2 python3 style_transfer_test.py --data_type yelp
--max_sent_len 18 --lm_seq_length 18 --use_beamsearch_decode --beam_width 1
--model_path MODEL_PATH --output_path OUTPUT_PATH
--batch_size 32

- MODEL_PATH: ../model/[DATA_TYPE]/model
- OUTPUT_PATH: the path where transferred sentences are saved

If you're considering using our code, please cite our paper:

@article{gong2019reinforcement,
  title={Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus},
  author={Gong, Hongyu and Bhat, Suma and Wu, Lingfei and Xiong, Jinjun and Hwu, Wen-mei},
  journal={arXiv preprint arXiv:1903.10671},
  year={2019}
}

Gong H, Bhat S, Wu L, Xiong J, Hwu WM. Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus. arXiv preprint arXiv:1903.10671. 2019 Mar 26.

