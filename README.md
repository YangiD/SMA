#### SMA Code

This is a demo code for the paper "[Beyond Score Changes: Adversarial Attack on No-Reference Image Quality Assessment from Two Perspectives](https://arxiv.org/abs/2404.13277)".

It contains the code for Stage One (generating target scores for all images) and Stage Two (generating the adversarial example for each image) in the paper.

#### Requirements

PyTorch
Python
scipy
pandas

#### Preliminary

1. Download the file named "livec_net_params_best.pkl" from [Google Drive](https://drive.google.com/file/d/1zMBRQKBeXyDJxmuwvhpJLcuzVHBoH2sV/view?usp=share_link) and move it into SMA_code/DBCNN/db_models folder.
2. Download the CLIVE dataset from [LIVE In the Wild Image Quality Challenge Database](https://live.ece.utexas.edu/research/ChallengeDB/index.html) and move the folder into SMA_code/CLIVE folder.

#### Usage with default setting

1. In Stage One, to attack DBCNN, run the following code to get the target score for each image:
   '''python
   python demo_attack_first_step.py
   '''
2. In Stage Two, run the following code to get the adversarial example attacking DBCNN for each image:
   '''python
   python demo_attack_second_step_DBCNN.py
   '''

#### Acknowledgement

1. https://github.com/zwx8981/DBCNN-PyTorch
2. https://github.com/google-research/fast-soft-sort
