[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](https://opensource.org/licenses/MIT)
[![Powered-By-Fast-AI](https://img.shields.io/badge/fastai%20v1.5.3%20%20-blueviolet.svg?logo=github)](https://github.com/fastai/fastai/tree/69231e6026b7fcbe5b67ab4eaa23d19be3ea0659)
[![Weights-And-Biases](https://img.shields.io/badge/Weights%20&%20Biases-black.svg?logo=google-analytics)](https://app.wandb.ai/github/issues_lang_model)

# A Language Model Trained On 16M+ GitHub Issues For Transfer Learning

TODO

# Appendix: Location of Model Artifacts For Best Language Model

### Google Cloud Storage

- **model for inference** (965 MB): `https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/trained_model_22zkdqlr.pkl`


- **encoder (for fine-tuning w/a classifier)** (965 MB): 
`https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/trained_model_encoder_22zkdqlr.pth`


- **fastai.databunch** (27.1 GB):
`https://storage.googleapis.com/issue_label_bot/model/lang_model/data_save.pkl`


- **checkpointed model** (2.29 GB): 
`https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/best_22zkdqlr.pth`

### Weights & Biases Run

`https://app.wandb.ai/github/issues_lang_model/runs/22zkdqlr/overview`