[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](https://opensource.org/licenses/MIT)
[![Powered-By-Fast-AI](https://img.shields.io/badge/fastai%20v1.5.3%20%20-blueviolet.svg?logo=github)](https://github.com/fastai/fastai/tree/69231e6026b7fcbe5b67ab4eaa23d19be3ea0659)
[![Weights-And-Biases](https://img.shields.io/badge/Weights%20&%20Biases-black.svg?logo=google-analytics)](https://app.wandb.ai/github/issues_lang_model)

# A Language Model Trained On 16M+ GitHub Issues For Transfer Learning

**Motivation:**  [Issue Label Bot](https://github.com/machine-learning-apps/Issue-Label-Bot) predicts 3 generic issue labels: `bug`, `feature request` and `question`.  However, it would be nice to predict personalized issue labels instead of generic ones.  To accomplish this, we can use the issues that are already labeled in a repository as training data for a model that can predict personalized issue labels.  One challenge with this approach is there is often a small number of labeled issues in each repository.  In order to mitigate this concern, we train a self-supervised language model over 16 million GitHub issues and use this model as a feature extractor.  This method of [transfer-learning](http://nlp.fast.ai/) allows us to to build models on smaller datasets.

# End-Product: An API that returns embeddings from GitHub Issue Text.

The manifest files in [/deployment](/deployment) define a service that will return 2400 dimensional embeddings given the text of an issue.  The api endpoints are hosted on https://gh-issue-labeler.com/

All routes expect `POST` requests with a header containing a `Token` field. Below is  a list of endpoints:

1. `https://gh-issue-labeler.com/text`:  expects a json payload of `title` and `body` and returns a single 2,400 dimensional vector that represents latent features of the text. For example, this is how you would interact with this endpoint from python:

    ```python
    import requests
    import json
    import numpy as np
    from passlib.apps import custom_app_context as pwd_context

    API_ENDPOINT = 'https://gh-issue-labeler.com/text'
    API_KEY = 'YOUR_API_KEY' # Contact maintainers to get this

    # A toy example of a GitHub Issue title and body
    data = {'title': 'Fix the issue', 
            'body': 'I am encountering an error\n when trying to push the button.'}

    # sending post request and saving response as response object 
    r = requests.post(url=API_ENDPOINT,
                    headers={'Token':pwd_context.hash(API_KEY)},
                    json=data)

    # convert string back into a numpy array
    embeddings = np.frombuffer(r.content, dtype='<f4')
    ```



2. `https://gh-issue-labeler.com//all_issues/<owner>/<repo>` this will return a numpy array of the shape (# of labeled issues in repo, 2400), as well a list of all the labels for each issue.  This endpoint is still under construction.

# Training the Language Model

TODO

# Appendix: Location of Language Model Artifacts

### Google Cloud Storage

- **model for inference** (965 MB): https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/trained_model_22zkdqlr.pkl


- **encoder (for fine-tuning w/a classifier)** (965 MB): 
https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/trained_model_encoder_22zkdqlr.pth


- **fastai.databunch** (27.1 GB):
https://storage.googleapis.com/issue_label_bot/model/lang_model/data_save.pkl


- **checkpointed model** (2.29 GB): 
https://storage.googleapis.com/issue_label_bot/model/lang_model/models_22zkdqlr/best_22zkdqlr.pth

### Weights & Biases Run

https://app.wandb.ai/github/issues_lang_model/runs/22zkdqlr/overview
