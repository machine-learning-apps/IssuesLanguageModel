from fastai.text import TextLMDataBunch as lmdb, load_data
from fastai.text.learner import language_model_learner
from fastai.text.models import AWD_LSTM
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, ReduceLROnPlateauCallback, CSVLogger
from fastai.text.models import AWD_LSTM
from fastai.train import ShowGraph
import pandas as pd
from pathlib import Path
from fastai.distributed import *

path = Path('lang_model/')

def pass_through(x):
    return x

data_lm = load_data(path, bs=128)

learn = language_model_learner(data=data_lm,
                               arch=AWD_LSTM,
                               pretrained=False)

# callbacks
escb = EarlyStoppingCallback(learn=learn, patience=4)
smcb = SaveModelCallback(learn=learn)
rpcb = ReduceLROnPlateauCallback(learn=learn, patience=3)
csvcb = CSVLogger(learn=learn)
callbacks = [escb, smcb, rpcb, csvcb]

learn.to_parallel()

learn.fit_one_cycle(cyc_len=1,
                    max_lr=1e-2*3,
                    tot_epochs=15,
                    callbacks=callbacks)