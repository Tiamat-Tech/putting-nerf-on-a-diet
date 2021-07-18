import os
import flax
from jax import random
from flax.training import checkpoints

from jaxnerf.nerf import models
from jaxnerf.nerf import utils
from app.src import config


rng = random.PRNGKey(0)
dummy_batch = None
dummy_lr = 1e-2


def load_trained_model(model_dir, model_fn):
    _, key = random.split(rng)
    model, init_variables = models.get_model(key, dummy_batch, config.NerfConfig)
    optimizer = flax.optim.Adam(dummy_lr).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables
    assert os.path.isfile(os.path.join(model_dir, model_fn))
    state = checkpoints.restore_checkpoint(model_dir, state, prefix=model_fn)
    print(f'Trained model loaded: {model_dir}/{model_fn}')
    return state


if __name__ == '__main__':
    _model_dir = '../ship_fewshot_wsc'
    _model_fn = 'checkpoint_345000'
    _state = load_trained_model(_model_dir, _model_fn)
