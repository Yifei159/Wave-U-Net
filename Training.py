from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os

import Datasets
import Utils
import Models.UnetSpectrogramSeparator
import Models.UnetAudioSeparator
import Test
import Evaluate

from tqdm import tqdm  # adding the bar
import functools
from tensorflow.contrib.signal.python.ops import window_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ex = Experiment('Waveunet Training', ingredients=[config_ingredient])

@ex.config
def set_seed():
    seed = 1337

@config_ingredient.capture
def train(model_config, experiment_id, load_model=None):
    # setting the model
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError

    # calculate input/output shape with get_padding()
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]
    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))

    # put into dataset
    dataset = Datasets.get_dataset(model_config, sep_input_shape, sep_output_shape, partition="train")
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    print("Training...")

    # fix the model
    separator_sources = separator_class.get_output(batch["mix"], True, not model_config["raw_audio_loss"], reuse=False)

    separator_loss = 0
    for key in model_config["source_names"]:
        real_source = batch[key]
        sep_source = separator_sources[key]

        if model_config["network"] == "unet_spectrogram" and not model_config["raw_audio_loss"]:
            window = functools.partial(window_ops.hann_window, periodic=True)
            stfts = tf.contrib.signal.stft(tf.squeeze(real_source, 2), frame_length=1024, frame_step=768,
                                           fft_length=1024, window_fn=window)
            real_mag = tf.abs(stfts)
            separator_loss += tf.reduce_mean(tf.abs(real_mag - sep_source))
        else:
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))

    separator_loss = separator_loss / float(model_config["num_sources"]) 

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)

    separator_vars = Utils.getTrainableVariables("separator")
    print("Sep_Vars: " + str(Utils.getNumParams(separator_vars)))
    print("Num of variables: " + str(len(tf.global_variables())))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=model_config["init_sup_sep_lr"]).minimize(separator_loss, var_list=separator_vars)

    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_config["log_dir"] + os.path.sep + str(experiment_id), graph=sess.graph)

    if load_model is not None:
        restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        restorer.restore(sess, load_model)
        print('Pre-trained model restored from file ' + load_model)

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    _global_step = sess.run(global_step)
    _init_step = _global_step

    # the epoch bar
    num_epochs = model_config["epoch_it"]
    progress_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

    for _ in progress_bar:
        _, _sup_summaries, loss_val = sess.run([separator_solver, sup_summaries, separator_loss])
        writer.add_summary(_sup_summaries, global_step=_global_step)
        _global_step = sess.run(increment_global_step)

        progress_bar.set_postfix(loss=loss_val)  # update loss

    print("Finished training!")
    save_path = saver.save(sess, model_config["model_base_dir"] + os.path.sep + str(experiment_id) + os.path.sep + str(experiment_id), global_step=int(_global_step))

    writer.flush()
    writer.close()
    sess.close()
    tf.reset_default_graph()

    return save_path

@config_ingredient.capture
def optimise(model_config, experiment_id):
    epoch = 0
    best_loss = 10000
    model_path = None
    best_model_path = None

    # adding the bar
    optim_progress = tqdm(range(2), desc="Optimizing", unit="stage")

    for i in optim_progress:
        worse_epochs = 0
        if i == 1:
            print("Finished first round of training, now entering fine-tuning stage")
            model_config["batch_size"] *= 2
            model_config["init_sup_sep_lr"] = 1e-5

        # epoches
        epoch_progress = tqdm(total=model_config["worse_epochs"], desc=f"Stage {i+1} - Training", unit="epoch")

        while worse_epochs < model_config["worse_epochs"]:
            model_path = train(load_model=model_path)
            curr_loss = Test.test(model_config, model_folder=str(experiment_id), partition="valid", load_model=model_path)
            epoch += 1

            if curr_loss < best_loss:
                worse_epochs = 0
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1

            epoch_progress.update(1)
            epoch_progress.set_postfix(loss=curr_loss, best_loss=best_loss)

        epoch_progress.close()

    print("TRAINING FINISHED - TESTING WITH BEST MODEL " + best_model_path)
    test_loss = Test.test(model_config, model_folder=str(experiment_id), partition="test", load_model=best_model_path)
    return best_model_path, test_loss

@ex.automain
def run(cfg):
    model_config = cfg["model_config"]
    print("SCRIPT START")
    for dir in [model_config["model_base_dir"], model_config["log_dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # the bar
    with tqdm(total=1, desc="Full Training Process", unit="task") as total_progress:
        sup_model_path, sup_loss = optimise()
        total_progress.update(1)

    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))
    Evaluate.produce_musdb_source_estimates(model_config, sup_model_path, model_config["musdb_path"], model_config["estimates_path"])
