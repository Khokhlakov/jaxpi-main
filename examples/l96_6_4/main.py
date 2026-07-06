import os

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import examples.l96_6_4.train as train
import examples.l96_6_4.eval as eval
import examples.l96_6_4.MLP_models as MLP_models
import examples.l96_6_4.eval_dd as eval_dd
import examples.l96_6_4.eval_pi_vs_dd as eval_pi_vs_dd


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_long":
        eval.evaluate_long(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_ekf":
        eval.evaluate_with_ekf(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_enkf":
        eval.evaluate_with_enkf(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_ekf_numerical":
        eval.evaluate_with_ekf_numerical(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_enkf_numerical":
        eval.evaluate_with_enkf_numerical(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "train_dd":
        train.train_and_evaluate_dd(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_dd":
        eval_dd.evaluate_dd(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_dd_enkf":
        eval_dd.evaluate_with_enkf_dd(FLAGS.config, FLAGS.workdir)

    # MLP:
    elif FLAGS.config.mode == "train_mlp":
        MLP_models.train_mlp(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_mlp":
        MLP_models.evaluate_mlp(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mlp_ekf":
        MLP_models.evaluate_mlp_with_ekf(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mlp_enkf":
        MLP_models.evaluate_mlp_with_enkf(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_ekf_numerical":
        eval.evaluate_with_ekf_numerical(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_enkf_numerical":
        eval.evaluate_with_enkf_numerical(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "train_dd":
        train.train_and_evaluate_dd(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_dd":
        eval_dd.evaluate_dd(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_dd_enkf":
        eval_dd.evaluate_with_enkf_dd(FLAGS.config, FLAGS.workdir)

    # MLP:
    elif FLAGS.config.mode == "train_mlp":
        MLP_models.train_mlp(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "eval_mlp":
        MLP_models.evaluate_mlp(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mlp_ekf":
        MLP_models.evaluate_mlp_with_ekf(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mlp_enkf":
        MLP_models.evaluate_mlp_with_enkf(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
