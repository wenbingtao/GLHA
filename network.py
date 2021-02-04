


import os
import sys

import numpy as np
import tensorflow as tf
from parse import parse
from tqdm import trange

from ops import tf_skew_symmetric
from tests import comp_process, test_process, my_test_process


class MyNetwork(object):
    """Network class """

    def __init__(self, config):

        self.config = config

        # Initialize thenosrflow session
        self._init_tensorflow()

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _init_tensorflow(self):
        # limit CPU threads with OMP_NUM_THREADS
        num_threads = os.getenv("OMP_NUM_THREADS", "")
        if num_threads != "":
            num_threads = int(num_threads)
            print("limiting tensorflow to {} threads!".format(
                num_threads
            ))
            # Limit
            tfconfig = tf.ConfigProto(
                intra_op_parallelism_threads=num_threads,
                inter_op_parallelism_threads=num_threads,
            )
        else:
            tfconfig = tf.ConfigProto()

        tfconfig.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tfconfig)

    def _build_placeholder(self):
        """Build placeholders."""

        # Make tensforflow placeholder
        self.x_in = tf.placeholder(tf.float32, [None, 1, None, 4], name="x_in")
        self.y_in = tf.placeholder(tf.float32, [None, None, 2], name="y_in")
        self.R_in = tf.placeholder(tf.float32, [None, 9], name="R_in")
        self.t_in = tf.placeholder(tf.float32, [None, 3], name="t_in")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")

        # Global step for optimization
        self.global_step = tf.get_variable(
            "global_step", shape=(),
            initializer=tf.zeros_initializer(),
            dtype=tf.int64,
            trainable=False)

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        # For now, do nothing
        pass

    def _build_model(self):
        """Build our MLP network."""

        with tf.variable_scope("Matchnet", reuse=tf.AUTO_REUSE):
            # For determining the runtime shape
            x_shp = tf.shape(self.x_in)

            # -------------------- Network archintecture --------------------
            # Import correct build_graph function
            from archs.arch import build_graph
            # Build graph
            print("Building Graph")
            self.logits, self.logits_pre, self.logits_pre2 = build_graph(self.x_in, self.is_training, self.config)
            # self.logits, self.logits_pre = build_graph(self.x_in, self.is_training, self.config)

            # self.logits = build_graph(self.x_in, self.is_training, self.config)
            # ---------------------------------------------------------------

            # Turn into weights for each sample
            weights = tf.nn.relu(tf.tanh(self.logits))

            self.weights = weights

            # Make input data (num_img_pair x num_corr x 4)
            xx = tf.transpose(tf.reshape(
                self.x_in, (x_shp[0], x_shp[2], 4)), (0, 2, 1))

            # Create the matrix to be used for the eight-point algorithm
            X = tf.transpose(tf.stack([
                xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
                xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
                xx[:, 0], xx[:, 1], tf.ones_like(xx[:, 0])
            ], axis=1), (0, 2, 1))
            print("X shape = {}".format(X.shape))
            wX = tf.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
            print("wX shape = {}".format(wX.shape))
            XwX = tf.matmul(tf.transpose(X, (0, 2, 1)), wX)
            print("XwX shape = {}".format(XwX.shape))

            # Recover essential matrix from self-adjoing eigen
            e, v = tf.self_adjoint_eig(XwX)
            self.e_hat = tf.reshape(v[:, :, 0], (x_shp[0], 9))
            # Make unit norm just in case
            self.e_hat /= tf.norm(self.e_hat, axis=1, keep_dims=True)


    def obtain_classification_loss(self, is_pos, is_neg, logit, beta):
        num_pos = tf.to_float(tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0)
        num_neg = tf.to_float(tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0)

        is_regard_pos = tf.to_float(
            logit >= 0
        )
        is_regard_neg = tf.to_float(
            logit < 0
        )

        is_true_pos = is_regard_pos * is_pos

        is_TP = is_true_pos
        is_TN = is_regard_neg * is_neg
        is_FP = is_regard_pos * is_neg
        is_FN = is_regard_neg * is_pos

        num_TP = tf.to_float(tf.reduce_sum(is_TP, axis=1))
        num_TN = tf.to_float(tf.reduce_sum(is_TN, axis=1))
        num_FP = tf.to_float(tf.reduce_sum(is_FP, axis=1))
        num_FN = tf.to_float(tf.reduce_sum(is_FN, axis=1))

        num_TP_relu = tf.to_float(tf.nn.relu(tf.reduce_sum(is_TP, axis=1) - 1.0) + 1.0)

        num_regard_pos = tf.to_float(tf.nn.relu(tf.reduce_sum(is_regard_pos, axis=1) - 1.0) + 1.0)

        # num_true_pos = tf.nn.relu(tf.reduce_sum(is_true_pos, axis=1) -1.0) + 1.0

        num_true_pos = tf.reduce_sum(is_true_pos, axis=1)

        pre_precision = num_TP_relu / tf.to_float(num_regard_pos)

        pre_recall = num_TP_relu / tf.to_float(num_pos)

        pre_F1 = (1 + beta * beta) * pre_precision * pre_recall / (beta * beta * pre_precision + pre_recall)

        cur_recall = tf.to_float(num_TP_relu + 1) / tf.to_float(num_pos)

        cur_precisioin = tf.to_float(num_TP_relu + 1) / tf.to_float(num_regard_pos + 1)

        cur_F1 = (1 + beta * beta) * cur_recall * cur_precisioin / (cur_recall + beta * beta * cur_precisioin)

        cur_precision2 = tf.to_float(num_true_pos) / tf.to_float(num_regard_pos + 1)

        cur_F1_2 = (1 + beta * beta) * pre_recall * cur_precision2 / (pre_recall + beta * beta * cur_precision2)

        d_F1_X = cur_F1 - pre_F1
        d_F1_Y = tf.clip_by_value(pre_F1 - cur_F1_2, 1e-6, 1.0)

        sig_logits = tf.nn.sigmoid(self.logits)

        classif_loss_p_array = -tf.log(tf.clip_by_value(sig_logits, 1e-10, 1.0)) * is_pos

        classif_loss_n_array = -tf.log(tf.clip_by_value(1 - sig_logits, 1e-10, 1.0)) * is_neg

        loss_TP = tf.clip_by_value(
            tf.reduce_sum(sig_logits * is_TP) / (tf.nn.relu(tf.reduce_sum(num_TP) - 1.0) + 1.0), 1e-10, 1.0)
        loss_TN = tf.clip_by_value(
            1 - tf.reduce_sum(sig_logits * is_TN) / (tf.nn.relu(tf.reduce_sum(num_TN) - 1.0) + 1.0), 1e-10, 1.0)
        loss_FP = tf.clip_by_value(
            1 - tf.reduce_sum(sig_logits * is_FP) / (tf.nn.relu(tf.reduce_sum(num_FP) - 1.0) + 1.0), 1e-10, 1.0)
        loss_FN = tf.clip_by_value(
            tf.reduce_sum(sig_logits * is_FN) / (tf.nn.relu(tf.reduce_sum(num_FN) - 1.0) + 1.0), 1e-10, 1.0)

        dif_pos = -tf.log(loss_FN) + tf.log(loss_TP)
        dif_neg = -tf.log(loss_FP) + tf.log(loss_TN)

        is_not_zero = tf.to_float((loss_TP > 0.5) & (loss_TN > 0.5) & (loss_FP < 0.5) & (loss_FN < 0.5))

        weight = tf.stop_gradient(1 - is_not_zero + is_not_zero * dif_neg / dif_pos)

        cur_weight = (d_F1_X / d_F1_Y) * (num_pos / num_neg)  # * weight

        cur_weight_p = cur_weight / (1 + cur_weight)
        cur_weight_n = 1 - cur_weight_p

        classif_loss_p = tf.reduce_sum(
            classif_loss_p_array, axis=1
        )
        classif_loss_n = tf.reduce_sum(
            classif_loss_n_array, axis=1
        )
        # cur_weight_p = 0.5
        # cur_weight_n = 0.5

        classif_loss = tf.reduce_mean(
            cur_weight_p * classif_loss_p / num_pos + cur_weight_n * classif_loss_n / num_neg
        )
        return classif_loss

    def _build_loss(self):

        # """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            x_shp = tf.shape(self.x_in)
            # The groundtruth epi sqr
            gt_geod_d = self.y_in[:, :, 0]
            # tf.summary.histogram("gt_geod_d", gt_geod_d)

            # Get groundtruth Essential matrix
            e_gt_unnorm = tf.reshape(tf.matmul(
                tf.reshape(tf_skew_symmetric(self.t_in), (x_shp[0], 3, 3)),
                tf.reshape(self.R_in, (x_shp[0], 3, 3))
            ), (x_shp[0], 9))
            e_gt = e_gt_unnorm / tf.norm(e_gt_unnorm, axis=1, keep_dims=True)

            # e_hat = tf.reshape(tf.matmul(
            #     tf.reshape(t_hat, (-1, 3, 3)),
            #     tf.reshape(r_hat, (-1, 3, 3))
            # ), (-1, 9))

            # Essential matrix loss
            essential_loss = tf.reduce_mean(tf.minimum(
                tf.reduce_sum(tf.square(self.e_hat - e_gt), axis=1),
                tf.reduce_sum(tf.square(self.e_hat + e_gt), axis=1)
            ))
            tf.summary.scalar("essential_loss", essential_loss)

            # Classification loss
            is_pos = tf.to_float(
                gt_geod_d < self.config.obj_geod_th
            )
            is_neg = tf.to_float(
                gt_geod_d >= self.config.obj_geod_th
            )
            c = is_pos - is_neg
            num_pos = tf.to_float(tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0)
            num_neg = tf.to_float(tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0)


            classif_loss_pre = self.obtain_classification_loss(is_pos, is_neg, self.logits_pre, 3)
            classif_loss_pre2 = self.obtain_classification_loss(is_pos, is_neg, self.logits_pre2, 2.5)
            classif_loss = self.obtain_classification_loss(is_pos, is_neg, self.logits, 2)

            tf.summary.scalar("classif_loss", classif_loss)
            tf.summary.scalar("classif_loss_pre", classif_loss_pre)
            tf.summary.scalar("classif_loss_pre2", classif_loss_pre2)
            # tf.summary.scalar("classif_loss_pre3", classif_loss_pre3)


            is_regard_pos = tf.to_float(
                self.logits >= 0
            )
            is_regard_neg = tf.to_float(
                self.logits < 0
            )

            is_true_pos = is_regard_pos * is_pos

            num_regard_pos = tf.to_float(tf.nn.relu(tf.reduce_sum(is_regard_pos, axis=1) - 1.0) + 1.0)

            # num_true_pos = tf.nn.relu(tf.reduce_sum(is_true_pos, axis=1) -1.0) + 1.0

            num_true_pos = tf.reduce_sum(is_true_pos, axis=1)

            precision = tf.reduce_mean(
                num_true_pos /
                num_regard_pos
            )

            tf.summary.scalar("precision", precision)
            # recall = tf.reduce_mean(
            #     tf.reduce_sum(tf.to_float(self.logits > 0) * is_pos, axis=1) /
            #     tf.reduce_sum(is_pos, axis=1)
            # )

            recall = tf.reduce_mean(
                num_true_pos /
                num_pos
            )
            tf.summary.scalar("recall", recall)

            # L2 loss
            for var in tf.trainable_variables():
                if "weights" in var.name:
                    tf.add_to_collection("l2_losses", tf.reduce_sum(var ** 2))
            l2_loss = tf.add_n(tf.get_collection("l2_losses"))
            tf.summary.scalar("l2_loss", l2_loss)



            # Check global_step and add essential loss
            self.loss = self.config.loss_decay * l2_loss
            if self.config.loss_essential > 0:
                self.loss += (
                        self.config.loss_essential * essential_loss * tf.to_float(
                    self.global_step >= tf.to_int64(
                        self.config.loss_essential_init_iter)))
            if self.config.loss_classif > 0:
                # self.loss += self.config.loss_classif * classif_loss
                # self.loss += 0.3 * classif_loss_pre
                # # # self.loss += 0.1 * classif_loss_pre2

                self.loss += self.config.loss_classif * classif_loss
                self.loss += 0.1 * classif_loss_pre
                self.loss += 0.1 * classif_loss_pre2

            tf.summary.scalar("loss", self.loss)

    def get_learning_rate(self,base_learn_rate, batch_iter):
        learning_rate = tf.train.exponential_decay(
            base_learn_rate,
            batch_iter,
            4000,
            0.9,
            staircase=True
        )
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optimization", reuse=tf.AUTO_REUSE):
            learning_rate = self.config.train_lr

            # learning_rate = self.get_learning_rate(self.config.train_lr, self.global_step)

            max_grad_norm = None
            optim = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optim.compute_gradients(self.loss)

                # gradient clipping
                if max_grad_norm is not None:
                    new_grads_and_vars = []
                    for idx, (grad, var) in enumerate(grads_and_vars):
                        if grad is not None:
                            new_grads_and_vars.append((
                                tf.clip_by_norm(grad, max_grad_norm), var))
                    grads_and_vars = new_grads_and_vars

                # Check numerics and report if something is going on. This
                # will make the backward pass stop and skip the batch
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}"
                            "".format(var.name))
                    new_grads_and_vars.append((grad, var))

                # Should only apply grads once they are safe
                self.optim = optim.apply_gradients(
                    new_grads_and_vars, global_step=self.global_step)

            # # Summarize all gradients
            # for grad, var in grads_and_vars:
            #     if grad is not None:
            #         tf.summary.histogram(var.name + '/gradient', grad)

    def _build_eval(self):
        """Build the evaluation related ops"""

        # We use a custom evaluate function. No building here...
        pass

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create suffix automatically if not provided
        suffix_tr = self.config.log_dir
        if suffix_tr == "":
            suffix_tr = "-".join(sys.argv)
        suffix_te = self.config.test_log_dir
        if suffix_te == "":
            suffix_te = suffix_tr

        # Directories for train/test
        self.res_dir_tr = os.path.join(self.config.res_dir, suffix_tr)
        self.res_dir_va = os.path.join(self.config.res_dir, suffix_te)
        self.res_dir_te = os.path.join(self.config.res_dir, suffix_te)

        # Create summary writers
        if self.config.run_mode == "train":
            self.summary_tr = tf.summary.FileWriter(
                os.path.join(self.res_dir_tr, "train", "logs"))
        if self.config.run_mode != "comp":
            self.summary_va = tf.summary.FileWriter(
                os.path.join(self.res_dir_va, "valid", "logs"))
        if self.config.run_mode == "test":
            self.summary_te = tf.summary.FileWriter(
                os.path.join(self.res_dir_te, "test", "logs"))

        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.res_dir_tr, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.res_dir_tr, "models-best")

        # Other savers
        self.va_res_file = os.path.join(self.res_dir_va, "valid", "va_res.txt")

    def train(self, data):
        """Training function.

        Parameters
        ----------
        data_tr : tuple
            Training data.

        data_va : tuple
            Validation data.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        print("Initializing...")
        self.sess.run(tf.global_variables_initializer())
        
        # ----------------------------------------
        # Resume data if it already exists
        latest_checkpoint = tf.train.latest_checkpoint(
            self.res_dir_tr)
        b_resume = latest_checkpoint is not None
        if b_resume:
            # Restore network
            print("Restoring from {}...".format(
                self.res_dir_tr))
            self.saver_cur.restore(
                self.sess,
                latest_checkpoint
            )
            # restore number of steps so far
            step = self.sess.run(self.global_step)
            # restore best validation result
            if os.path.exists(self.va_res_file):
                with open(self.va_res_file, "r") as ifp:
                    dump_res = ifp.read()
                dump_res = parse(
                    "{best_va_res:e}\n", dump_res)
                best_va_res = dump_res["best_va_res"]
        else:
            print("Starting from scratch...")
            step = 0
            best_va_res = -1

        # ----------------------------------------
        # Unpack some data for simple coding
        xs_tr = data["train"]["xs"]
        ys_tr = data["train"]["ys"]
        Rs_tr = data["train"]["Rs"]
        ts_tr = data["train"]["ts"]

        # ----------------------------------------
        # The training loop
        batch_size = self.config.train_batch_size
        max_iter = self.config.train_iter
        for step in trange(step, max_iter, ncols=self.config.tqdm_width):

            # ----------------------------------------
            # Batch construction

            # Get a random training batch
            ind_cur = np.random.choice(
                len(xs_tr), batch_size, replace=False)
            # Use minimum kp in batch to construct the batch
            numkps = np.array([xs_tr[_i].shape[1] for _i in ind_cur])
            cur_num_kp = numkps.min()
            # Actual construction of the batch
            xs_b = np.array(
                [xs_tr[_i][:, :cur_num_kp, :] for _i in ind_cur]
            ).reshape(batch_size, 1, cur_num_kp, 4)
            ys_b = np.array(
                [ys_tr[_i][:cur_num_kp, :] for _i in ind_cur]
            ).reshape(batch_size, cur_num_kp, 2)
            Rs_b = np.array(
                [Rs_tr[_i] for _i in ind_cur]
            ).reshape(batch_size, 9)
            ts_b = np.array(
                [ts_tr[_i] for _i in ind_cur]
            ).reshape(batch_size, 3)

            # ----------------------------------------
            # Train

            # Feed Dict
            feed_dict = {
                self.x_in: xs_b,
                self.y_in: ys_b,
                self.R_in: Rs_b,
                self.t_in: ts_b,
                self.is_training: True,
            }
            # Fetch
            fetch = {
                "optim": self.optim,
            }
            # Check if we want to write summary and check validation
            b_write_summary = ((step + 1) % self.config.report_intv) == 0
            b_validate = ((step + 1) % self.config.val_intv) == 0
            if b_write_summary or b_validate:
                fetch["summary"] = self.summary_op
                fetch["global_step"] = self.global_step
            # Run optimization
            try:
                res = self.sess.run(fetch, feed_dict=feed_dict)
            except (ValueError, tf.errors.InvalidArgumentError):
                print("Backward pass had numerical errors. "
                      "This training batch is skipped!")
                continue
            # Write summary and save current model
            if b_write_summary:
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"])
                self.saver_cur.save(
                    self.sess, self.save_file_cur,
                    global_step=self.global_step,
                    write_meta_graph=False)

            # ----------------------------------------
            # Validation
            if b_validate:
                va_res = 0
                cur_global_step = res["global_step"]
                va_res = test_process(
                    "valid", self.sess, cur_global_step,
                    self.summary_op, self.summary_va,
                    self.x_in, self.y_in, self.R_in, self.t_in,
                    self.is_training,
                    None, None, None,
                    self.logits, self.e_hat, self.loss,
                    data["valid"],
                    self.res_dir_va, self.config, True)
                # Higher the better
                if va_res > best_va_res:
                    print(
                        "Saving best model with va_res = {}".format(
                            va_res))
                    best_va_res = va_res
                    # Save best validation result
                    with open(self.va_res_file, "w") as ofp:
                        ofp.write("{:e}\n".format(best_va_res))
                    # Save best model
                    self.saver_best.save(
                        self.sess, self.save_file_best,
                        write_meta_graph=False,
                    )

    def test(self, data):
        """Test routine"""

        # Check if model exists
        if not os.path.exists(self.save_file_best + ".index"):
            print("Model File {} does not exist! Quiting".format(
                self.save_file_best))
            exit(1)

        # Restore model
        print("Restoring from {}...".format(
            self.save_file_best))
        self.saver_best.restore(
            self.sess,
            self.save_file_best)

        # Run Test
        cur_global_step = 0     # dummy
        if self.config.vis_dump:
            test_mode_list = ["test"]
        else:
            # test_mode_list = ["valid", "test"]
            test_mode_list = ["test"]  # Only run testing
        for test_mode in test_mode_list:
            # my_test_process(
            #     test_mode, self.sess,
            #     cur_global_step,
            #     self.summary_op, getattr(self, "summary_" + test_mode[:2]),
            #     self.x_in, self.y_in, self.R_in, self.t_in,
            #     self.is_training,
            #     None, None, None,
            #     self.logits, self.e_hat, self.loss, data[test_mode],
            #     getattr(self, "res_dir_" + test_mode[:2]), self.config)
            Presision, Recall, F_score, F2_score, F0_5_score = my_test_process(
                test_mode, self.sess,
                cur_global_step,
                self.summary_op, getattr(self, "summary_" + test_mode[:2]),
                self.x_in, self.y_in, self.R_in, self.t_in,
                self.is_training,
                None, None, None,
                self.logits, self.e_hat, self.loss, data[test_mode],
                getattr(self, "res_dir_" + test_mode[:2]), self.config)
            p_ = np.expand_dims(np.mean(np.array(Presision)), axis=0)
            r_ = np.expand_dims(np.mean(np.array(Recall)), axis=0)
            f_ = np.expand_dims(np.mean(np.array(F_score)), axis=0)
            f2_ = np.expand_dims(np.mean(np.array(F2_score)), axis=0)
            f0_5_ = np.expand_dims(np.mean(np.array(F0_5_score)), axis=0)

            np.savetxt(os.path.join("./", self.config.res_dir, "Precision.txt"), p_ * 100)
            np.savetxt(os.path.join("./", self.config.res_dir, "Recall.txt"), r_ * 100)
            np.savetxt(os.path.join("./", self.config.res_dir, "F-measure.txt"), f_ * 100)
            np.savetxt(os.path.join("./", self.config.res_dir, "F2-measure.txt"), f2_ * 100)
            np.savetxt(os.path.join("./", self.config.res_dir, "F0_5-measure.txt"), f0_5_ * 100)

    def comp(self, data):
        """Goodie for competitors"""

        # Run competitors on dataset
        for test_mode in ["test", "valid"]:
            comp_process(
                test_mode,
                data[test_mode],
                getattr(self, "res_dir_" + test_mode[:2]), self.config)


#
# network.py ends here
