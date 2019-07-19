import torch

class Train:
    def __init__():
        self.best_acc = 1.0
        self.current_acc = 0.0

    def start_train():
        global_step = 0
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=global_step, decay_steps=5000,                                           decay_rate=0.95)
        d_raw_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        d_gvs = d_raw_opt.compute_gradients(self.d_loss, var_list=self.d_vars)
        d_opt = d_raw_opt.apply_gradients(d_gvs)
        g_raw_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        g_gvs = g_raw_opt.compute_gradients(self.g_loss, var_list=self.g_vars)
        g_opt = g_raw_opt.apply_gradients(g_gvs)

        start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Modify by ZhangAnlan
        # here we restore the variables of GTSRB model
        restore_vars = [var for var in tf.global_variables() if var.name.startswith('GTSRB')]
        saver = tf.train.Saver(restore_vars)
        saver.restore(sess=self.sess, save_path=self.target_model_dir)

        # liuaishan get validation set and train set
        # val_data_x, val_data_y, val_data_z = shuffle_augment_and_load(self.base_image_num, self.valid_img_dir, self.base_patch_num, self.valid_patch_dir, self.batch_size)
        self.train_pair_set = get_initial_image_patch_pair(self.image_all_num, self.patch_all_num)
        valid_pair_set = get_initial_image_patch_pair(self.image_val_num, self.patch_val_num, True)

        val_data_x, val_data_y, val_data_z = load_data_in_pair(valid_pair_set, self.batch_size, self.valid_img_dir,
                                                               self.valid_patch_dir, self.class_num)
        val_data_x = np.array(val_data_x).astype(np.float32)
        val_data_y = np.array(val_data_y).astype(np.float32)
        val_data_z = np.array(val_data_z).astype(np.float32)
        print(self.sess.run(learning_rate))

        for epoch in range(self.epoch):
            batch_iteration = self.image_all_num * self.patch_all_num / self.batch_size

            for id in range(int(batch_iteration)):

                # batch_data_x, batch_data_y, batch_data_z  = \
                #    shuffle_augment_and_load(self.base_image_num, self.image_dir, self.base_patch_num,
                #                             self.patch_dir, self.batch_size )

                batch_data_x, batch_data_y, batch_data_z = load_data_in_pair(self.train_pair_set, self.batch_size,
                                                                             self.image_dir, self.patch_dir,
                                                                             self.class_num)
                batch_data_x = np.array(batch_data_x).astype(np.float32)
                batch_data_y = np.array(batch_data_y).astype(np.float32)
                batch_data_z = np.array(batch_data_z).astype(np.float32)

                # liuas 2018.5.7 trick: we train G once while D d_train_freq times in one iteration
                # if (id + 1) % self.d_train_freq == 0:
                self.sess.run([g_opt], feed_dict={self.real_image: batch_data_x, self.y: batch_data_y,
                                                  self.real_patch: batch_data_z})

                self.sess.run([d_opt],
                              feed_dict={self.real_image: batch_data_x,
                                         self.y: batch_data_y,
                                         self.real_patch: batch_data_z})

                counter += 1

                # test the accuracy
                # liuas 2018.5.9 validation

                if np.mod(counter, 40) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, id, batch_iteration, time.time() - start_time))
                    print("[Validation].......")

                    print("learning_rate: %.8f" % self.sess.run(learning_rate))

                    # test! Show logits and probs when validating
                    print("[top 3 fake_logits].......")
                    current_fake_logits = self.fake_logits_f.eval(
                        {self.real_image: val_data_x, self.real_patch: val_data_z})
                    for i in range(len(current_fake_logits)):
                        top1 = np.argsort(current_fake_logits[i])[-1]
                        top2 = np.argsort(current_fake_logits[i])[-2]
                        top3 = np.argsort(current_fake_logits[i])[-3]
                        print("%d %.8f %d %.8f %d %.8f" % (
                        top1, current_fake_logits[i][top1], top2, current_fake_logits[i][top2], top3,
                        current_fake_logits[i][top3]))
                    # print(current_fake_logits)
                    print("[top 3 fake_prob].......")
                    current_fake_prob = self.fake_prob_f.eval(
                        {self.real_image: val_data_x, self.real_patch: val_data_z})
                    for i in range(len(current_fake_prob)):
                        top1 = np.argsort(current_fake_prob[i])[-1]
                        top2 = np.argsort(current_fake_prob[i])[-2]
                        top3 = np.argsort(current_fake_prob[i])[-3]
                        print("%d %.8f %d %.8f %d %.8f" % (
                        top1, current_fake_prob[i][top1], top2, current_fake_prob[i][top2], top3,
                        current_fake_prob[i][top3]))

                    errAE = self.ae_loss.eval({self.real_image: val_data_x,
                                               self.y: val_data_y,
                                               self.real_patch: val_data_z})

                    errD = self.d_loss.eval({self.real_image: val_data_x,
                                             self.y: val_data_y,
                                             self.real_patch: val_data_z})

                    errG = self.g_loss.eval({self.real_image: val_data_x,
                                             self.y: val_data_y,
                                             self.real_patch: val_data_z})

                    acc = self.accuracy.eval({self.real_image: val_data_x,
                                              self.y: val_data_y,
                                              self.real_patch: val_data_z})

                    errPadSim = self.pad_sim_loss.eval({self.real_image: val_data_x,
                                                        self.y: val_data_y,
                                                        self.real_patch: val_data_z})

                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))
                    print("Accuracy of classification: %4.4f" % acc)

                    acc_batch = self.accuracy.eval({self.real_image: batch_data_x,
                                                    self.y: batch_data_y,
                                                    self.real_patch: batch_data_z})
                    print("train batch acc: %4.4f" % acc_batch)

                    print("ae_loss: %.8f" % errAE)

                    print("pad_sim_loss: %.8f" % errPadSim)

                    print("sum_loss: %.8f" % (errAE + errPadSim))
                    if acc < self.best_acc:
                        self.best_acc = acc
                        print("Saving model.")
                        self.save(self.checkpoint_dir, counter)
                    print("current acc: %.4f, best acc: %.4f" % (acc, self.best_acc))

                # liuas 2018.5.10 test
                if np.mod(counter, 1000) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, id, batch_iteration, time.time() - start_time))

                    # accuracy in the test set 2018.5.10 ZhangAnlan
                    print("[Test].......")

                    batch_data_x, batch_data_y, batch_data_z = \
                        shuffle_augment_and_load(self.base_image_num, self.test_img_dir, self.base_patch_num,
                                                 self.patch_dir, self.batch_size)

                    batch_data_x = np.array(batch_data_x).astype(np.float32)
                    batch_data_y = np.array(batch_data_y).astype(np.float32)
                    batch_data_z = np.array(batch_data_z).astype(np.float32)

                    errD, errG, acc, fake_image, predictions, real_label = \
                        self.sess.run([self.d_loss, self.g_loss, self.accuracy, self.fake_image, self.predictions,
                                       self.real_label],
                                      feed_dict={self.real_image: batch_data_x,
                                                 self.y: batch_data_y,
                                                 self.real_patch: batch_data_z})
                    print("g_loss: %.8f , d_loss: %.8f" % (errG, errD))
                    print("Accuracy of classification: %4.4f" % acc)

                    # plot accuracy
                    self.acc_history.append(float(acc))
                    '''
                    plot_acc(self.acc_history, filename=self.output_dir+'/' +'Accrucy.png')
                    # save patches
                    save_patches(self.fake_patch.eval({self.real_image: batch_data_x,
                                                         self.y: batch_data_y,
                                                 self.real_patch: batch_data_z}),
                             filename=self.output_dir+'/' + str(time.time()) +'_fake_patches.png')
                    # show images and acc
                    self.show_images_and_acc(fake_image, predictions, real_label, num=9,
                                             filename=self.output_dir+'/' + str(time.time()) +'_fake_images.png')
                    '''

train = Train('cosine', transform, trainset, valset, testset, root=root, model_name=name_model, datafile=datafile,
              batch_size=batch_size)
print(train.batch_size)
train.start_train(epoch=epoch_num, learning_rate=learning_rate)
