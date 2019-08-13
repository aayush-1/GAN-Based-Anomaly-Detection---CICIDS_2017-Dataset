
import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.cicids_2017_utilities as network
import data.cicids_2017 as data
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd 
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score


RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter
#*
def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

#*
def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

#*
def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}".format(weight, method, rd)


def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the CICIDS_2017 dataset
    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("BiGAN.cicids_2017.{}".format(method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    #********************************************************************** 

    

    #**********************************************************************

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    
    

    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder
    dis = network.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, latent_dim])
        x_gen = gen(z, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen,input_pl, is_training=is_training_pl)
        l_generator, inter_layer_rct = dis(z, x_gen, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator
        loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator))
        # encoder
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder),logits=l_encoder))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema),logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                 keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score


    logdir = create_logdir(weight, method, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
                             save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        epoch = 0
        
        while not sv.should_stop() and epoch < nb_epochs:
            tr_loss_gen=0
            tr_loss_enc=0
            tr_loss_dis=0
            list_1=[x for x in range(0,8)]
            train_batch = 0
            for i in range(4):

#******************************************************************************************************************************************

                #training data preprocessing
            
                train_list=np.random.choice(list_1,2,replace=False)
                train_path='MachineLearningCVE/'
                train_list_path=[train_path + str(x) + '.csv' for x in train_list]
                InD = np.zeros((0,79),dtype=object)
                for x in train_list_path:
                    InD=np.vstack((InD,pd.read_csv(x)))

                #Dt=Data
                Dt=InD[:,:-1].astype(float)


                #Remove nan values
                #L_n -- label without nan values
                L_N=InD[~np.isnan(Dt).any(axis=1),-1]
                #DtNMV -- data without nan values
                D_N=Dt[~np.isnan(Dt).any(axis=1)]


                #Remove Inf values
                #labels without nan and inf values
                L_NI=L_N[~np.isinf(D_N).any(axis=1)]
                #data without nan and inf values
                D_NI=D_N[~np.isinf(D_N).any(axis=1)]
                del(D_N)

                D_NI=MinMaxScaler().fit_transform(D_NI)

                x_train_net=D_NI[L_NI=='BENIGN',:]
                x_train_net=x_train_net[rng.permutation(x_train_net.shape[0])]
                trainx=x_train_net[:int(x_train_net.shape[0]*0.6),:]

                y_train=L_NI[L_NI=='BENIGN']

                if i==0:
                    x_test_anomaly=D_NI[L_NI!='BENIGN',:]
                    x_test_benign=x_train_net[int(x_train_net.shape[0]*0.6):,:]
                x_test_anomaly=np.vstack((x_test_anomaly,D_NI[L_NI!='BENIGN',:]))
                x_test_benign=np.vstack((x_test_benign,x_train_net[int(x_train_net.shape[0]*0.6):,:]))
                del(D_NI)
                del(L_NI)

                #trainx, trainy = data.get_train()
                trainx_copy = trainx.copy()

    #******************************************************************************************************************************************

                lr = starting_lr
                begin = time.time()

                 # construct randomly permuted minibatches
                trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
                trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
                train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

                nr_batches_train = int(trainx.shape[0] / batch_size)

                # training
                for t in range(nr_batches_train):
                    
                    display_progression_epoch(t, nr_batches_train)             
                    ran_from = t * batch_size
                    ran_to = (t + 1) * batch_size

                    # train discriminator
                    feed_dict = {input_pl: trainx[ran_from:ran_to],
                                 is_training_pl: True,
                                 learning_rate:lr}

                    _, ld, sm = sess.run([train_dis_op,
                                          loss_discriminator,
                                          sum_op_dis],
                                         feed_dict=feed_dict)
                    train_loss_dis += ld
                    writer.add_summary(sm, train_batch)

                    # train generator and encoder
                    feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                                 is_training_pl: True,
                                 learning_rate:lr}
                    _,_, le, lg, sm = sess.run([train_gen_op,
                                                train_enc_op,
                                                loss_encoder,
                                                loss_generator,
                                                sum_op_gen],
                                               feed_dict=feed_dict)
                    train_loss_gen += lg
                    train_loss_enc += le
                    writer.add_summary(sm, train_batch)

            
                    train_batch += 1

                del(trainx)
                del(trainx_copy)    
                list_1=[x for x in list_1 if x not in train_list]
                train_loss_gen /= (nr_batches_train)
                tr_loss_gen+=train_loss_gen
                train_loss_enc /= (nr_batches_train)
                tr_loss_enc+=train_loss_enc
                train_loss_dis /= (nr_batches_train)
                tr_loss_dis+=train_loss_dis
            tr_loss_dis /= 4
            tr_loss_gen /= 4
            tr_loss_enc /= 4
            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f " % (epoch, time.time() - begin, tr_loss_gen, tr_loss_enc, tr_loss_dis))

            epoch += 1
            
                      
        logger.warn('Testing evaluation...')
#*****************************************************************************************************************************
        #testing data pre processing
        # Normal data: label =0, anomalous data: label =1
        rho=0.2


        # normal data - x_test_benign

        # anomalous data - x_test_anomaly

        inds = rng.permutation(x_test_anomaly.shape[0])
        x_test_anomaly=x_test_anomaly[inds]

        inds = rng.permutation(x_test_benign.shape[0])
        x_test_benign = x_test_benign[inds] 

        size_test = x_test_benign.shape[0]
        out_size_test = int(size_test*rho/(1-rho))

        x_test_anomaly = x_test_anomaly[:out_size_test]

        y_test_benign=np.ones(x_test_benign.shape[0])
        y_test_anomaly=np.zeros(x_test_anomaly.shape[0])

        testx = np.concatenate((x_test_benign,x_test_anomaly), axis=0)
        testy = np.concatenate((y_test_benign,y_test_anomaly), axis=0)



#*****************************************************************************************************************************

        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]  # shuffling  dataset
        testy = testy[inds] # shuffling  dataset
        scores = []
        inference_time = []

        nr_batches_test = int(testx.shape[0] / batch_size)

        # Create scores
        for t in range(nr_batches_test):

            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl:False}

            scores += sess.run(list_scores,
                                         feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (
            np.mean(inference_time)))

        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        fill = np.ones([batch_size - size, 78])

        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch,
                     is_training_pl: False}

        batch_score = sess.run(list_scores,
                           feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        # Highest 80% are anomalous
        per = np.percentile(scores, 90)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)

        y_pred[inds] = 1
        y_pred[inds_comp] = 0


        precision, recall, f1,_ = precision_recall_fscore_support(testy,
                                                                  y_pred,
                                                                  average='binary')

        print(
            "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
            % (precision, recall, f1))

        print("accuracy : ", accuracy_score(testy, y_pred))



def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed)