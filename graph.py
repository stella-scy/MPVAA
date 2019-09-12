#-*- coding: UTF-8 -*-

import tensorflow as tf
from modules import *


class Graph():
    def __init__(self, conf, is_training=True):
        self.graph = tf.Graph()
        print('test_0')
        with self.graph.as_default():
            print('test_1')
            # Encoder
            self.x = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='x')
            self.x_mask = tf.placeholder(tf.int32, shape=[None,None], name='x_mask')
            self.y = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='y')
            self.y_mask = tf.placeholder(tf.int32, shape=[None,None], name='y_mask')
            self.y_target = tf.placeholder(tf.int32, shape=[None,None], name='y_target')
            self.drop = tf.placeholder(tf.bool, shape=[], name='drop')
            
            self.train_inps  = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop, 'y':self.y, 'y_mask':self.y_mask, 'y_target':self.y_target}
            self.valid_inps  = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop, 'y':self.y, 'y_mask':self.y_mask, 'y_target':self.y_target}
            self.decode_inps = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop}
            self.encode_inps = {'x':self.x, 'x_mask':self.x_mask, 'drop':self.drop}
            
            self.enc = self.x
            print('test_2')
            print('self.enc', self.enc)
            
            self.px = tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1])
            if conf['option']['position'] == 'sin':
                self.enc += positional_encoding(self.px,
                                                vocab_size=conf['option']['maxlen']+2,
                                                num_units=conf['option']['dim_word'],
                                                zero_pad=False,
                                                scale=False,
                                                scope='enc_pos')
            elif conf['option']['position'] == 'emb':
                self.enc += embedding(self.px,
                                      vocab_size=conf['option']['maxlen']+2,
                                      num_units=conf['option']['dim_word'],
                                      zero_pad=False,
                                      scale=False,
                                      scope='enc_pos')
            else:
                pass
            print('test_2')
            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=conf['option']['drop_rate'],
                                         training=self.drop)
            print('test_3')
            ## Layers
            for i in range(conf['option']['layer_n']):
                with tf.variable_scope('enc_layers_{}'.format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   drop=self.drop,
                                                   dropout_rate=conf['option']['drop_rate'],
                                                   num_units=conf['option']['dim_model'],
                                                   num_heads=conf['option']['head'],
                                                   causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])
            print('test_4')
            print('self.enc', self.enc)
            ## Pooling
            enc_mask = tf.tile(tf.expand_dims(self.x_mask, -1), [1, 1, tf.shape(self.enc)[-1]])
            print('enc_mask', enc_mask)
            enc_mask_float = tf.to_float(enc_mask)
            print('enc_mask_float', enc_mask_float)
            self.enc_mean = tf.reduce_sum(self.enc * enc_mask_float, 1) / tf.reduce_sum(enc_mask_float, 1)
            print('self.enc_mean', self.enc_mean)
            
            min_paddings = tf.ones_like(self.enc)*(-2**32+1)
            self.enc_max = tf.where(tf.equal(enc_mask, 0), min_paddings, self.enc)
            self.enc_max = tf.reduce_max(self.enc_max, 1)

            print('latest self.enc_mean', self.enc_mean)
            print('latest self.enc_max', self.enc_max)
            print('tf.expand_dims(self.enc_mean, 1)', tf.expand_dims(self.enc_mean, 1))
            print('tf.expand_dims(self.enc_max, 1)', tf.expand_dims(self.enc_max, 1))

            tmp_enc_mean = tf.expand_dims(self.enc_mean, 1)
            tmp_enc_max = tf.expand_dims(self.enc_max, 1)

            ####lamb_1 = tf.Variable(tf.random_uniform([1, 250], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None), name='lamb_1')
            lamb_1 = tf.Variable(tf.random_uniform([1, 250], minval=0, maxval=1, dtype=tf.float32, seed=None, name=None), name='lamb_1')
            print('lamb_1', lamb_1)
##########            lamb_1 = tf.cast(lamb_1, tf.float32)
##########            print('later lamb_1', lamb_1)

            self.ctx = tf.multiply(lamb_1, tmp_enc_max) + tf.multiply(1-lamb_1, tmp_enc_mean)
            
            print('self.ctx', self.ctx)
            temp_mean = tf.expand_dims(self.enc_mean, 1)
            print('temp_mean', temp_mean)
#############################################################################################################################################################

            self.x_2 = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='x_2')
            self.x_mask_2 = tf.placeholder(tf.int32, shape=[None,None], name='x_mask_2')
            self.y_2 = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='y_2')
            self.y_mask_2 = tf.placeholder(tf.int32, shape=[None,None], name='y_mask_2')
            self.y_target_2 = tf.placeholder(tf.int32, shape=[None,None], name='y_target_2')
            self.drop_2 = tf.placeholder(tf.bool, shape=[], name='drop_2')
            
            self.train_inps_2  = {'x_2':self.x_2, 'x_mask_2':self.x_mask_2, 'drop_2':self.drop_2, 'y_2':self.y_2, 'y_mask_2':self.y_mask_2, 'y_target_2':self.y_target_2}
            self.valid_inps_2  = {'x_2':self.x_2, 'x_mask_2':self.x_mask_2, 'drop_2':self.drop_2, 'y_2':self.y_2, 'y_mask_2':self.y_mask_2, 'y_target_2':self.y_target_2}
            self.decode_inps_2 = {'x_2':self.x_2, 'x_mask_2':self.x_mask_2, 'drop_2':self.drop_2}
            self.encode_inps_2 = {'x_2':self.x_2, 'x_mask_2':self.x_mask_2, 'drop_2':self.drop_2}
            
            self.enc_2 = self.x_2
            print('test_2_2')
            print('self.enc_2', self.enc_2)
            
            self.px_2 = tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_2)[1]), 0), [tf.shape(self.x_2)[0], 1])
            if conf['option']['position'] == 'sin':
                self.enc_2 += positional_encoding(self.px_2,
                                                vocab_size=conf['option']['maxlen']+2,
                                                num_units=conf['option']['dim_word'],
                                                zero_pad=False,
                                                scale=False,
                                                scope='enc_pos_2')
            elif conf['option']['position'] == 'emb':
                self.enc_2 += embedding(self.px_2,
                                      vocab_size=conf['option']['maxlen']+2,
                                      num_units=conf['option']['dim_word'],
                                      zero_pad=False,
                                      scale=False,
                                      scope='enc_pos_2')
            else:
                pass
            print('test_2')
            ## Dropout
            self.enc_2 = tf.layers.dropout(self.enc_2,
                                         rate=conf['option']['drop_rate'],
                                         training=self.drop)
            print('test_3')
            ## Layers
            for i in range(conf['option']['layer_n']):
                with tf.variable_scope('enc_layers_2_{}'.format(i)):
                    ### Multihead Attention
                    self.enc_2 = multihead_attention(queries=self.enc_2,
                                                   keys=self.enc_2,
                                                   drop=self.drop_2,
                                                   dropout_rate=conf['option']['drop_rate'],
                                                   num_units=conf['option']['dim_model'],
                                                   num_heads=conf['option']['head'],
                                                   causality=False)
                    
                    ### Feed Forward
                    self.enc_2 = feedforward(self.enc_2, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])
            print('test_4_2')
            print('self.enc_2', self.enc_2)
            ## Pooling
            enc_mask_2 = tf.tile(tf.expand_dims(self.x_mask_2, -1), [1, 1, tf.shape(self.enc_2)[-1]])
            enc_mask_float_2 = tf.to_float(enc_mask_2)
            self.enc_mean_2 = tf.reduce_sum(self.enc_2 * enc_mask_float_2, 1) / tf.reduce_sum(enc_mask_float_2, 1)
            
            min_paddings_2 = tf.ones_like(self.enc_2)*(-2**32+1)
            self.enc_max_2 = tf.where(tf.equal(enc_mask_2, 0), min_paddings_2, self.enc_2)
            self.enc_max_2 = tf.reduce_max(self.enc_max_2, 1)

            tmp_enc_mean_2 = tf.expand_dims(self.enc_mean_2, 1)
            tmp_enc_max_2 = tf.expand_dims(self.enc_max_2, 1)

            self.ctx_2 = tf.multiply(lamb_1, tmp_enc_max_2) + tf.multiply(1-lamb_1, tmp_enc_mean_2)        
            print('self.ctx_2', self.ctx_2)


#############################################################################################################################################################

            self.x_3 = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='x_3')
            self.x_mask_3 = tf.placeholder(tf.int32, shape=[None,None], name='x_mask_3')
            self.y_3 = tf.placeholder(tf.float32, shape=[None,None,conf['option']['dim_word']], name='y_3')
            self.y_mask_3 = tf.placeholder(tf.int32, shape=[None,None], name='y_mask_3')
            self.y_target_3 = tf.placeholder(tf.int32, shape=[None,None], name='y_target_3')
            self.drop_3 = tf.placeholder(tf.bool, shape=[], name='drop_3')
            
            self.train_inps_3  = {'x_3':self.x_3, 'x_mask_3':self.x_mask_3, 'drop_3':self.drop_3, 'y_3':self.y_3, 'y_mask_3':self.y_mask_3, 'y_target_3':self.y_target_3}
            self.valid_inps_3  = {'x_3':self.x_3, 'x_mask_3':self.x_mask_3, 'drop_3':self.drop_3, 'y_3':self.y_3, 'y_mask_3':self.y_mask_3, 'y_target_3':self.y_target_3}
            self.decode_inps_3 = {'x_3':self.x_3, 'x_mask_3':self.x_mask_3, 'drop_3':self.drop_3}
            self.encode_inps_3 = {'x_3':self.x_3, 'x_mask_3':self.x_mask_3, 'drop_3':self.drop_3}
            
            self.enc_3 = self.x_3
            print('test_2_3')
            print('self.enc_3', self.enc_3)
            
            self.px_3 = tf.tile(tf.expand_dims(tf.range(tf.shape(self.x_3)[1]), 0), [tf.shape(self.x_3)[0], 1])
            if conf['option']['position'] == 'sin':
                self.enc_3 += positional_encoding(self.px_3,
                                                vocab_size=conf['option']['maxlen']+2,
                                                num_units=conf['option']['dim_word'],
                                                zero_pad=False,
                                                scale=False,
                                                scope='enc_pos_3')
            elif conf['option']['position'] == 'emb':
                self.enc_3 += embedding(self.px_3,
                                      vocab_size=conf['option']['maxlen']+2,
                                      num_units=conf['option']['dim_word'],
                                      zero_pad=False,
                                      scale=False,
                                      scope='enc_pos_3')
            else:
                pass
            print('test_3')
            ## Dropout
            self.enc_3 = tf.layers.dropout(self.enc_3,
                                         rate=conf['option']['drop_rate'],
                                         training=self.drop)
            print('test_4')
            ## Layers
            for i in range(conf['option']['layer_n']):
                with tf.variable_scope('enc_layers_3_{}'.format(i)):
                    ### Multihead Attention
                    self.enc_3 = multihead_attention(queries=self.enc_3,
                                                   keys=self.enc_3,
                                                   drop=self.drop_3,
                                                   dropout_rate=conf['option']['drop_rate'],
                                                   num_units=conf['option']['dim_model'],
                                                   num_heads=conf['option']['head'],
                                                   causality=False)
                    
                    ### Feed Forward
                    self.enc_3 = feedforward(self.enc_3, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])
            print('test_4_3')
            print('self.enc_3', self.enc_3)
            ## Pooling
            enc_mask_3 = tf.tile(tf.expand_dims(self.x_mask_3, -1), [1, 1, tf.shape(self.enc_3)[-1]])
            enc_mask_float_3 = tf.to_float(enc_mask_3)
            self.enc_mean_3 = tf.reduce_sum(self.enc_3 * enc_mask_float_3, 1) / tf.reduce_sum(enc_mask_float_3, 1)
            
            min_paddings_3 = tf.ones_like(self.enc_3)*(-2**32+1)
            self.enc_max_3 = tf.where(tf.equal(enc_mask_3, 0), min_paddings_3, self.enc_3)
            self.enc_max_3 = tf.reduce_max(self.enc_max_3, 1)

            tmp_enc_mean_3 = tf.expand_dims(self.enc_mean_3, 1)
            tmp_enc_max_3 = tf.expand_dims(self.enc_max_3, 1)

            self.ctx_3 = tf.multiply(lamb_1, tmp_enc_max_3) + tf.multiply(1-lamb_1, tmp_enc_mean_3)
            print('self.ctx_3', self.ctx_3)

#############################################################################################################################################################

            #Multi-View Decoder
            sh = tf.shape(self.ctx)[0]
            print('sh', sh)
            print('self.ctx', self.ctx)
            print('self.ctx_2', self.ctx_2)
            print('self.ctx_3', self.ctx_3)

            ctx_r = tf.reshape(self.ctx, [-1,250])
            ctx_2_r = tf.reshape(self.ctx_2, [-1,250])
            ctx_3_r = tf.reshape(self.ctx_3, [-1,250])
 
            self.w1 = tf.Variable(tf.random_normal([250,250], stddev=0.1), name='w1')
            print('self.w1', self.w1)
            self.w2 = tf.Variable(tf.random_normal([250,250], stddev=0.1), name='w2')
            self.w3 = tf.Variable(tf.random_normal([250,250], stddev=0.1), name='w3')

            print('self.w2', self.w2)
            print('self.ctx', self.ctx)
            print('self.ctx_2', self.ctx_2)
            print('self.ctx_3', self.ctx_3)
            contx_temp = tf.reduce_sum([tf.matmul(ctx_r, self.w1), tf.matmul(ctx_2_r, self.w2), tf.matmul(ctx_3_r, self.w3)],0)
            print('contx_temp', contx_temp)
            contx_prob = tf.nn.softmax(contx_temp)
            print('contx_prob', contx_prob)
            contx_new = tf.multiply(contx_prob, ctx_r)
            print('here contx_new', contx_new)
            contx_new = tf.reshape(contx_new, [-1, 1,250])
            print('new contx_new', contx_new)
            logits = self.decode(conf, self.y, contx_new)
            print('logits', logits)
            self.probs = tf.nn.softmax(logits)
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1))
            y_istarget = tf.to_float(self.y_mask)
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_target))*y_istarget)
            tf.summary.scalar('acc', self.acc)
            
            y_smoothed = label_smoothing(tf.one_hot(self.y_target, depth=conf['option']['vocab_size']))
            print('self.y_target', self.y_target)
            print('y_smoothed', y_smoothed)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)                
            self.mean_loss = tf.reduce_sum(loss*y_istarget) / (tf.reduce_sum(y_istarget))
            tf.summary.scalar('mean_loss', self.mean_loss)

#####################################################################################################################################################
            self.merged = tf.summary.merge_all()

            if is_training:
                print('self.enc', self.enc)
                print('test_5')
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lrate = tf.Variable(conf['option']['lrate'], trainable=False)
                if conf['option']['optimizer'] == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(self.lrate)
                
                updates = tf.trainable_variables()
                grads = tf.gradients(self.mean_loss, updates)
                if conf['option']['clip_grad'] > 0.:
                    clip_grads, _ = tf.clip_by_global_norm(grads, conf['option']['clip_grad'])
                else:
                    clip_grads = grads
                self.train_op = self.optimizer.apply_gradients(zip(clip_grads, updates), global_step=self.global_step)


    def decode(self, conf, y, contx):
        dec = y
        ## Positional Encoding
        py = tf.tile(tf.expand_dims(tf.range(tf.shape(y)[1]), 0), [tf.shape(y)[0], 1])
        if conf['option']['position'] == 'sin':
            dec += positional_encoding(py,
                                       vocab_size=conf['option']['maxlen']+2,
                                       num_units=conf['option']['dim_word'],
                                       zero_pad=False,
                                       scale=False,
                                       scope='dec_pos')
        elif conf['option']['position'] == 'emb':
            dec += embedding(py,
                             vocab_size=conf['option']['maxlen']+2,
                             num_units=conf['option']['dim_word'],
                             zero_pad=False, 
                             scale=False,
                             scope='dec_pos')
        else:
            pass

        ## Dropout
        dec = tf.layers.dropout(dec,
                                rate=conf['option']['drop_rate'],
                                training=self.drop)

        ## Layers
        for i in range(conf['option']['layer_n']):
            with tf.variable_scope('dec_s_layers_{}'.format(i)):
                ## Multihead Attention ( self-attention)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          drop=self.drop,
                                          dropout_rate=conf['option']['drop_rate'],
                                          num_units=conf['option']['dim_model'],
                                          num_heads=conf['option']['head'],
                                          causality=True,
                                          scope='self_attention')
                
                dec = multihead_attention(queries=dec,
                                          keys=contx,
                                          drop=self.drop,
                                          dropout_rate=conf['option']['drop_rate'],
                                          num_units=conf['option']['dim_model'],
                                          num_heads=conf['option']['head'],
                                          causality=False,
                                          residual=True,
                                          scope='vanilla_attention')
                
                ## Feed Forward
                dec = feedforward(dec, num_units=[conf['option']['dim_inner'], conf['option']['dim_model']])

        # Final linear projection
        logits = tf.layers.dense(dec, conf['option']['vocab_size'])

        return logits


