# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.util import nest

class Seq2Seq(object):
    def __init__(self,rnn_size,num_layer_enc,num_layer_dec,emb_size,mode,beam_size,lr,
                 vocab_size,start=2,stop=2,max_iterations=40,max_gradient_norm=5.0):
        self.rnn_size = rnn_size
        self.lr = lr
        self.num_layer_enc = num_layer_enc
        self.num_layer_dec = num_layer_dec
        self.emb_size = emb_size        
        self.mode = mode
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.start = start
        self.stop = stop
        self.max_iterations = max_iterations
        self.max_gradient_norm = max_gradient_norm
        
        self.model()
    
    def create_LSTMCell(self,rnn_size,num_layer):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(rnn_size)
            #添加dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        #列表中每个元素都是调用single_rnn_cell函数
        if num_layer > 1:
            num_layer = int(num_layer / 2)
        else:
            num_layer = num_layer
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(num_layer)])
        return cell

    def _build_bidirectional_rnn(self,inputs,sequence_length,num_layer):
        
        fw_cell = self.create_LSTMCell(self.rnn_size,num_layer)
        bw_cell = self.create_LSTMCell(self.rnn_size,num_layer)
        
        encoder_outputs, encoder_states=tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32)
        encoder_fw_outputs, encoder_bw_outputs = encoder_outputs
        if num_layer>1:
            num_layer = int(num_layer/2)
        else:
            num_layer = num_layer
        encoder_state = ()
        for layer_id in range(num_layer):
            if num_layer == 1:
                encoder_fw_state,encoder_bw_state=encoder_states
                encoder_fw_state = encoder_fw_state[0]
                encoder_bw_state = encoder_bw_state[0]
            else:
                encoder_fw_state, encoder_bw_state = encoder_states[layer_id]
            encoder_state_c = tf.concat((encoder_fw_state.c,encoder_bw_state.c),1)
            encoder_state_h = tf.concat((encoder_fw_state.h,encoder_bw_state.h),1)            
            encoder_state_ = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c,h=encoder_state_h)
            encoder_state += (encoder_state_,)        
        encoder_outputs = tf.concat((encoder_fw_outputs,encoder_bw_outputs),-1)
        return encoder_outputs,encoder_state
    
    def AttentionLayer(self,inputs,name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.rnn_size * 2]), name='u_context')
            h = tf.contrib.layers.fully_connected(inputs,self.rnn_size*2,activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output
        
    def encoder(self,encoder_inputs_embeded):
        sequence_length = self.encoder_length
        encoder_outputs,encoder_state = self._build_bidirectional_rnn(encoder_inputs_embeded,sequence_length,self.num_layer_enc)
        return encoder_outputs,encoder_state
    
    def decoder(self):
        def ft_1():
            encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs,multiplier=self.beam_size)
            encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s,self.beam_size),self.encoder_state)
            encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_length,multiplier=self.beam_size)
            encoder_inputs = tf.contrib.seq2seq.tile_batch(self.encoder_inputs,multiplier=self.beam_size)
            batch_size = self.batch_size*self.beam_size
            return encoder_outputs,encoder_state,encoder_inputs_length,encoder_inputs,batch_size
        def ff_1():
            encoder_outputs = self.encoder_outputs
            encoder_state = self.encoder_state
            encoder_inputs_length = self.encoder_length
            encoder_inputs=self.encoder_inputs
            batch_size = self.batch_size
            return encoder_outputs,encoder_state,encoder_inputs_length,encoder_inputs,batch_size
        
        encoder_outputs,encoder_state,encoder_inputs_length,encoder_inputs,batch_size = tf.cond(tf.equal(self.beam_search,True),ft_1,ff_1)
        #attention for encoder-output
        #encoder_outputs = self.AttentionVAE(encoder_outputs,"encoder-attention")
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size*2,memory=encoder_outputs,
                                                                memory_sequence_length=encoder_inputs_length)
        decoder_cell = self.create_LSTMCell(self.rnn_size*2,self.num_layer_dec)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.rnn_size*2,name="Attention_Wrapper")
        decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
        output_layer = tf.layers.Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
        
        if self.mode == 'train':
            print("========train======")
            #decoder_inputs_embedded:[batch_size,decoder_length,embedding_size]
            ending = tf.strided_slice(self.decoder_inputs,[0,0],[self.batch_size,-1],[1,1])
            decoder_input = tf.concat([tf.fill([self.batch_size,1],self.start),ending],1)
            decoder_inputs_embeded = tf.nn.embedding_lookup(self.embedding,decoder_input)
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embeded,
                                                                 sequence_length=self.decoder_length,
                                                                 time_major=False,name="training_helper")
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=training_helper,
                                                               initial_state=decoder_initial_state,output_layer=output_layer)
                
            decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=self.max_target_sequence_length)
            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train,axis=-1,name="decoder_pred_train")
            #?????
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                         targets=self.decoder_inputs,
                                                         weights=self.mask)
                
            #training summary for the current batch_loss
            tf.summary.scalar('loss',self.loss)
            self.summary_op = tf.summary.merge_all()
            
            def valid():
                start_tokens = tf.ones([self.batch_size,],tf.int32)*self.start
                end_token = self.stop
                def ft_2():
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                             embedding=self.embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                    decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                            maximum_iterations=self.max_iterations)
                    decoder_predict_valid = decoder_outputs.predicted_ids
                    return decoder_predict_valid
                def ff_2():
                    print("Greedy_search_valid======")
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                    decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                            maximum_iterations=self.max_iterations)
                    decoder_logits_valid = tf.identity(decoder_outputs.rnn_output)
                    decoder_predict_valid = tf.argmax(decoder_logits_valid,axis=-1,name="decoder_predict_valid",output_type=tf.int32)
                    return decoder_predict_valid
                decoder_predict_valid = tf.cond(tf.equal(self.beam_search,True),ft_2,ff_2)
                return decoder_predict_valid
            
            self.decoder_predict_valid = valid()
            optimizer = tf.train.AdamOptimizer(self.lr)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss,trainable_params)
            clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))
            
    def model(self):
        print("building model....")      
        '''==================================1. placeholder'''
        self.encoder_inputs = tf.placeholder(tf.int32,[None,None],name="encoder_inputs")
        self.encoder_length = tf.placeholder(tf.int32,[None],name="encoder_length")
        self.decoder_inputs = tf.placeholder(tf.int32,[None,None],name="decoder_inputs")
        self.decoder_length = tf.placeholder(tf.int32,[None],name="decoder_length")
        self.beam_search = tf.placeholder(tf.bool,name="beam_search")
        self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")
        self.keep_prob_placeholder = tf.placeholder(tf.float32,name="keep_prob_placeholder")
        self.lr = tf.placeholder(tf.float32,[],name="learning_rate")
        self.max_target_sequence_length = tf.reduce_max(self.decoder_length,name="max_target_sequence_length")
        self.mask = tf.sequence_mask(self.decoder_length,self.max_target_sequence_length,dtype=tf.float32,name="mask")
        '''==================================2. encoder'''
        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable("embedding",[self.vocab_size,self.emb_size])
            encoder_inputs_embeded = tf.nn.embedding_lookup(self.embedding,self.encoder_inputs)

        with tf.variable_scope("Encoder"):
            self.encoder_outputs,self.encoder_state = self.encoder(encoder_inputs_embeded)
        '''=================================3. decoder'''
        with tf.variable_scope("Decoder"):
            self.decoder()
        '''=======================4. save model'''
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver_best = tf.train.Saver(tf.global_variables(),max_to_keep=3)
    def train(self,sess,x,learning_rate,keep_prob_placeholder_train):
        feed_dict = {self.encoder_inputs:x['enc_in'],
                     self.encoder_length:x['enc_len'],
                     self.decoder_inputs:x['dec_in'],
                     self.decoder_length:x['dec_len'],
                     self.beam_search:False,
                     self.keep_prob_placeholder:0.5,
                     self.lr:learning_rate,
                     self.batch_size:len(x['enc_in'])}
        _,loss,summary = sess.run([self.train_op,self.loss,self.summary_op],feed_dict=feed_dict)
        return loss,summary
    
    def validate(self,sess,x,beam_search,keep_prob_placeholder_test):
        feed_dict = {self.encoder_inputs:x['enc_in'],
                     self.encoder_length:x['enc_len'],
                     self.beam_search:beam_search,
                     self.keep_prob_placeholder:keep_prob_placeholder_test,
                     self.batch_size:len(x['enc_in'])}
        predicts = sess.run([self.decoder_predict_valid],feed_dict=feed_dict)
        return predicts
    