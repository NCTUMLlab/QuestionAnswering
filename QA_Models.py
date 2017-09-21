from __future__ import print_function
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Add, Dot, Multiply, Permute, Dropout
from keras.layers import LSTM, Input, Lambda, Reshape, Flatten, RepeatVector, Concatenate
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
from babi_util import *

def rev_model(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,word_dim)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(256)(match)
    match         = Dense(256,activation='softplus')(match)
    match         = Dropout(0.3)(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)


    response    = Multiply()([input_encoder_m,att])
    response_f1 = LSTM(64)(response)
    response_f1 = Reshape((1,64))(response_f1)

    query_new   = Dot(axes=[2,2])([question_encoder,response_f1])
    query_new   = Flatten()(query_new)
    query_new   = Dense(64,activation='softplus')(query_new)
    query_new   = Reshape((1,64))(query_new)

    match2      = Dot(axes=[2,2])([input_encoder_m,query_new])
    match2      = Flatten()(match2)
    match2       = Dense(256,activation='softplus')(match2)
    match2       = Dropout(0.3)(match2)
    match2_mu    = Dense(story_maxlen,activation='softplus')(match2)
    att2         = Activation('sigmoid')(match2_mu)
    att2         = RepeatVector(64)(att2)
    att2         = Permute((2,1))(att2)        

    response2  = Multiply()([input_encoder_m,att2])
    response_f2= LSTM(64)(response2)
    resp_all   = Flatten()(response_f1)
    resp_all   = Concatenate(axis=1)([resp_all,response_f2])
    response_h = RepeatVector(story_maxlen_sup)(resp_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)
        
    resp   = Reshape((1,64))(response_f2)
    answer = Concatenate(axis=1)([question_encoder,response_f1,resp])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def rev_model_concat(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,word_dim)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(256)(match)
    match         = Dense(256,activation='softplus')(match)
    match         = Dropout(0.3)(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)


    response    = Multiply()([input_encoder_m,att])
    response_f1 = LSTM(64)(response)
    response_f1 = Reshape((1,64))(response_f1)

    query_new   = Concatenate(axis=1)([question_encoder,response_f1])
    query_new   = LSTM(128,activation='sigmoid')(query_new)
    query_new   = Dense(64,activation='softplus')(query_new)
    query_new   = Reshape((1,64))(query_new)

    match2      = Dot(axes=[2,2])([input_encoder_m,query_new])
    match2      = Flatten()(match2)
    match2       = Dense(256,activation='softplus')(match2)
    match2       = Dropout(0.3)(match2)
    match2_mu    = Dense(story_maxlen,activation='softplus')(match2)
    att2         = Activation('sigmoid')(match2_mu)
    att2         = RepeatVector(64)(att2)
    att2         = Permute((2,1))(att2)        

    
    response2  = Multiply()([input_encoder_m,att2])
    response_f2= LSTM(64)(response2)
    resp_all   = Flatten()(response_f1)
    resp_all   = Concatenate(axis=1)([resp_all,response_f2])
    response_h = RepeatVector(story_maxlen_sup)(resp_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_f2)
    answer = Concatenate(axis=1)([question_encoder,response_f1,resp])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def rev_model_tie(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,word_dim)
    controller_h1 = LSTM(256)
    controller_h2 = Dense(256,activation='softplus')
    controller_h3 = Dense(story_maxlen,activation='softplus')
    encoder = LSTM(64)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = controller_h1(match)
    match         = controller_h2(match)
    match         = Dropout(0.3)(match)
    match_mu      = controller_h3(match)
    att           = Activation('softmax')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)


    response    = Multiply()([input_encoder_m,att])
    response_f1 = encoder(response)
    response_f1 = Reshape((1,64))(response_f1)

    query_new   = Concatenate(axis=1)([question_encoder,response_f1])
    

    match2      = Dot(axes=[2,2])([query_new,input_encoder_m])
    match2      = controller_h1(match2)
    match2       = controller_h2(match2)
    match2       = Dropout(0.3)(match2)
    match2_mu    = controller_h3(match2)
    att2         = Activation('softmax')(match2_mu)
    att2         = RepeatVector(64)(att2)
    att2         = Permute((2,1))(att2)        

    response2  = Multiply()([input_encoder_m,att2])
    response_f2= LSTM(64)(response2)
    resp_all   = Flatten()(response_f1)
    resp_all   = Concatenate(axis=1)([resp_all,response_f2])
    response_h = RepeatVector(story_maxlen_sup)(resp_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)
        
    resp   = Reshape((1,64))(response_f2)
    answer = Concatenate(axis=1)([question_encoder,response_f1,resp])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def biway_model(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,word_dim)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)
    
    match_q2s     = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match_q2s     = LSTM(story_maxlen)(match_q2s)
    match_q2s     = Dense(256,activation='softplus')(match_q2s)
    
    match_s2q     = Dot(axes=[2,2])([input_encoder_m,question_encoder])
    match_s2q     = LSTM(query_maxlen)(match_s2q)
    match_s2q     = Dense(256,activation='softplus')(match_s2q)


    match         = Add()([match_q2s,match_s2q])
    match         = Dense(512,activation='softplus')(match)
    match         = Dropout(0.3)(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(word_dim)(att)
    att           = Permute((2,1))(att)


    response    = Multiply()([input_encoder_m,att])
    response    = LSTM(word_dim)(response)

    response_h = RepeatVector(story_maxlen_sup)(response)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)


    resp   = Reshape((1,64))(response)
    answer = Concatenate(axis=1)([question_encoder,resp])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def varMem(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,64)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(story_maxlen)(match)
    match         = Dense(256,activation='softplus')(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    #match_log_var = Dense(story_maxlen,activation='sigmoid')(match)
    #att           = Lambda(sampling_normal)([match_mu,match_log_var])
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)

    response   = Multiply()([input_encoder_m,att])
    response_f = LSTM(64)(response)
    response_h = RepeatVector(story_maxlen_sup)(response_f)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_f)
    answer = Concatenate(axis=1)([resp,question_encoder])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def varMem_key(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    key_emb  = Embedding(vocab_size,64)
    data_emb = Embedding(vocab_size,64)

    input_encoder_m = key_emb(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)
    output_encoder_m = data_emb(story)
    output_encoder_m = Dropout(0.3)(output_encoder_m)

    question_encoder = key_emb(query)
    question_encoder = Dropout(0.3)(question_encoder)
    question_output_encoder = data_emb(query)
    question_output_encoder = Dropout(0.3)(question_output_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(story_maxlen)(match)
    match         = Dense(256,activation='softplus')(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    #match_log_var = Dense(story_maxlen,activation='sigmoid')(match)
    #att           = Lambda(sampling_normal)([match_mu,match_log_var])
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)

    response   = Multiply()([output_encoder_m,att])
    response_f = LSTM(64)(response)
    response_h = RepeatVector(story_maxlen_sup)(response_f)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_f)
    answer = Concatenate(axis=1)([resp,question_output_encoder])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def varMem_key_multihop(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    key_emb  = Embedding(vocab_size,64)
    key_emb_2 = Embedding(vocab_size,64)
    data_emb = Embedding(vocab_size,64)
    encoder = LSTM(64)

    input_encoder_m = key_emb(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)
    input_encoder_m_2 = key_emb_2(story)
    input_encoder_m_2 = Dropout(0.3)(input_encoder_m_2)
    output_encoder_m = data_emb(story)
    output_encoder_m = Dropout(0.3)(output_encoder_m)

    question_encoder = key_emb(query)
    question_encoder = Dropout(0.3)(question_encoder)
    question_output_encoder = data_emb(query)
    question_output_encoder = Dropout(0.3)(question_output_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(story_maxlen)(match)
    match         = Dense(256,activation='softplus')(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    #match_log_var = Dense(story_maxlen,activation='sigmoid')(match)
    #att           = Lambda(sampling_normal)([match_mu,match_log_var])
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)

    response   = Multiply()([output_encoder_m,att])
    response_f = encoder(response)
    #2nd hop
    response_tmp = Reshape((1,64))(response_f)
    query_new = Concatenate(axis=1)([response_tmp,question_encoder])
    match_2 = Dot(axes=[2,2])([query_new,input_encoder_m_2])
    match_2 = LSTM(256)(match_2)
    match_2 = Dense(256,activation='softplus')(match_2)
    match_2 = Dense(story_maxlen,activation='softplus')(match_2)
    att_2 = Activation('sigmoid')(match_2)
    att_2 = RepeatVector(64)(att_2)
    att_2 = Permute((2,1))(att_2)

    response_2 = Multiply()([output_encoder_m,att_2])
    response_f_2 = encoder(response_2)
    response_all = Add()([response_f,response_f_2])

    response_h = RepeatVector(story_maxlen_sup)(response_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_all)
    answer = Concatenate(axis=1)([resp,question_output_encoder])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def varMem_key_multihop_v2(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    key_emb  = Embedding(vocab_size,64)
    key_emb_2 = Embedding(vocab_size,64)
    data_emb = Embedding(vocab_size,64)
    encoder = LSTM(64)

    input_encoder_m = key_emb(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)
    input_encoder_m_2 = key_emb_2(story)
    input_encoder_m_2 = Dropout(0.3)(input_encoder_m_2)
    output_encoder_m = data_emb(story)
    output_encoder_m = Dropout(0.3)(output_encoder_m)

    question_encoder = key_emb(query)
    question_encoder = Dropout(0.3)(question_encoder)
    question_encoder_2 = key_emb_2(query)
    question_encoder_2 = Dropout(0.3)(question_encoder_2)
    question_output_encoder = data_emb(query)
    question_output_encoder = Dropout(0.3)(question_output_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(story_maxlen)(match)
    match         = Dropout(0.3)(match)
    match         = Dense(256,activation='softplus')(match)
    match         = Dropout(0.3)(match)
    match_mu      = Dense(story_maxlen,activation=None)(match)
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)

    response   = Multiply()([input_encoder_m_2,att])
    response_f = encoder(response)
    #2nd hop
    response_tmp = Reshape((1,64))(response_f)
    query_new = Concatenate(axis=1)([response_tmp,question_encoder_2])
    match_2 = Dot(axes=[2,2])([query_new,input_encoder_m_2])
    match_2 = LSTM(256)(match_2)
    match_2 = Dropout(0.3)(match_2)
    match_2 = Dense(256,activation='softplus')(match_2)
    match_2 = Dropout(0.3)(match_2)
    match_2 = Dense(story_maxlen,activation=None)(match_2)
    att_2 = Activation('sigmoid')(match_2)
    att_2 = RepeatVector(64)(att_2)
    att_2 = Permute((2,1))(att_2)

    response_2 = Multiply()([output_encoder_m,att_2])
    response_f_2 = encoder(response_2)
    response_all = response_f_2

    response_h = RepeatVector(story_maxlen_sup)(response_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h1  = Dropout(0.3)(decode_h1)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_all)
    answer = Concatenate(axis=1)([resp,question_output_encoder])
    answer = LSTM(64,activation='relu')(answer)
    answer = Dropout(0.3)(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN

def varMem_2reader(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,64)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(story_maxlen)(match)
    match         = Dense(256,activation='softplus')(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    match2_mu     = Dense(story_maxlen,activation='softplus')(match)
    #match_log_var = Dense(story_maxlen,activation='sigmoid')(match)
    #att           = Lambda(sampling_normal)([match_mu,match_log_var])
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)
    att2          = Activation('sigmoid')(match2_mu)
    att2          = RepeatVector(64)(att2)
    att2          = Permute((2,1))(att2)

    response   = Multiply()([input_encoder_m,att])
    response_f = LSTM(64)(response)
    response2  = Multiply()([input_encoder_m,att2])
    response_f2= LSTM(64)(response2)
    resp_t     = Concatenate(axis=1)([response_f,response_f2])
    response_h = RepeatVector(story_maxlen_sup)(resp_t)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((2,64))(resp_t)
    answer = Concatenate(axis=1)([resp,question_encoder])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN



def multistep_model(batch_size,query_maxlen,story_maxlen,story_maxlen_sup,word_dim,vocab_size):
    story   = Input(batch_shape=(batch_size,story_maxlen))
    query   = Input(batch_shape=(batch_size,query_maxlen))
    word  = Embedding(vocab_size,64)

    input_encoder_m = word(story)
    input_encoder_m = Dropout(0.3)(input_encoder_m)

    question_encoder = word(query)
    question_encoder = Dropout(0.3)(question_encoder)

    match         = Dot(axes=[2,2])([question_encoder,input_encoder_m])
    match         = LSTM(256)(match)
    match         = Dense(256,activation='softplus')(match)
    match         = Dropout(0.3)(match)
    match_mu      = Dense(story_maxlen,activation='softplus')(match)
    #match_log_var = Dense(story_maxlen,activation='sigmoid')(match)
    #att           = Lambda(sampling_normal)([match_mu,match_log_var])
    att           = Activation('sigmoid')(match_mu)
    att           = RepeatVector(64)(att)
    att           = Permute((2,1))(att)


    response    = Multiply()([input_encoder_m,att])
    response_f1 = LSTM(64)(response)
    response_f1 = Reshape((1,64))(response_f1)
    #input_encoder_m2 = merge([input_encoder_m,response_f1],mode='concat',concat_axis=1)
    #match2      = merge([question_encoder,input_encoder_m2],mode='dot',dot_axes=[2,2])
    match2      = Dot(axes=[2,2])([input_encoder_m,response_f1])
    match2      = Flatten()(match2)
    #match2      = LSTM(256)(match2)
    match2       = Dense(256,activation='softplus')(match2)
    match2       = Dropout(0.3)(match2)
    match2_mu    = Dense(story_maxlen,activation='softplus')(match2)
    att2         = Activation('sigmoid')(match2_mu)
    att2         = RepeatVector(64)(att2)
    att2         = Permute((2,1))(att2)        


    response2  = Multiply()([input_encoder_m,att2])
    response_f2= LSTM(64)(response2)
    resp_all   = Flatten()(response_f1)
    resp_all   = Concatenate(axis=1)([resp_all,response_f2])
    response_h = RepeatVector(story_maxlen_sup)(resp_all)
    decode_h1  = LSTM(64,activation = 'relu',return_sequences=True)(response_h)
    decode_h2  = TimeDistributed(Dense(50,activation = 'relu'))(decode_h1)
    pred_recon = TimeDistributed(Dense(vocab_size,activation = 'linear'),name='pred_recon')(decode_h2)

    resp   = Reshape((1,64))(response_f2)
    answer = Concatenate(axis=1)([question_encoder,response_f1,resp])
    answer = LSTM(64,activation='relu')(answer)
    pred_y = Dense(vocab_size,activation='softmax',name='pred_y')(answer)
    
    
    prior_model = Model([story,query],pred_recon)
    prior_model.compile(optimizer='adam',loss='mse',clipvalue=0.5)

    VMemNN = Model([story,query],[pred_recon,pred_y])
    VMemNN.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],clipvalue=0.5)
    
    return prior_model,VMemNN
