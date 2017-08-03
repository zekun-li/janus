# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:15:37 2017

@author: yue_wu

This is a sample script for Janus Face Template Feature Pooling
"""
import os
if ( os.environ.has_key( 'SGE_GPU' ) ) :
    gpu_id = os.environ['SGE_GPU']
    os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32,lib.cnmem=1,blas.ldflags=\"-L/usr/lib64 -lblas\"" % gpu_id
    print "INFO: use SGE_GPU =", gpu_id
elif ( int(1) in range(8) ) :
    gpu_id = "1"
    os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32,blas.ldflags=\"-L/usr/lib64 -lblas\"" % gpu_id
    print "INFO: use Assigned_GPU =", gpu_id
else :
    os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"
    print "INFO: use CPU only"

import sys
import theano
keras_lib = "/nfs/isicvlnas01/users/yue_wu/thirdparty/keras_1.2.0/"
sys.path.insert( 0, keras_lib )
import keras
from keras.engine import Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, merge, Dropout, Activation, Reshape
from keras import backend as K
import numpy as np

def mean_pooling_over_subj_samples( x ) :
    return K.mean( x, axis = 1, keepdims = True ) # shape = ( nb_templates, 1, nb_feat_dims )

def std_pooling_over_subj_samples( x ) :
    return K.maximum( 1e-5, K.std( x, axis = 1, keepdims = True ) ) # shape = ( nb_templates, 1, nb_feat_dims )

def output_shape_of_my_pooling( input_shape ) :
    nb_tmplts, nb_samples, nb_feats = input_shape
    return tuple( [ nb_tmplts, 1, nb_feats ] )

def normalize_merge( x_tuple ) :
    x, x_avg, x_std  = x_tuple
    return K.clip( ( x - x_avg ) / x_std, -10, 10 ) # shape = (nb_templates, nb_images, nb_feat_dims )

def output_shape_of_normalize( input_shape_list ) :
    return input_shape_list[0]

def create_image_weight_predictor( input_shape = ( 2048, ) ) :
    '''Core module to predict image weight
    Below is just an example architecture, but might be modified later
    #TODO
    '''
    x_in = Input( shape = input_shape )
    x = Dropout( 0.25 )( x_in )
    x = Dense( 128, activation = 'relu' )( x )
    x = Dropout( 0.25 )( x )
    x = Dense( 8, activation = 'relu' )( x )
    x = Dropout( 0.25 )( x )
    x_ou = Dense( 1, activation = 'relu' )( x )
    return Model( input = x_in, output = x_ou, name = 'weight_predictor' )

def weighted_sum_merge( x_tuple ) :
    x, weight = x_tuple
    weight = K.expand_dims( weight, 2 )
    return K.sum( x * weight, axis = 1 )

def output_shape_weighted_sum( input_shape_list ) :
    nb_templates, nb_images, nb_feat_dims = input_shape_list[0]
    return tuple( [ nb_templates, nb_feat_dims ] )

def create_resnet_clf( input_shape = ( 2048, ), nb_classes = 10249, initial_weights = None ) :
    x = Input( shape = input_shape )
    y = Dense( nb_classes, activation = 'softmax' )( x )
    clf = Model( input = x, output = y, name = 'subj_predictor' )
    # load pretrained weights
    if ( initial_weights is not None ) :
        clf.load_weights( initial_weights )
    # freeze this classifier's weights
    for l in clf.layers :
        l.trainable = False
    return clf
    
#--------------------------------------------------------------------------------
# set arguments
#--------------------------------------------------------------------------------
nb_images, nb_feat_dims = 8, 2048
pretrain_clf_weight_file = None #'FILE_PATH_TO_FILL'
nb_subjects = 10000
#--------------------------------------------------------------------------------
# define modules
#--------------------------------------------------------------------------------
img_tmplt_shape = ( nb_images, nb_feat_dims )
weight_predictor = create_image_weight_predictor()
subj_clf = create_resnet_clf( initial_weights = pretrain_clf_weight_file, nb_classes = nb_subjects, input_shape = ( nb_feat_dims, ) )
#--------------------------------------------------------------------------------
# define end-to-end model
#--------------------------------------------------------------------------------
# input
tmplt_in = Input( shape = img_tmplt_shape ) # ( nb_templates, nb_images, nb_feat_dims )
# compute template-wise stats, output shape = ( nb_templates, 1, nb_feat_dims )
tmplt_avg = Lambda( function = mean_pooling_over_subj_samples, output_shape = output_shape_of_my_pooling, name = 'avg_pooling' )( tmplt_in )
tmplt_std = Lambda( function = std_pooling_over_subj_samples, output_shape = output_shape_of_my_pooling, name = 'std_pooling' )( tmplt_in )
# normalize input, output shape = ( nb_templates, nb_images, nb_feat_dims )
tmplt_norm = merge( [ tmplt_in, tmplt_avg, tmplt_std ], mode = normalize_merge, output_shape = output_shape_of_normalize, name = 'normalize' )
# compute weight
tmplt_raw_weight = TimeDistributed( weight_predictor, name = 'pred_image_weight' )( tmplt_norm )
tmplt_flat_weight = Reshape( ( nb_images, ) )( tmplt_raw_weight )
tmplt_norm_weight = Activation ( activation = 'softmax' )( tmplt_flat_weight )
# compute pooled feature
tmplt_pool = merge( [ tmplt_in, tmplt_norm_weight ], mode = weighted_sum_merge, output_shape = output_shape_weighted_sum, name = 'feat_pooling' )
tmplt_subj = subj_clf( tmplt_pool )
# end-to-end model
end_to_end_model = Model( input = tmplt_in, output = tmplt_subj, name = 'face_template_pooling' )
print end_to_end_model.summary()

#--------------------------------------------------------------------------------
# test forward
#--------------------------------------------------------------------------------
pool_model = Model( input = tmplt_in, output = tmplt_pool, name = 'face_template_featex' )
weight_model = Model( input = tmplt_in, output = tmplt_norm_weight, name = 'face_template_weight' )

a = np.random.rand( 2, nb_images, nb_feat_dims )
b = end_to_end_model.predict( a )
print b.shape
print b.sum( axis = -1 ) # output should be close to 1
# get weights
c = weight_model.predict( a )
print c
print c.sum( axis = -1 ) # output should be close to 1
# get pooled feature
d = pool_model.predict( a )
# verify pooled feature via numpy
e = np.sum( a * np.expand_dims( c, axis = 2 ), axis = 1 )
print np.abs( d - e ).mean() # output should be close to 0
