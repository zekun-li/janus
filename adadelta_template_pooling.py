import sys
import os
import numpy as np
gpu_id = '1'
#os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32,profile=True,profile_memory=True" % gpu_id
os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32" % gpu_id
print os.environ["THEANO_FLAGS"]
#sys.path.insert(1,"/nfs/isicvlnas01/users/yue_wu/thirdparty/keras_1.2.0/")
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout,Dense,Input,merge
from keras.legacy.layers import Merge
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint,CSVLogger
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import keras_utils
import random

#nb_class_labels = 68465
nb_class_labels = 68906
len_feature = 2048
tmp_dir= '/lfs2/tmp/zekunl/'
 
# x is 3D tensor, nb_subjects x nb_photos_per_subject x len_feature
def mean_subjects(x):
    return tt.mean(x, axis = 1, keepdims = True)

def std_subjects(x):
    return tt.maximum(1e-5, tt.std(x, axis = 1, keepdims = True )) # avoid zero as divider

def stats_subjects_shape(input_shape):
    return tuple((input_shape[0], 1, input_shape[-1]))

def normalized_subjects(input_tensor):
    x, means, stds = input_tensor
    normed_x = (x-means)/stds
    normed_x = tt.clip(normed_x, -10, 10)
    return normed_x

def normalized_output_shape(input_shape):
    return input_shape[0]

def softmax3d( x ) :
    ndim = K.ndim(x)
    e = K.exp( x - K.max(x, axis=1, keepdims=True))
    s = K.sum( e, axis=1, keepdims=True)
    return e / s

def weight_predictor( len_feature = 2048 ) :
    feat_in = Input( shape = ( None, len_feature ) ) # 3d, nb_sample, nb_image,nb_feat
    feat_d1 = Dropout( 0.25 )( feat_in )
    feat_c1 = Conv1D( 128, 1, padding = 'valid', activation = 'relu' )( feat_d1 )
    feat_d2 = Dropout( 0.25 )( feat_c1 )
    feat_c2 = Conv1D( 8, 1, padding = 'valid', activation = 'relu' )( feat_d2 )
    feat_d3 = Dropout( 0.25 )( feat_c2 )
    feat_c3 = Conv1D( 1, 1, padding = 'valid', activation = softmax3d )( feat_d3 )
    model = Model( inputs = feat_in, outputs = feat_c3, name = 'weight_predictor' )
    return model

def fun_weighted_sum(in_tensor):
    all_people, weights = in_tensor
    # make last dimension of weights broadcastable
    weights = weights.reshape((weights.shape[0],-1))
    weights = K.expand_dims(weights,2)
    return K.sum(all_people*weights, axis = 1)

# return 2D tensor, nb_people x len_features
def fun_weighted_sum_shape(input_shapes):
    all_people_shape, _ = input_shapes
    return tuple((all_people_shape[0],all_people_shape[-1]))

def classifier(initial_weights = tmp_dir +'big_caffe_clf_weights.pkl'):
    clf = Sequential(name = "pred")
    clf.add(Dense(nb_class_labels, activation = 'softmax', input_shape = (len_feature,)))
    
    if (initial_weights is not None):
        import cPickle as pickle
        with open(initial_weights,"r") as f:
            weights = pickle.load(f)
        clf.layers[0].set_weights(weights)
    # freeze weights
    for l in clf.layers:
        l.trainable = False
    
    return clf

def template_classifier():
    # ------------------------normalize subjects ---------------------
    all_people = Input(shape=(None, len_feature),name = 'feat') # nb_photos_per_person x feature_vector_length
    mean_all_people = Lambda(mean_subjects, output_shape = stats_subjects_shape, name = 'mean layer')(all_people)
    std_all_people = Lambda(std_subjects, output_shape = stats_subjects_shape, name = 'std layer')(all_people)
    normed_subjects = merge(inputs = [all_people, mean_all_people, std_all_people], mode = normalized_subjects, output_shape = normalized_output_shape, name = 'normalization layer')
    

    # --------------------------predict weights and compute weighted sum-------------------
    weights = weight_predictor()(normed_subjects)
    
    pooled_template = merge(inputs = [all_people, weights], mode = fun_weighted_sum, output_shape = fun_weighted_sum_shape, name = 'weighted_sum')
    softmax_proba = classifier()(pooled_template)     
    
    model = Model(inputs = [all_people],outputs = [softmax_proba])
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # ---------------------------average pooling --------------
    
    avgpool_softmax = classifier()(mean_all_people)
    mean_model = Model(inputs = [all_people], outputs = [avgpool_softmax])
    mean_model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # ------------------------- for testing -----------------------
    norm_model = Model(inputs=[all_people], outputs = [normed_subjects])
    norm_model.compile(optimizer = 'adadelta', loss = 'mse')
    
    weight_model = Model(inputs=[all_people], outputs = [weights])
    weight_model.compile(optimizer = 'adadelta', loss = 'mse')

    pool_model = Model(inputs = [all_people],outputs = [pooled_template])
    pool_model.compile(optimizer = 'adadelta', loss = 'mse', metrics = ['accuracy'])


    return model,mean_model, norm_model, weight_model, pool_model
        

def test_nn():
    inputX = np.random.randint(10,size = (2,3,len_feature))
    
    model, mean_model, norm_model, weight_model, pool_model = template_classifier()
    
    print 'inputX:', inputX

    normed_input =  norm_model.predict(inputX)
    print 'mean and std of normed_input:'
    print np.mean(normed_input), np.std(normed_input) # close to [0,1]
    
    weights = weight_model.predict(inputX)
    print 'weights:'
    print weights.shape

    means =  mean_model.predict(inputX)
    print 'means'
    print means

    model_pool_out =  pool_model.predict(inputX) 
    print 'pooled_template:'
    print model_pool_out
    
    print 'model output:'
    out = model.predict(inputX) # softmax probabilities. sum up to 1
    print np.sum(out, axis = 1)
    
    np_verify_out = np.sum(inputX * weights, axis = 1)
    print np.abs(model_pool_out - np_verify_out).sum() # close to 0
    
def train_with_generator(snap_weight = None):
    #key_file_base = '/nfs/isicvlnas01/users/zekunl/projects/janus/keyfile/keys/'
    key_file_base = tmp_dir
    lmdb_base = tmp_dir
    feat_per_batch = 512


    lmdb_file0 = 'comb-featexCOW-crop.lmdb'
    lmdb_file1 = 'comb-featexCOW-real.lmdb'
    lmdb_file2 = 'comb-featexCOW-render-45.lmdb'
    lmdb_file3 = 'comb-featexCOW-render-75.lmdb'
    lmdb_file4 = 'comb-featexCOW-render-frontal.lmdb'
    mode = 'train'
    key_file0 = mode + '-' +lmdb_file0.split('.lmdb')[0] + '.key'
    key_file1 = mode + '-' +lmdb_file1.split('.lmdb')[0] + '.key'
    key_file2 = mode + '-' +lmdb_file2.split('.lmdb')[0] + '.key'
    key_file3 = mode + '-' +lmdb_file3.split('.lmdb')[0] + '.key'
    key_file4 = mode + '-' +lmdb_file4.split('.lmdb')[0] + '.key'

    p_array = [ 0.29991916,  0.20776071,  0.14389652,  0.09943411,  0.06952304, 0.04850445,  0.03395311,  0.02425222,  0.01697656,  0.01293452, 0.00970089,  0.00727567,  0.00565885,  0.00485044,  0.00404204, 0.00323363,  0.00323363,  0.00242522,  0.00242522]

    trn = keras_utils.DataGenerator( [[lmdb_base + lmdb_file0, key_file_base + key_file0],[lmdb_base + lmdb_file1, key_file_base + key_file1],[lmdb_base + lmdb_file2, key_file_base + key_file2],[lmdb_base + lmdb_file3, key_file_base + key_file3],[lmdb_base + lmdb_file4, key_file_base + key_file4]], mode = 'training', nb_batches_per_epoch = 2000, max_feat_per_batch = feat_per_batch, tmplt_size_proba_list = p_array )
    #print trn[0][0]['feat'].shape,np.nonzero(trn[0][1]['pred'])

    val_mode = 'val'
    val_key_file0 = val_mode + '-' +lmdb_file0.split('.lmdb')[0] + '.key'
    val_key_file1 = val_mode + '-' +lmdb_file1.split('.lmdb')[0] + '.key'
    val_key_file2 = val_mode + '-' +lmdb_file2.split('.lmdb')[0] + '.key'
    val_key_file3 = val_mode + '-' +lmdb_file3.split('.lmdb')[0] + '.key'
    val_key_file4 = val_mode + '-' +lmdb_file4.split('.lmdb')[0] + '.key'
    
    val = keras_utils.DataGenerator( [[lmdb_base + lmdb_file0, key_file_base + val_key_file0],[lmdb_base + lmdb_file1, key_file_base + val_key_file1],[lmdb_base + lmdb_file2, key_file_base + val_key_file2],[lmdb_base + lmdb_file3, key_file_base + val_key_file3],[lmdb_base + lmdb_file4, key_file_base + val_key_file4]], mode = 'validating', nb_batches_per_epoch = 2000, max_feat_per_batch = feat_per_batch, tmplt_size_proba_list = p_array )

    prefix = 'adadelta-featperbatch'+str(feat_per_batch)
    model, mean_model, _, _, _ = template_classifier()
    model.summary()
    if snap_weight is not None:
        assert os.path.isfile(snap_weight)
        print 'loading weights from ',snap_weight
        model.load_weights(snap_weight)


    csv_logger = CSVLogger('logs/'+prefix+'-train.log')
    model_save_path = 'weights/'+prefix+'-{epoch:02d}-{val_loss:.2f}.hdf5'
    model_save_best = 'weights/best-'+prefix+'-{epoch:02d}-{val_loss:.2f}.hdf5'
    check_point = ModelCheckpoint(model_save_path, save_best_only=False,save_weights_only=False,period = 10)
    check_point_best = ModelCheckpoint(model_save_best, save_best_only=True,save_weights_only=False)
    model.fit_generator(trn, steps_per_epoch = 2000, epochs = 500, validation_data = val, validation_steps = 2000, callbacks = [csv_logger, check_point, check_point_best])


def test_avgpool_with_generator():
    #key_file_base = '/nfs/isicvlnas01/users/zekunl/projects/janus/keyfile/keys/'
    key_file_base = tmp_dir
    lmdb_base = tmp_dir
    lmdb_file0 = 'comb-featexCOW-crop.lmdb'
    lmdb_file1 = 'comb-featexCOW-real.lmdb'
    lmdb_file2 = 'comb-featexCOW-render-45.lmdb'
    lmdb_file3 = 'comb-featexCOW-render-75.lmdb'
    lmdb_file4 = 'comb-featexCOW-render-frontal.lmdb'
    mode = 'test'
    key_file0 = mode + '-' +lmdb_file0.split('.lmdb')[0] + '.key'
    key_file1 = mode + '-' +lmdb_file1.split('.lmdb')[0] + '.key'
    key_file2 = mode + '-' +lmdb_file2.split('.lmdb')[0] + '.key'
    key_file3 = mode + '-' +lmdb_file3.split('.lmdb')[0] + '.key'
    key_file4 = mode + '-' +lmdb_file4.split('.lmdb')[0] + '.key'

    p_array = [ 0.29991916,  0.20776071,  0.14389652,  0.09943411,  0.06952304, 0.04850445,  0.03395311,  0.02425222,  0.01697656,  0.01293452, 0.00970089,  0.00727567,  0.00565885,  0.00485044,  0.00404204, 0.00323363,  0.00323363,  0.00242522,  0.00242522]

    test = keras_utils.DataGenerator( [[lmdb_base + lmdb_file0, key_file_base + key_file0],[lmdb_base + lmdb_file1, key_file_base + key_file1],[lmdb_base + lmdb_file2, key_file_base + key_file2],[lmdb_base + lmdb_file3, key_file_base + key_file3],[lmdb_base + lmdb_file4, key_file_base + key_file4]], mode = 'testing', tmplt_size_proba_list = p_array, max_feat_per_batch = 512, nb_batches_per_epoch = 6000 )

    model, mean_model, _, _, _ = template_classifier()
    mean_model.summary()
    print model.evaluate_generator(test, steps=2, max_q_size=10, workers=1)
    

if __name__ == "__main__":
    #test_nn()
    train_with_generator()
    #test_avgpool_with_generator()
