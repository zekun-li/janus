import sys
import os
import numpy as np
gpu_id = '0'
os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32" % gpu_id
#os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32,exception_verbosity=high" % gpu_id
print os.environ["THEANO_FLAGS"]
sys.path.insert(1,"/nfs/isicvlnas01/users/yue_wu/thirdparty/keras_1.2.0/")
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout,Dense,Input,merge
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Lambda
from keras.models import Model
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import keras_utils
import random

#nb_class_labels = 68465
nb_class_labels = 68906
len_feature = 2048
input_lmdb = '/nfs/isicvlnas01/projects/glaive/expts/00036-zekun-featex-aug_resnet_new_layer_mow_cfd_0.4_3_clean/expts/res101-feat/feat_lmdb'
 
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

def weight_predictor():
    predictor = Sequential(name = 'weight_predictor')
    predictor.add(Dense(128, activation = 'relu', input_shape = (len_feature,) ))
    predictor.add(Dropout(0.25))
    predictor.add(Dense(8, activation = 'relu'))
    predictor.add(Dropout(0.25))
    predictor.add(Dense(1, activation = 'relu'))
    return predictor

def softmax3d( x ) :
    ndim = K.ndim(x)
    e = K.exp( x - K.max(x, axis=1, keepdims=True))
    s = K.sum( e, axis=1, keepdims=True)
    return e / s
'''
def weight_predictor( len_feature = 2048 ) :
    #feat_in = Input( shape = ( None, len_feature ) ) # 3d, nb_sample, nb_image,nb_feat
    #feat_d1 = Dropout( 0.25 ,input_shape = (None,len_feature) )
    predictor = Sequential( name = 'weight_predictor')
    predictor.add( Conv1D( 128, 1, padding = 'valid', activation = 'relu' , input_shape = (None, len_feature) ))
    predictor.add( Dropout( 0.25 ))
    predictor.add( Conv1D( 8, 1, padding = 'valid', activation = 'relu' ))
    predictor.add( Dropout( 0.25 ))
    predictor.add( Conv1D( 1, 1, padding = 'valid', activation = softmax3d ))
    #model = Model( inputs = feat_in, outputs = feat_c3, name = 'weight_predictor' )
    #return model
    return predictor

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
'''

def fun_softmax_weights(x):
    return softmax(x.reshape((x.shape[0],-1)))

def fun_weighted_sum(in_tensor):
    all_people, weights = in_tensor
    weights = K.expand_dims(weights,2)
    return K.sum(all_people*weights, axis = 1)
    #return all_people
# return 2D tensor, nb_people x len_features
def weighted_sum_shape(input_shapes):
    all_people_shape, _ = input_shapes
    return all_people_shape

def fun_weighted_sum_shape(input_shapes):
    all_people_shape, _ = input_shapes
    return tuple((all_people_shape[0],all_people_shape[-1]))

def classifier(initial_weights = 'caffe/big_caffe_clf_weights.pkl'):
    clf = Sequential(name = "classifier")
    clf.add(Dense(nb_class_labels, activation = 'softmax', input_shape = (len_feature,)))
    '''
    if (initial_weights is not None):
        import cPickle as pickle
        with open(initial_weights,"r") as f:
            weights = pickle.load(f)
        clf.layers[0].set_weights(weights)
    # freeze weights
    for l in clf.layers:
        l.trainable = False
    '''
    return clf

def template_classifier():
    # ------------------------normalize subjects ---------------------
    all_people = Input(shape=(None, len_feature),name = 'input') # nb_photos_per_person x feature_vector_length
    mean_all_people = Lambda(mean_subjects, output_shape = stats_subjects_shape, name = 'mean layer')(all_people)
    std_all_people = Lambda(std_subjects, output_shape = stats_subjects_shape, name = 'std layer')(all_people)
    normed_subjects = merge(inputs = [all_people, mean_all_people, std_all_people], mode = normalized_subjects, output_shape = normalized_output_shape, name = 'normalization layer')
    

    # --------------------------predict weights and compute weighted sum-------------------
    weights = TimeDistributed(weight_predictor())(normed_subjects)
    #weights = weight_predictor()(normed_subjects)
    
    softmax_weights = Lambda(fun_softmax_weights, output_shape=(None, None), name = 'softmax layer')(weights)
    #debug_template = merge(inputs = [all_people,all_people], mode = fun_weighted_sum, output_shape = weighted_sum_shape)
    
    pooled_template = merge(inputs = [all_people, softmax_weights], mode = fun_weighted_sum, output_shape = fun_weighted_sum_shape, name = 'weighted_sum')
    softmax_proba = classifier()(pooled_template) 
    
    mean_model = Model(input = [all_people], output = [mean_all_people])
    mean_model.compile(optimizer = 'adadelta', loss = 'mse')
    

    model = Model(input = [all_people],output = [softmax_proba])
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    # ---------------------------for testing --------------
    
    norm_model = Model(input=[all_people], output = [normed_subjects])
    norm_model.compile(optimizer = 'adadelta', loss = 'mse')
    
    weight_model = Model(input=[all_people], output = [weights])
    weight_model.compile(optimizer = 'adadelta', loss = 'mse')
    
    weight_model1 = Model(input=[all_people], output = [softmax_weights])
    weight_model1.compile(optimizer = 'adadelta', loss = 'mse')
    
    pool_model = Model(input = [all_people],output = [pooled_template])
    pool_model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model,norm_model, weight_model, weight_model1,pool_model,mean_model
    #return model,norm_model, pool_model, weight_model

def data_generator(data_list, mode = 'training', nb_epoch = -1, batch_size = 3):
    # hardcode probability to generate num_photos_per_person
    p_array = [ 0.29991916,  0.20776071,  0.14389652,  0.09943411,  0.06952304, 0.04850445,  0.03395311,  0.02425222,  0.01697656,  0.01293452, 0.00970089,  0.00727567,  0.00565885,  0.00485044,  0.00404204, 0.00323363,  0.00323363,  0.00242522,  0.00242522]
    size_array = np.arange(21)[2:]
    num_photos_per_person = np.random.choice(size_array, size=1,replace = False, p=p_array)[0]
    epoch = 0
    nb_subjects = len(data_list)
    index_array = range(nb_subjects)
    batch_index = 0
    while ( epoch < nb_epoch ) or (nb_epoch < 0):
        if batch_index == 0:
            if (mode == "training"):
                np.random.shuffle(index_array)
        current_index = (batch_index * batch_size) % nb_subjects
        if nb_subjects > current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = nb_subjects - current_index
            batch_index = 0

        pick_indices = index_array[current_index: current_index + current_batch_size]
        buffer = None
        features = []
        feature_labels = []
        for subject_idx in pick_indices:
            subject = data_list[subject_idx]
            num_photos = len(subject['features'])
            if num_photos >= num_photos_per_person:
                # add (num_photos_per_person) number of photos to buffer
                pick_photo_idx = random.sample(range(num_photos), num_photos_per_person)
               
            else:
                # add all photos to buffer and append (num_photos_per_person - num_photos) to buffer
                repeat_photo_idx = random.sample(range(num_photos), (num_photos_per_person - num_photos))
                pick_photo_idx = range(num_photos)+repeat_photo_idx

            if features == []:
                features = data_list[subject_idx]['features'][[pick_photo_idx]][np.newaxis,:]
                feature_labels = data_list[subject_idx]['labels']
                feature_labels = to_categorical(feature_labels, nb_class_labels )
            else:
                features = np.vstack((features, data_list[subject_idx]['features'][[pick_photo_idx]][np.newaxis,:]))
                tmp_labels = data_list[subject_idx]['labels']
                feature_labels = np.vstack((feature_labels, to_categorical(tmp_labels,nb_class_labels)))


        #print "epoch: ", epoch, "batch_idx", current_index, "batch size:", current_batch_size

        epoch = epoch +1
        buffer = (features,feature_labels)
        if ( buffer is not None):
            yield buffer
        

def test_nn():
    inputX = np.random.randint(10,size = (2,3,len_feature))
    model,norm_model, weight_model,weight_model1, pool_model, mean_model = template_classifier()
    #normed_input =  norm_model.predict(inputX)
    #print np.mean(normed_input), np.std(normed_input) # close to [0,1]
    
    weights = weight_model.predict(inputX)
    print weights.shape, weights
    weights = weight_model1.predict(inputX)
    print weights.shape, weights

    model_pool_out =  pool_model.predict(inputX) 
    print model_pool_out

    print mean_model.predict(inputX).shape
    
    print model.predict(inputX) # softmax probabilities. sum up to 1

    
    #np_verify_out = np.sum(inputX * np.expand_dims(weights, axis = 2), axis = 1)
    #print np.abs(model_pool_out - np_verify_out).sum() # close to 0
    
def test_generator():
    
    #trn = data_generator(dummy_file, mode = 'training', nb_epoch = 34, batch_size = 3)
    for feat, label in trn:
        print feat['feat'].shape, label['pred'].shape

def training():
    import cPickle as pickle
    with open("dummy_file_small.pkl","r") as f:
        dummy_file = pickle.load(f)
    trn = data_generator(dummy_file, mode = 'training', nb_epoch = 51, batch_size = 2)
    model, _,_,_ = template_classifier()
    model.fit_generator(trn, samples_per_epoch = 10,nb_epoch = 10,max_q_size = 2)


if __name__ == "__main__":
    test_nn()
    #test_generator()
    #training()
