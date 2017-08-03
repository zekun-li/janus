# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:17:07 2017

@author: yue_wu
"""
import os
import numpy as np
import lmdb
from keras.utils.np_utils import to_categorical
import pickle

class LMDBSequence( object ) :
    items = []
    def __init__( self, lmdb_dir, key_list_file, lmdb_id = 0 ) :
        assert os.path.isdir( lmdb_dir ), "ERROR: cannot find input lmdb_dir %s" % lmdb_dir
        self.lmdb_env = lmdb.open( lmdb_dir, readonly = True, lock = False )
        assert os.path.isfile( key_list_file ), "ERROR: cannot find input lmdb_key_file %s" % key_list_file
        self.old2newTable = self._old2new() 
        self.key_list_file = key_list_file
        self.lmdb_id = lmdb_id
        self._prepare()
        return
    def _prepare( self ) :
        '''prepare LMDB keys and associated target ids
        '''
        with open( self.key_list_file, 'r' ) as IN :
            keys = [ line.strip() for line in IN.readlines() ]
        tids = []
        for key in keys :
            tid = self._parse_id_from_key( key )
            tids.append( tid )
        self.items = zip( keys, tids )
        return

    # conver mow(old-cow) labels to cow labels
    def _old2new( self):
        convert_file = '/nfs/isicvlnas01/projects/glaive/expts/00080-zekun-label-convert/old2new_label.pickle'
        with open(convert_file,'r') as old2new_f:
            old2new = pickle.load(old2new_f)
        return old2new

    def _parse_id_from_key( self, key ) :
        '''parse target id from a given key file
        '''
        old_label =  key.split('_')[0]
        new_label = self.old2newTable[old_label]
        return int(new_label)

    def __len__( self ) :
        return len( self.items ) 
    def __getitem__( self, idx ) :
        key = None
        with self.lmdb_env.begin() as txn :
            try :
                key, tid = self.items[ idx ]
                feat = np.frombuffer( txn.get( key ), dtype = 'float32' )
            except Exception, e :
                if ( key ) :
                    print "WARNING: fail to retrieve feature using key", key, e
                else :
                    print "WARNING: fail to get a valid key", e
                return None
        return feat


class TemplateSizeSeq( object ) :
    precomputed_tmplt_size = None
    def __init__( self, 
                  nb_batches_per_epoch = 1000,
                  tmplt_size_proba_list = np.ones( [20, ] ) / 20., 
                  mode = 'training' ) :
        self.mode = mode 
        self.nb_batches_per_epoch = nb_batches_per_epoch
        self.tmplt_size_proba_list = tmplt_size_proba_list
        self._parse_min_max_tmplt_size()
        self.nb_sizes = len( tmplt_size_proba_list )
    def _precompute_test_tmplt_size( self ) :
        cum_proba = np.cumsum( self.tmplt_size_proba_list )
        th_list = self.nb_batches_per_epoch * cum_proba
        precomputed_tmplt_size = np.zeros( [ self.nb_batches_per_epoch, ], dtype = int )
        th0 = 0
        for tsize, th1 in zip( self.tmplt_size_list, th_list ) :
            th1 = int(np.ceil( th1 ))
            precomputed_tmplt_size[ th0:th1 ] = tsize
            th0 = th1
        return precomputed_tmplt_size
    def _parse_min_max_tmplt_size( self ) :
        min_size = 2
        max_size = min_size + len( self.tmplt_size_proba_list )
        self.tmplt_size_list = range( min_size, max_size  )
        return
    def __getitem__( self, idx ) :
        if ( self.mode == 'training' ) :
            tmplt_size = np.random.choice( self.tmplt_size_list, size = 1, p = self.tmplt_size_proba_list )
        else :
            if ( self.precomputed_tmplt_size is None ) :
                self.precomputed_tmplt_size = self._precompute_test_tmplt_size()
            tmplt_size = self.precomputed_tmplt_size[ idx % self.nb_sizes ]
        return tmplt_size

class DataGenerator( object ) :
    idx = 0
    def __init__( self, lmdb_dir_and_key_pairs, 
                  mode = 'training', 
                  nb_batches_per_epoch = 1000, 
                  max_feat_per_batch = 256,
                  tmplt_size_proba_list = np.ones( [20,] ) / 20.,
                  prng_seed = 12345, 
                  generate_lmdb_id = False ) :
        self.lmdb_seqs = []
        for lmdb_id, lmdb_pair in enumerate( lmdb_dir_and_key_pairs ) :
            lmdb_dir, key_file  = lmdb_pair
            partition_lmdb = LMDBSequence( lmdb_dir, key_file, lmdb_id )
            self.lmdb_seqs.append( partition_lmdb )
        self.nb_partitions = len( self.lmdb_seqs )
        self.mode = mode
        self.tmplt_seq = TemplateSizeSeq( nb_batches_per_epoch, tmplt_size_proba_list, mode )
        self.max_feat_per_batch = max_feat_per_batch
        np.random.seed( prng_seed )
        self.tid_2_sample_lut = self._prepare_tid_to_sample_lut()
        self.nb_batches_per_epoch = nb_batches_per_epoch
        self.tids = self.tid_2_sample_lut.keys()
        self.nb_tids = np.max( self.tids ) + 1 # tid starts with 0
        self.generate_lmdb_id = generate_lmdb_id
        return
    def _prepare_tid_to_sample_lut( self ) :
        mega_lut = dict()
        # group info by target_id
        for lmdb_id, lmdb_par in enumerate( self.lmdb_seqs ) :
            idx_2_key_tid_lut = lmdb_par.items
            for idx, key_tid in enumerate( idx_2_key_tid_lut ) :
                key, tid = key_tid
                if ( not mega_lut.has_key( tid ) ) :
                    mega_lut[ tid ] = []
                mega_lut[ tid ].append( [ lmdb_id, idx ] )
                # we can fetch this sample by using
                # self.lmdb_seqs[ lmdb_id ][ idx ]
        return mega_lut
    def __len__( self ) :
        return self.nb_batches_per_epoch
    def __getitem__( self, idx ) :
        #print idx
        # determine the template size of current batch
        cur_tsize = self.tmplt_seq[idx]
        # determine the set of target ids in current batch
        cur_bsize = self.max_feat_per_batch // cur_tsize         
        if ( self.mode == 'training' ) :
            # randomly select cur_bsize of target ids
            cur_tids = np.random.choice( self.tids, cur_bsize )
        elif ( self.mode == 'testing' or self.mode == 'validating' ):
            # reset prng to ensure the identical testing sample generation
            np.random.seed( idx % self.nb_batches_per_epoch )
            # select cur_bsize of target ids
            cur_tids = np.random.choice( self.tids, cur_bsize )
        else :
            raise NotImplementedError, "ERROR: mode = %s is NOT supported" % self.mode
        X, L = [], []
        #print 'cur_tsize, cur_bsize,cur_tids'
        #print cur_tsize, cur_bsize, cur_tids
        for tid in cur_tids :
            # foreach target id, pick cur_tsize of features
            nb_tsamples = len( self.tid_2_sample_lut[ tid ] ) # [lmdb_id, feat_idx]
            tid_indices = np.random.randint( 0, nb_tsamples, cur_tsize )
            tX, tL = [], []
            # retrieve each feature from lmdb on the fly
            for sidx in tid_indices :
                lmdb_id, idx = self.tid_2_sample_lut[ tid ][ sidx ]
                feat = self.lmdb_seqs[ lmdb_id ][ idx ] # (2048,)
                tX.append( feat )
                tL.append( lmdb_id )
            # concat all features belonging to this target id
            #tX = np.row_stack( feat ) # ( cur_tsize, 2048 )
            tX = np.row_stack( tX ) # ( cur_tsize, 2048 )
            X.append( np.expand_dims( tX, axis = 0 ) )
            # concat all lmdb_ids belonging to this target id
            tL = to_categorical( np.array( tL ), self.nb_partitions ) # ( cur_tsize, nb_partitions )
            L.append( np.expand_dims( tL, axis = 0 ) )
        # concat all features belonging to this batch
        X = np.concatenate( X, axis = 0 ) # ( cur_bsize, cur_tsize, 2048 )
        # concat all lmdb_ids belonging to this batch
        L = np.concatenate( L, axis = 0 ) # ( cur_bsize, cur_tsize, nb_partitions )
        # concat all target ids belonging to this batch
        Y = to_categorical( np.array( cur_tids ), self.nb_tids ) # ( cur_bsize, nb_tids )
        # generate output
        if ( self.generate_lmdb_id ) :
            return ( { 'feat' : X, 'part' : L }, { 'pred' : Y } )
        else :
            return ( { 'feat' : X }, { 'pred' : Y } )
    def __iter__( self ) :
        return self
    def next( self ) :
        idx = self.idx
        self.idx = ( self.idx + 1 ) % self.nb_batches_per_epoch   
        return self[idx]
