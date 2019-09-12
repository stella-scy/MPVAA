#-*- coding: UTF-8 -*-

import numpy
import copy
import logging


class Data():

    def __init__(self, path_to_data, word2idx, sep=None, batch_size=128, minlen=1, maxlen=150, fresh=False):
        self.sep = sep
        self.batch_size = batch_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.fresh = fresh
        
        self.read_data(path_to_data)
        self.prepare(word2idx)
        self.reset()

    def read_data(self, path_to_data):
        with open(path_to_data, 'rU') as fin:
            lines = fin.readlines()
       
        self.text = list()
        self.total = len(lines)
        for i in range(self.total):
            line = lines[i].strip(' \n')
            self.text.append(line)
            
            if (i+1) % 10000000 == 0:
                logging.info('reading data line %d' % (i+1))
        logging.info('reading data line %d' % self.total)

    def prepare(self, word2idx):
        self.idxs = list()
        self.lengths = list()
        
        for i in range(self.total):
            sent_len, sent_true = self.sent_judge(self.text[i], word2idx)
            if sent_true:
                self.idxs.append( i )
                self.lengths.append( sent_len )

        print('self.idxs', self.idxs)
        self.qtotal = len(self.idxs)
        print('self.qtotal', self.qtotal)
        self.len_unique = numpy.unique(self.lengths)
        print('self.len_unique', self.len_unique)
        self.len_indices = dict()
        self.len_counts = dict()
        print('self.lengths', self.lengths)
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            print('numpy.where(self.lengths == ll)', numpy.where(self.lengths == ll))
            print('self.len_indices[ll]', self.len_indices[ll])
            self.len_counts[ll] = len(self.len_indices[ll])
            print('self.len_counts[ll]', self.len_counts[ll])

    def sent_judge(self, sent, word2idx):
        words = sent.split(self.sep)
        nwords = len(words)
        lenTrue = ((nwords <= self.maxlen) and (nwords >= self.minlen))
        if not lenTrue:
            return nwords, False
        else:
            unk_count = 0
            for w in words:
                #if not word2idx.has_key(w):
                if w not in word2idx:
                    unk_count += 1
            if unk_count > 0:
                return nwords, False
            else:
                return nwords, True

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        print('self.len_counts', self.len_counts)
        print('self.len_curr_counts', self.len_curr_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        print('orig self.len_unique', self.len_unique)
        print('self.len_unique', self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        print('self.len_indices_pos', self.len_indices_pos)
        print('self.len_indices', self.len_indices)
        print('later self.len_indices', self.len_indices)
        self.len_idx = -1

    def __next__(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            print('self.len_idx', self.len_idx)
            print('self.len_curr_counts[self.len_unique[self.len_idx]]', self.len_curr_counts[self.len_unique[self.len_idx]])
            print('self.len_curr_counts', self.len_curr_counts)
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        print('len(self.len_unique)', len(self.len_unique))
        print('count', count)
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()
            #return

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        print('self.len_unique', self.len_unique)
        print('self.len_indices', self.len_indices)
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        print('self.len_indices_pos', self.len_indices_pos)
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size
        print('self.len_curr_counts', self.len_curr_counts)

        if self.fresh:
            self.reset()
        print('curr_indices', curr_indices)
        batch_data = [self.text[self.idxs[i]] +' </s>' for i in curr_indices]
        return batch_data

    def __iter__(self):
        return self


def prepare_data(batch_data, word2vec, word2vec_2, word2vec_3, word2idx, word_dim=16, sep=None):

    batch_data_ = list()
    for i in range(len(batch_data)):
        batch_data_.append( [w for w in batch_data[i].split(sep) if w in word2vec] )

    lens = list()
    for i in range(len(batch_data_)):
        lens.append( len(batch_data_[i]) )
    max_len = numpy.max(lens)
    n_batches = len(batch_data_)

    x = numpy.zeros((n_batches, max_len, word_dim), dtype='float32')
    x_mask = numpy.zeros((n_batches, max_len), dtype='int32')
    y = numpy.zeros((n_batches, max_len, word_dim), dtype='float32')
    y_mask = numpy.zeros((n_batches, max_len), dtype='int32')
    y_target = numpy.zeros((n_batches, max_len), dtype='int32')

    batch_data_2_ = list()
    for i in range(len(batch_data)):
        batch_data_2_.append( [w for w in batch_data[i].split(sep) if w in word2vec_2] )

    lens_2 = list()
    for i in range(len(batch_data_2_)):
        lens_2.append( len(batch_data_2_[i]) )
    max_len_2 = numpy.max(lens_2)
    n_batches_2 = len(batch_data_2_)

    x_2 = numpy.zeros((n_batches_2, max_len_2, word_dim), dtype='float32')
    x_mask_2 = numpy.zeros((n_batches_2, max_len_2), dtype='int32')
    y_2 = numpy.zeros((n_batches_2, max_len_2, word_dim), dtype='float32')
    y_mask_2 = numpy.zeros((n_batches_2, max_len_2), dtype='int32')
    y_target_2 = numpy.zeros((n_batches_2, max_len_2), dtype='int32')

    batch_data_3_ = list()
    for i in range(len(batch_data)):
        batch_data_3_.append( [w for w in batch_data[i].split(sep) if w in word2vec_3] )

    lens_3 = list()
    for i in range(len(batch_data_3_)):
        lens_3.append( len(batch_data_3_[i]) )
    max_len_3 = numpy.max(lens_3)
    n_batches_3 = len(batch_data_3_)

    x_3 = numpy.zeros((n_batches_3, max_len_3, word_dim), dtype='float32')
    x_mask_3 = numpy.zeros((n_batches_3, max_len_3), dtype='int32')
    y_3 = numpy.zeros((n_batches_3, max_len_3, word_dim), dtype='float32')
    y_mask_3 = numpy.zeros((n_batches_3, max_len_3), dtype='int32')
    y_target_3 = numpy.zeros((n_batches_3, max_len_3), dtype='int32')

    for i in range(n_batches):
        x_mask[i, :lens[i]] = 1
        y_mask[i, :lens[i]] = 1
        for j in range(lens[i]):
            x[i, j, :] = word2vec[batch_data_[i][j]]
            y_target[i, j] = word2idx[batch_data_[i][j]]

        if '<s>' in word2vec:
            y[i, 0, :] = word2vec['<s>']
        for j in range(lens[i]-1):
            y[i, j+1, :] = word2vec[batch_data_[i][j]]
    print('x', x)
    print('x.shape', x.shape)
    print('y', y)
    print('y.shape', y.shape)
    print('y_target', y_target)
    print('y_target.shape', y_target.shape)

    for i in range(n_batches_2):
        x_mask_2[i, :lens_2[i]] = 1
        y_mask_2[i, :lens_2[i]] = 1
        for j in range(lens_2[i]):
            x_2[i, j, :] = word2vec_2[batch_data_2_[i][j]]
            y_target_2[i, j] = word2idx[batch_data_2_[i][j]]

        if '<s>' in word2vec_2:
            y_2[i, 0, :] = word2vec_2['<s>']
        for j in range(lens_2[i]-1):
            y_2[i, j+1, :] = word2vec_2[batch_data_2_[i][j]]
    print('x_2', x_2)
    print('x_2.shape', x_2.shape)
    print('y_2', y_2)
    print('y_2.shape', y_2.shape)
    print('y_target_2', y_target_2)
    print('y_target_2.shape', y_target_2.shape)


    for i in range(n_batches_3):
        x_mask_3[i, :lens_3[i]] = 1
        y_mask_3[i, :lens_3[i]] = 1
        for j in range(lens_3[i]):
            x_3[i, j, :] = word2vec_3[batch_data_3_[i][j]]
            y_target_3[i, j] = word2idx[batch_data_3_[i][j]]

        if '<s>' in word2vec_3:
            y_3[i, 0, :] = word2vec_3['<s>']
        for j in range(lens_3[i]-1):
            y_3[i, j+1, :] = word2vec_3[batch_data_3_[i][j]]
    print('x_3', x_3)
    print('x_3.shape', x_3.shape)
    print('y_3', y_3)
    print('y_3.shape', y_3.shape)
    print('y_target_3', y_target_3)
    print('y_target_3.shape', y_target_3.shape)

    return {'x':x, 'x_mask':x_mask, 'y':y, 'y_mask':y_mask, 'y_target':y_target,'x_2':x_2, 'x_mask_2':x_mask_2, 'y_2':y_2, 'y_mask_2':y_mask_2, 'y_target_2':y_target_2
            ,'x_3':x_3, 'x_mask_3':x_mask_3, 'y_3':y_3, 'y_mask_3':y_mask_3, 'y_target_3':y_target_3}


