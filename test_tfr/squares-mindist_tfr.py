#!/usr/bin/env python
from optparse import OptionParser
import os
from tqdm import tqdm

import numpy as np
from scipy.stats import beta, zscore
import tensorflow as tf

from basenji.dna_io import dna_1hot



'''
Name

Description...
'''

################################################################################
# main
########################################################################
def main():
    usage = 'usage: %prog [options] arg'
    ctcf_consensus= np.array(ctcf_consensus)
    ctcf_revcomp = np.array(ctcf_revcomp)
    motif_len = len(ctcf_consensus)
    spacer_len = 10

    import random
    def ranks(sample):
        """
        Return the ranks of each element in an integer sample.
        """
        indices = sorted(range(len(sample)), key=lambda i: sample[i])
        return sorted(indices, key=lambda i: indices[i])

    def sample_with_minimum_distance(n=40, k=4, d=10):
        """
        Sample of k elements from range(n), with a minimum distance d.
        """
        sample = random.sample(range(n-(k-1)*(d-1)), k)
        return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]


    from cooltools.lib.numutils import observed_over_expected

    # define options
    tf_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    for ti in tqdm(range(num_tfr)):
        tfr_file = '%s/gc-%d.tfr' % (out_dir, ti)

        with tf.python_io.TFRecordWriter(tfr_file, tf_opts) as writer:

            for si in range(seqs_per_tfr):

                num_boundaries = np.random.randint(4,8)
                boundary_positions = np.sort(  sample_with_minimum_distance(n=(seq_length-2*motif_len-spacer_len), k=num_boundaries, d= 100) )
                boundary_positions += motif_len+spacer_len//2
                boundary_positions = np.array( [0] + list(boundary_positions) + [seq_length])

                targetMatrix = np.zeros((seq_bins,seq_bins))
                for i in range(len(boundary_positions)-1):
                    s = boundary_positions[i] //bin_size
                    e = boundary_positions[i+1] //bin_size
                    targetMatrix[ s:e,s:e] = 1
                targetMatrix = np.log(observed_over_expected(targetMatrix+.1)[0])

                seq_dna = np.random.choice(['A','C','G','T'], size=seq_length, p= [.25,.25,.25,.25])
                for i in range(1,len(boundary_positions)-1):
                    seq_dna[boundary_positions[i]-motif_len - spacer_len//2:boundary_positions[i]- spacer_len//2 ]  = ctcf_consensus
                    seq_dna[boundary_positions[i] + spacer_len//2: boundary_positions[i] + motif_len + spacer_len//2 ]  =ctcf_revcomp


                # collapse list
                seq_dna = ''.join(seq_dna)

                # 1 hot code
                seq_1hot = dna_1hot(seq_dna)

                # compute targets
                seq_targets = targetMatrix.astype('float16')

                # make example
                example = tf.train.Example(features=tf.train.Features(feature={
                  'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                  'target': _bytes_feature(seq_targets[:,:].flatten().tostring())}))

                # write example
                writer.write(example.SerializeToString())

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
