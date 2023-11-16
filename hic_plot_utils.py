import os
import json
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' ### run on CPU


from cooltools.lib.numutils import set_diag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
import matplotlib.colors as mcolors

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append('/home1/yxiao977/labwork/train_akita/')
from masked_akita.basenji import dataset, dna_io, seqnn

class HiCPlot():
    def __init__(self, model_dir, data_dir) -> None:
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.seqnn_model = self.load_seqnn_model()
        self.hic_targets = pd.read_csv(data_dir+'/targets.txt',sep='\t')
        self.hic_file_dict_num = dict(zip(self.hic_targets['index'].values, self.hic_targets['file'].values))
        self.hic_file_dict = dict(zip(self.hic_targets['identifier'].values, self.hic_targets['file'].values))
        self.hic_num_to_name_dict = dict(zip(self.hic_targets['index'].values, self.hic_targets['identifier'].values))
        self.data_stats = self.read_data_parameters()
        self.seq_length = self.data_stats['seq_length']
        self.target_length = self.data_stats['target_length']
        self.hic_diags =  self.data_stats['diagonal_offset']
        self.target_crop = self.data_stats['crop_bp'] // self.data_stats['pool_width']
        self.target_length1 = self.data_stats['seq_length'] // self.data_stats['pool_width']
        self.target_length1_cropped = self.target_length1 - 2*self.target_crop
        self.test_inputs, self.test_targets, self.test_masks = self.get_test_data()
        try:
            self.test_sequences = self.get_sequences()
        except FileNotFoundError:
            pass

    def load_seqnn_model(self):
        params_file = self.model_dir+'params.json'
        model_file  = self.model_dir+'model_best.h5'
        with open(params_file) as params_open:
            params = json.load(params_open)
            params_model = params['model']
            params_train = params['train']

        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file)
        print('successfully loaded')

        return seqnn_model

    def read_data_parameters(self):
        data_stats_file = '%s/statistics.json' % self.data_dir
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        return data_stats

    def get_test_data(self):
        test_data = dataset.SeqDataset(self.data_dir, 'valid', batch_size=8)
        return test_data.numpy()
    
    def get_sequences(self, group='test'):
        sequences = pd.read_csv(self.data_dir+'sequences.bed', sep='\t', names=['chr','start','stop','type'])
        sequences = sequences.iloc[sequences['type'].values==group]
        sequences.reset_index(inplace=True, drop=True)
        return sequences

    ### for converting from flattened upper-triangluar vector to symmetric matrix  ###
    def from_upper_triu(self, vector_repr, matrix_len, num_diags):
        z = np.zeros((matrix_len,matrix_len))
        triu_tup = np.triu_indices(matrix_len,num_diags)
        z[triu_tup] = vector_repr
        for i in range(-num_diags+1,num_diags):
            set_diag(z, np.nan, i)
        return z + z.T

    def plot_pred_target(self, output_path, vmin=-0.7, vmax=0.7, target_range=(0, 2), print_seq=False):
        fig = make_subplots(rows=target_range[1] - target_range[0], cols=3, vertical_spacing=0.1, subplot_titles=['Pred', 'Interpolated Target', 'Original Target'])
        
        row_index = 0
        for test_index in range(target_range[0], target_range[1]):
            if print_seq:
                chrm, seq_start, seq_end = self.test_sequences.iloc[test_index][0:3]
                myseq_str = chrm+':'+str(seq_start)+'-'+str(seq_end)
                print(' ')
                print(myseq_str)

            test_target = self.test_targets[test_index:test_index+1,:,:]
            test_mask = self.test_masks[test_index:test_index+1,:,:]
            test_input = self.test_inputs[test_index:test_index+1,:,:]
            test_pred = self.seqnn_model.model.predict(test_input)

            target_index = 0
            hic_diag = 2
            colorscale = 'RdBu_r' #['#313695','#ffffff','#a50026']
    
            # plot pred
            mat = self.from_upper_triu(test_pred[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin), row=row_index+1, col=1)
        
            # plot interpolated target 
            mat = self.from_upper_triu(test_target[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin, showlegend=False), row=row_index+1, col=2)


            # plot target 
            ind = np.argwhere(test_mask == 0)
            original_target = test_target.copy()
            original_target[:, list(zip(*ind)), target_index] = np.nan
            mat = self.from_upper_triu(original_target[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin, showlegend=False), row=row_index+1, col=3)
            row_index += 1
            
        fig.update_layout(template='simple_white', font=dict(size=30), width=1500, height=1000)
        fig.update_xaxes(side="top")
        fig.update_yaxes(autorange="reversed")
        fig.update_annotations(yshift=40, font_size=30)
        fig.show()
        fig.write_image(output_path+'.pdf')
        fig.write_image(output_path+'.svg')

    def show_pred_target(self, num_pred=2, print_seq=False):
        for test_index in range(num_pred):
            fig = make_subplots(rows=1, cols=3, vertical_spacing=0.1, subplot_titles=['Pred', 'Interpolated Target', 'Original Target'])

            if print_seq:
                chrm, seq_start, seq_end = self.test_sequences.iloc[test_index][0:3]
                myseq_str = chrm+':'+str(seq_start)+'-'+str(seq_end)
                print(' ')
                print(myseq_str)

            test_target = self.test_targets[test_index:test_index+1,:,:]
            test_mask = self.test_masks[test_index:test_index+1,:,:]
            test_input = self.test_inputs[test_index:test_index+1,:,:]
            test_pred = self.seqnn_model.model.predict(test_input)

            target_index = 0
            vmin=-0.7; vmax=0.7
            hic_diag = 2
            colorscale = 'RdBu_r' #['#313695','#ffffff','#a50026']
    
            # plot pred
            mat = self.from_upper_triu(test_pred[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin), row=1, col=1)
        
            # plot interpolated target 
            mat = self.from_upper_triu(test_target[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin, showlegend=False), row=1, col=2)


            # plot target 
            ind = np.argwhere(test_mask == 0)
            original_target = test_target.copy()
            original_target[:, list(zip(*ind)), target_index] = np.nan
            mat = self.from_upper_triu(original_target[:,:,target_index], self.target_length1_cropped, hic_diag)
            fig.add_trace(go.Heatmap(z=mat, colorscale=colorscale, zmax=vmax, zmin=vmin, showlegend=False), row=1, col=3)
            fig.update_layout(template='simple_white', width=1500, height=500, font=dict(size=25))
            fig.update_xaxes(side="top")
            fig.update_yaxes(autorange="reversed")
            fig.update_annotations(yshift=40, font_size=25)
            fig.show()