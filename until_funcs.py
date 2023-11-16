def plot_pred_target(seqnn_model, test_targets, num_pred=2, seq_length=25000):
    fig = make_subplots(rows=1, cols=3, subplot_titles=['Pred', 'Interpolated Target', 'Original Target'])
    for test_index in range(num_pred):
        test_target = test_targets[test_index:test_index+1,:,:]
        test_mask = test_masks[test_index:test_index+1,:,:]
        test_input = test_inputs[test_index:test_index+1,:,:]
        test_pred = seqnn_model.model.predict(test_input)


        plt.figure(figsize=(8,4))
        target_index = 0
        vmin=-1; vmax=1

        hic_diag = 2
        
        # plot pred
        plt.subplot(131) 
        mat = from_upper_triu(test_pred[:,:,target_index], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap= 'RdBu_r', vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad = 0.05)
        plt.title('Pred')

        # plot interpolated target 
        plt.subplot(132) 
        mat = from_upper_triu(test_target[:,:,target_index], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap= 'RdBu_r', vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad = 0.05)
        plt.title('Interpolated Target')

        # plot target 
        plt.subplot(133)
        ind = np.argwhere(test_mask == 0)
        original_target = test_target.copy()
        original_target[:, list(zip(*ind)), target_index] = np.nan
        mat = from_upper_triu(original_target[:,:,target_index], target_length1_cropped, hic_diags)
        im = plt.matshow(mat, fignum=False, cmap= 'RdBu_r', vmax=vmax, vmin=vmin)
        plt.colorbar(im, fraction=.04, pad = 0.05)
        plt.title('Original Target')

        plt.tight_layout()
        plt.show()

def plot_pred_target(output_path, seqnn_model, test_targets, num_pred=2, seq_length=25000):
    fig = make_subplots(rows=num_pred, cols=3, vertical_spacing=0.1, subplot_titles=['Pred', 'Interpolated Target', 'Original Target'])
    for test_index in range(num_pred):
        test_target = test_targets[test_index:test_index+1,:,:]
        test_mask = test_masks[test_index:test_index+1,:,:]
        test_input = test_inputs[test_index:test_index+1,:,:]
        test_pred = seqnn_model.model.predict(test_input)

        target_index = 0
        vmin=0; vmax=1
        hic_diag = 2
 
        # plot pred
        mat = from_upper_triu(test_pred[:,:,target_index], target_length1_cropped, hic_diag)
        fig.add_trace(go.Heatmap(z=mat, colorscale='Viridis', zmax=vmax, zmin=vmin), row=test_index+1, col=1)
       
        # plot interpolated target 
        mat = from_upper_triu(test_target[:,:,target_index], target_length1_cropped, hic_diag)
        fig.add_trace(go.Heatmap(z=mat, colorscale='Viridis', zmax=vmax, zmin=vmin, showlegend=False), row=test_index+1, col=2)


        # plot target 
        ind = np.argwhere(test_mask == 0)
        original_target = test_target.copy()
        original_target[:, list(zip(*ind)), target_index] = np.nan
        mat = from_upper_triu(original_target[:,:,target_index], target_length1_cropped, hic_diag)
        fig.add_trace(go.Heatmap(z=mat, colorscale='Viridis', zmax=vmax, zmin=vmin, showlegend=False), row=test_index+1, col=3)
    fig.update_layout(template='simple_white', width=1500, height=1000, font=dict(size=25))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_annotations(yshift=40, font_size=25)
    fig.show()
    fig.write_image(output_path+'.pdf')
    fig.write_image(output_path+'.png')