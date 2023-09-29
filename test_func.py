import numpy as np
import pandas as pd
import cooler

request = {'cis_total_path': '/home1/yxiao977/sc1/train_akita/data/5000res_bins_cistotal_ratio.bed'

}

def test_cis_total_filter(request):
    '''
    

    '''
    
    # Read data
    clr = cooler.Cooler(request['hic_path'])
    cis_total_cov = pd.read_table(request['cis_total_path'], sep='\t')
    cis_total_cov = cis_total_cov['cis-total-ratio'].to_numpy()

    # Find bad bins
    bad_bins_index = np.argwhere((cis_total_cov <= 0.5)).reshape(-1)
    bad_bins = clr.bins()[:][clr.bins()[:].index.isin(bad_bins_index)]

    #Find good bins
    good_bins_index = np.argwhere((cis_total_cov > 0.5)).reshape(-1)
    good_bins = clr.bins()[:][clr.bins()[:].index.isin(good_bins_index)]


    # Check if bad bins are correctly filter out
    bad_pixels_binid1 = [clr.pixels().fetch(bin.values[:3]) for index, bin in bad_bins.iterrows()]

    good_pixels_dfs = [clr.pixels().fetch(bin.values[:3]) for index, bin in good_bins.iterrows()]
    bad_pixels_binid2 = [df[df.bin2_id.isin(bad_bins_index)] for df in good_pixels_dfs]
     
    # Use np.nansum to see if the sum equals to nan for bad pixels
    counts_ls = [np.nansum(df['count']) for df in bad_pixels_binid1]
    assert np.isnan(np.nansum(counts_ls))

    counts_ls = [np.nansum(df['count']) for df in bad_pixels_binid2]
    assert np.isnan(np.nansum(counts_ls))


