import numpy as np
import pandas as pd
import cooler
import logging

logger = logging.getLogger('one')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.disabled = True


def test_cis_total_filter(request):
    '''
    Test function for filtering out bad bins based on cis-total-ratio coverage.

    Parameters:
    -----------
    request : dict
        A dictionary containing the paths to the Hi-C file and the cis-total-ratio coverage file.

        Format:
        request = {
            'hic_path':'path',
            'cis_total_path':'path'
        }

    Returns:
    --------
    None

    Raises:
    -------
    AssertionError
        If the bad bins are not correctly filtered out.

    Notes:
    ------
    This function reads the Hi-C file and the cis-total-ratio coverage file, and filters out the bins with a cis-total-ratio coverage lower than or equal to 0.5. It then checks if the bad bins are correctly filtered out by comparing the sum of the counts of the pixels that involve bad bins with NaN. The function raises an AssertionError if the bad bins are not correctly filtered out.
    '''
    
    # Read data
    clr = cooler.Cooler(request['hic_path'])
    cis_total_cov = pd.read_table(request['cis_total_path'], sep='\t')
    cis_total_cov = cis_total_cov['cis-total-ratio'].to_numpy()

    # Find bad bins
    logger.debug('Start to find bad bins')
    bad_bins_index = np.argwhere((cis_total_cov <= 0.5)).reshape(-1)
    bad_bins = clr.bins()[:][clr.bins()[:].index.isin(bad_bins_index)]

    #Find good bins
    logger.debug('Start to find good bins')
    good_bins_index = np.argwhere((cis_total_cov > 0.5)).reshape(-1)
    good_bins = clr.bins()[:][clr.bins()[:].index.isin(good_bins_index)]


    # Check if bad bins are correctly filter out
    logger.debug('Start to create pixel dataframe based on binid1')
    bad_pixels_binid1 = [clr.pixels().fetch(bin.values[:3]) for index, bin in bad_bins.iterrows()]

    logger.debug('Start to create pixel dataframe based on binid2')
    good_pixels_dfs = [clr.pixels().fetch(bin.values[:3]) for index, bin in good_bins.iterrows()]
    bad_pixels_binid2 = [df[df.bin2_id.isin(bad_bins_index)] for df in good_pixels_dfs]
     
    # 
    logger.debug('Start to sum up counts')
    check_nan = lambda x: np.isnan(x['count']).all()

    check_result = np.sum(list(map(check_nan, bad_pixels_binid1)))
    assert check_result == len(bad_pixels_binid1)
    logger.info('Pass bin1 checkpoint')


    check_result = np.sum(list(map(check_nan, bad_pixels_binid2)))
    assert check_result == len(bad_pixels_binid2)
    logger.info('Pass bin2 checkpoint')


