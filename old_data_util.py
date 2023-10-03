import numpy as np
import pandas as pd
import cooltools
import cooler
import logging
logger = logging.getLogger('one')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

### Update: 1.make bin table mask, instead of filtering use mask 2. abstract the function 3. filtering in the balance_cooler 4. make threshold as an option 
### 5. use chunks instead of saving the whole pixel matrix 6. we want to have some fucntion that takes cooler and bins to create a filtered cooler file

### functions:
    ### 1. filter_cis_total(): cal coverage; filter bin mask 2. filter_pixels_by_bin_mask()

### Questions: 1. for chunk, how do we modify pixels data in chunks, directly modify cool file? 2. general cis-total-ratio column 3. for any size of bin table we should be able to generate a new cooler file

def bad_bins_filter(file_path, output_path, cis_total_cache_path=None, save_cis_total_cache=None):
    """
    Filters out bad bins from a Hi-C contact matrix and creates a new cooler file with only good bins.

    Parameters:
    -----------
    file_path : str
        The path to the input cooler file.
    output_path : str
        The path to the output cooler file.
    cis_total_cache_path : str, optional
        The path to the cis-total-ratio cache file. If provided, the function will read the cis-total-ratio from the cache file instead of generating it.
    save_cis_total_cache : str, optional
        The path to save the cis-total-ratio cache file. If provided, the function will save the cis-total-ratio to the cache file.

    Returns:
    --------
    None
    """

    clr = cooler.Cooler(file_path)

    # if cis_total_cache_path:
    #     logger.debug('Start to read cis-total-ratio...')

    #     cis_total_cov = pd.read_table(cis_total_cache_path, sep='\t')
    #     cis_total_cov = cis_total_cov['cis-total-ratio'].to_numpy()
    try:
        cis_total_cov = clr.bins()[:]['_cis'] / clr.bins()[:]['_tot']
    except KeyError:
        logger.debug('Start to generate cis-total-ratio...')

        coverage = cooltools.coverage(clr, clr_weight_name='weight', store=True) # is it better to set store to True by default
        cis_total_cov = coverage[0] / coverage[1]
        # if save_cis_total_cache:
        #     cis_total_cov_df = pd.DataFrame(cis_total_cov, columns=['cis-total-ratio'])
        #     cis_total_cov_df.to_csv(save_cis_total_cache, sep='\t', index=None)


    good_bins_index = np.argwhere((cis_total_cov > 0.5)).reshape(-1)
    good_bins = clr.bins()[:].iloc[good_bins_index]

    logger.debug('Start to find good pixels...')

    good_pixels_dfs = [clr.pixels().fetch(bin.values[:3]) for index, bin in good_bins.iterrows()] #iterate through chunks of pixel table
    good_pixels_dfs = [df[df.bin2_id.isin(good_bins_index)] for df in good_pixels_dfs]

    logger.debug('Start to generate cooler...')

    cooler.create_cooler(output_path, bins=clr.bins()[:][['chrom','start','end']], pixels=good_pixels_dfs, ordered=True, columns=['count']) # wihtout this create function
    ## write it into hdf5 directly instead of using the create_cooler