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








    import numpy as np
import pandas as pd
import cooltools
import cooler
import logging
import functools
logger = logging.getLogger('one')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

    
def _bins_cis_total_ratio_filter(clr, thres):
    """
    Filter out bins with low cis-to-total coverage ratio from a Cooler object.

    Parameters
    ----------
    clr : cooler.Cooler
        A Cooler object containing Hi-C contact matrices.
    thres : float
        The threshold cis-to-total coverage ratio below which bins are considered bad.

    Returns
    -------
    numpy.ndarray
        An array of bin mask.
    """
    coverage = cooltools.coverage(clr)
    cis_total_cov = coverage[0] / coverage[1]
    bins_mask = cis_total_cov <= thres

    return bins_mask

def _write_filtered_pixels_files(bin_table, chunk_pixels, bin_mask, count_output_path):
    """
    Write the counts of interactions between genomic regions to a text file.

    This function takes a cooler object containing the genomic regions, a DataFrame containing the counts of interactions
    between genomic regions, and the path to the output text file. It then writes the counts of interactions to the output
    text file.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler object containing the genomic regions.
    chunk_pixels : pandas.DataFrame
        A DataFrame containing the counts of interactions between genomic regions.
    count_output_path : str
        The path to the output text file.

    Returns
    -------
    None

    """

    ### directly write into hdf5 file and use h5.copy to get other groups ###

    bad_bins_index = np.array(range(bin_table.shape[0]))[bin_mask]
    pixels_mask = chunk_pixels['bin1_id'].isin(bad_bins_index)
    chunk_pixels.loc[pixels_mask, 'count'] = 0
    pixels_mask = chunk_pixels['bin2_id'].isin(bad_bins_index)
    chunk_pixels.loc[pixels_mask, 'count'] = 0
    # Then write the counts in text file:
    with open(count_output_path, 'a+') as count_file:
        for i, row in chunk_pixels.iterrows():
            bin1_id, bin2_id, count = row
            chrom1, start1, end1 = list(bin_table.values[bin1_id])[:3]
            chrom2, start2, end2 = list(bin_table.values[bin2_id])[:3]

            count_file.write(f"{chrom1}\t{start1}\t{end1}\t{chrom2}\t{start2}\t{end2}\t{count}\n")



def filter_pixels_by_masked_bin(clr, thres, output_path, chunksize=10_000_000):
    """
    Filter the pixels of a cooler object based on a binary mask of masked bins.

    Parameters
    ----------
    clr : cooler.Cooler
        The cooler object to filter.
    thres : float
        The threshold for the cis-total ratio filter.
    chromsize_output_path : str
        The path to the output file where the chromosome sizes will be written.
    count_output_path : str
        The path to the output file where the filtered pixels will be written.
    chunksize : int, optional
        The size of the chunks to process the pixels in, in number of pixels.

    Returns
    -------
    None
    
    """

    logger.debug('Start to make bin mask...')
    bin_mask = _bins_cis_total_ratio_filter(clr, thres)
    bins_table = clr.bins()[:]
    tot_pixels = clr.pixels().shape[0]


    # First write the chromosome sizes:
    logger.debug('Start to create chromsize file...')
    with open(chromsize_output_path, 'a+') as chromsize_file:
        for i, chromsize in enumerate(clr.chromsizes):
            chromsize_file.write(f"{clr.chromnames[i]}\t{chromsize}\n")


    iter_num = tot_pixels // chunksize
    for i in range(iter_num):
        logger.debug(f'Start to process pixels chunk {i}...')
        chunk_pixels = clr.pixels()[chunksize * i : chunksize * (i + 1)]
        
        ### Here we might use multiprocessing to boost the speed, but does order matter###
        _write_filtered_pixels_files(bins_table, chunk_pixels, bin_mask, count_output_path)
    
    # remainder 
    chunk_pixels = clr.pixels()[chunksize * iter_num:]
    _write_filtered_pixels_files(bins_table, chunk_pixels, count_output_path)