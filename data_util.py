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

def _write_filtered_pixels_files(clr, chunk_pixels, bin_mask, count_output_path):
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
    bad_bins_index = np.array(range(clr.bins().shape[0]))[bin_mask]
    pixels_mask = chunk_pixels['bin1_id'].isin(bad_bins_index)
    chunk_pixels.loc[pixels_mask, 'count'] = 0
    pixels_mask = chunk_pixels['bin2_id'].isin(bad_bins_index)
    chunk_pixels.loc[pixels_mask, 'count'] = 0
    # Then write the counts in text file:
    with open(count_output_path, 'a+') as count_file:
        for i, row in chunk_pixels.iterrows():
            bin1_id, bin2_id, count = row
            chrom1, start1, end1 = list(clr.bins()[bin1_id])[:3]
            chrom2, start2, end2 = list(clr.bins()[bin2_id])[:3]

            count_file.write(f"{chrom1}\t{start1}\t{end1}\t{chrom2}\t{start2}\t{end2}\t{count}\n")

def filter_pixels_by_masked_bin(clr, thres, chromsize_output_path, count_output_path, chunksize=10_000_000):
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
    tot_pixels = clr.pixels().shape[0]
    iter_num = tot_pixels // chunksize

    # First write the chromosome sizes:
    logger.debug('Start to create chromsize file...')
    with open(chromsize_output_path, 'a+') as chromsize_file:
        for i, chromsize in enumerate(clr.chromsizes):
            chromsize_file.write(f"{clr.chromnames[i]}\t{chromsize}\n")

    logger.debug('Start to create pixels counts file...')
    for i in range(iter_num):
        chunk_pixels = clr.pixels()[chunksize * i : chunksize * (i + 1)]
        
        ### Here we might use multiprocessing to boost the speed, but does order matter###
        _write_filtered_pixels_files(clr, chunk_pixels, bin_mask, count_output_path)
    
    # remainder 
    chunk_pixels = clr.pixels()[chunksize * iter_num:]
    _write_filtered_pixels_files(clr, chunk_pixels, count_output_path)