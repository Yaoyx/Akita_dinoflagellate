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

def _pixels_filter(bins_table, chunk_pixels, bin_mask):
    """
    Filter out pixels that belong to bad bins.

    Parameters:
    -----------
    bin_table : numpy.ndarray
        A 2D array of shape (n_bins, n_features) containing the features of each bin.
    chunk_pixels : pandas.DataFrame
        A DataFrame containing the pixels to filter.
    bin_mask : numpy.ndarray
        A boolean array of shape (n_bins,) indicating which bins are bad.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing only the pixels that belong to good bins.
    """
    bad_bins_index = np.array(range(bins_table.shape[0]))[bin_mask]
    pixels_mask = chunk_pixels['bin1_id'].isin(bad_bins_index) + chunk_pixels['bin2_id'].isin(bad_bins_index)
    return chunk_pixels[pixels_mask]

def pixels_filter_generator(bins_table, bin_mask):
    """
    Returns a partial function that filters pixels based on the provided bins table and bin mask.

    Parameters:
    -----------
    bins_table : dict
        A dictionary containing the bins table.
    bin_mask : numpy.ndarray
        A numpy array containing the bin mask.

    Returns:
    --------
    partial function
        A partial function that filters pixels based on the provided bins table and bin mask.
    """

    return functools.partial(_pixels_filter, bins_table=bins_table, bin_mask=bin_mask)

class pixels_iterator:
    def __init__(self, clr_pixels_selector, pixels_size, chunksize):
        self.chunksize = chunksize
        self.max = pixels_size
        self.pixels = clr_pixels_selector

    def __iter__(self):
        self.pivot = 0
        return self

    def __next__(self):
        if (self.pivot + self.chunksize) < self.max:
            pivot = self.pivot
            self.pivot += self.chunksize
            return self.pixels[pivot : self.pivot]
        elif self.pivot < self.max:
            pivot = self.pivot
            self.pivot = self.max
            return self.pixels[pivot : self.pivot]
        else:
            raise StopIteration

def filter_pixels_by_masked_bin(clr, thres, output_path, chunksize=10_000_000):
    """
    Filter pixels of a cooler object based on a binary mask of genomic bins.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler object containing contact matrices and genomic bin information.
    thres : float
        A threshold value for filtering genomic bins based on their cis/trans ratio.
    output_path : str
        The path to the output cooler file.
    chunksize : int, optional
        The number of pixels to process at a time. Default is 10,000,000.

    Returns
    -------
    None

    Notes
    -----
    This function creates a binary mask of genomic bins based on their cis/trans ratio,
    and uses it to filter the pixels of the input cooler object. The resulting filtered
    pixels are written to a new cooler file at the specified output path.

    """

    logger.debug('Start to make bin mask...')
    bin_mask = _bins_cis_total_ratio_filter(clr, thres)
    bins_table = clr.bins()[:]
    tot_pixels = clr.pixels().shape[0]

    logger.debug('Start to create pixels counts file...')
    pixels_chunks = pixels_iterator(clr.pixels(), tot_pixels, chunksize)
    pixels_filter = pixels_filter_generator(bins_table, bin_mask)

    cooler.create_cooler(output_path, bins=bins_table, pixels=map(pixels_filter, pixels_chunks), ordered=True, columns=['count']) # input an iterator to pixels parameter

    