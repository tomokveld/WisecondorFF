import argparse
import logging
import os
import sys
import warnings
import pysam
import re
import signal
import operator
import functools
import copy
import random


from pathlib import Path
from time import time
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger(__name__)

def profile(f):
    def wrap(*args, **kwargs):
        fname = f.__name__
        argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
        filled_args = ', '.join('%s=%r' % entry for entry in list(zip(argnames, args[:len(argnames)])) + [("args", list(args[len(argnames):]))] + [("kwargs", kwargs)])
        logger.info(f"Started: {fname}({filled_args})")
        starting_time = time()
        output = f(*args, **kwargs)
        logger.info(f"Ended: {fname}, duration {time() - starting_time}s")
        return output
    return wrap


def signal_kill(signal, frame):
    print("Killed!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_kill)


def read_pair_gen(bam_pointer, stats_map, region_str=None):
    """
    Return read pairs as they are encountered, includes the position of the first previously aligned reads
    """

    read_dict = defaultdict(lambda: [None, None])
    lr_pos = -1

    for read in bam_pointer.fetch(region=region_str):
        stats_map["reads_seen"] += 1

        if not read.is_proper_pair:
            stats_map["reads_pairf"] += 1
            continue

        if read.is_secondary or read.is_supplementary:
            stats_map["reads_secondary"] += 1
            continue

        query_name = read.query_name
        if query_name not in read_dict:
            if read.is_read1:
                read_dict[query_name][0] = (read, lr_pos)
            else:
                read_dict[query_name][1] = (read, lr_pos)
        else:
            if read.is_read1:
                yield (read, lr_pos), read_dict[query_name][1]
            else:
                yield read_dict[query_name][0], (read, lr_pos)
            del read_dict[query_name]
        lr_pos = read.pos


def convert_bam(bam_path, binsize=5000, min_mapq=1):
    stats_map = defaultdict(int)
    chr_re = re.compile("chr", re.IGNORECASE)

    rc_chr_map = {}
    for chrom in map(str, range(1, 23)):
        rc_chr_map[chrom] = None
    fs_chr_map = copy.deepcopy(rc_chr_map)

    def get_midpoint(r1, r2):
        return ((r1.reference_start + r2.reference_end) // 2), r1.template_length

    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        try:
            bam_file.check_index()
        except ValueError:
            logger.info("No BAM index found! Trying to generate...")
            pysam.index(bam_path)
            exit(2)
        except AttributeError:
            logger.info("File is SAM formatted and thus has no index.")
            exit(3)

        for index, chrom in enumerate(bam_file.references):
            chrom_name = chr_re.sub("", chrom)
            if chrom_name not in rc_chr_map:
                continue

            chr_size = bam_file.lengths[index]
            n_bins = (int(chr_size / float(binsize))) + 1

            logging.info(f"Processing: {chrom}, filling: {n_bins} bins")

            fs_dict = [defaultdict(int) for __ in range(n_bins)]
            rc_counts = np.zeros(int(bam_file.lengths[index] / float(binsize) + 1), dtype=np.int32)

            for ((read1, read1_prevpos), (read2, read2_prevpos)) in read_pair_gen(bam_file, stats_map, chrom):
                if read1.pos == read1_prevpos or read2.pos == read2_prevpos:
                    stats_map["reads_rmdup"] += 2
                    continue

                if read1.mapping_quality < min_mapq or read2.mapping_quality < min_mapq:
                    stats_map["reads_mapq"] += 2
                    continue

                stats_map["reads_kept"] += 2

                rc_counts[int(read1.pos / binsize)] += 1
                rc_counts[int(read2.pos / binsize)] += 1

                if read1.template_length > 0:
                    mid, insert_size = get_midpoint(read1, read2)
                else:
                    mid, insert_size = get_midpoint(read2, read1)
                fs_dict[int(mid // binsize)][insert_size] += 1

            rc_chr_map[chrom_name] = rc_counts
            fs_chr_map[chrom_name] = fs_dict

        qual_info = {
            "mapped": bam_file.mapped,
            "unmapped": bam_file.unmapped,
            "no_coordinate": bam_file.nocoordinate,
            "filter_rmdup": stats_map["reads_rmdup"],
            "filter_mapq": stats_map["reads_mapq"],
            "pre_retro": stats_map["reads_seen"],
            "post_retro": stats_map["reads_kept"],
            "pair_fail": stats_map["reads_pairf"],
        }

    return {"RC" : rc_chr_map, "FS" : fs_chr_map}, qual_info


@profile
def wcr_convert(args):
    sample, qual_info = convert_bam(args.infile, args.binsize, args.map_quality)
    np.savez_compressed(args.outfile,
                        quality=qual_info,
                        args={"binsize" : args.binsize,
                              "map_quality" : args.map_quality,
                              "infile" : args.infile,
                              "outfile" : args.outfile},
                        sample=sample)

########

def scale_sample(sample, from_size, to_size, scaling_function=None):
    if scaling_function is None:
        logging.critical(f"No scaling function given!")

    if to_size is None or from_size == to_size:
        return sample

    if to_size == 0 or from_size == 0 or to_size < from_size or to_size % from_size > 0:
        logging.critical(
            f"Impossible binsize scaling requested: {from_size:,} to {to_size:,}!"
        )
        # sys.exit()?
        return False

    scaled_sample = dict()
    scale = to_size // from_size

    logging.info(f"Scaling up by a factor of {scale} from {from_size:,} to {to_size:,}.")

    for chrom, data in sample.items():
        new_len = int(np.ceil(len(data) / float(scale)))
        scaled_chrom = []
        for i in range(new_len):
            scaled_chrom.append(scaling_function(data[i * scale : (i * scale) + scale]))
        scaled_sample[chrom] = np.array(scaled_chrom)

    return scaled_sample


def scale_sample_array(sample, from_size, to_size):
    return scale_sample(sample, from_size, to_size, np.sum)


def merge_counters(counters):
    merged = Counter()
    for c in counters:
        merged.update(c)
    return merged


def merge_dicts(dicts, defaultdict=defaultdict, int=int):
    merged = defaultdict(int)
    for d in dicts:
        for k in d:
            merged[k] += d[k]
    return merged

# This also works for a list of dicts
def scale_sample_counter(sample, from_size, to_size):
    return scale_sample(sample, from_size, to_size, merge_counters)


def scale_sample_dict(sample, from_size, to_size):
    return scale_sample(sample, from_size, to_size, merge_dicts)


def clip_counter(counter, lo, hi):
    if lo == hi:
        return counter
    return Counter(dict(filter(lambda x: x[0] > lo and x[0] <= hi, counter.items())))


def clip_sample(sample, clip_lo=0, clip_hi=300):
    for key in sample.keys():
        for i, counter in enumerate(sample[key]):
            sample[key][i] = clip_counter(counter, clip_lo, clip_hi)


def get_chromosome_freq(sample):
    sample_freq = {}
    for k, v in sample.items():
        chrom_array = []
        for binv in v:
            chrom_array.append(sum(binv.values()))
        sample_freq[k] = np.array(chrom_array)
    return sample_freq


def join_chromosomes(sample):
    return np.concatenate(list(sample.values()))


def freq_mask_sample(sample, min_observations):
    n = 0
    for key in sample.keys():
        for i, counter in enumerate(sample[key]):
            n_observations = sum(counter.values())
            if n_observations >= min_observations:
                continue
            sample[key][i] = Counter()
            n += 1
    logging.info(f"Removed {n} bins that have < {min_observations} observations")


def norm_freq_mask_sample(sample, cutoff=0.0001, min_cutoff=500):
    sample_freq = get_chromosome_freq(sample)
    sample_freq_full = join_chromosomes(sample_freq)
    cutoff = int(cutoff * sample_freq_full.sum())
    if min_cutoff:
        cutoff = max(cutoff, min_cutoff)
    freq_mask_sample(sample, cutoff)


def freq_to_mean(sample):
    for key in sample.keys():
        new_values = []
        for counter in sample[key]:
            mean = 0.0
            if counter:
                mean = sum(key * count for key, count in counter.items()) / sum(counter.values())
            new_values.append(mean)
        sample[key] = np.array(new_values)


def freq_to_median(sample):
    for key in sample.keys():
        new_values = []
        for counter in sample[key]:
            median = 0.0
            if counter:
                val = np.array(list(counter.keys()))
                freq = np.array(list(counter.values()))
                ordr = np.argsort(val)
                cdf = np.cumsum(freq[ordr])
                median = val[ordr][np.searchsorted(cdf, cdf[-1] // 2)]
            new_values.append(median)
        sample[key] = np.array(new_values)


def convert(text):
    return int(text) if text.isdigit() else text.lower()

def natural_sort(xs):
    return sorted(xs, key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])

def get_mask(samples, rc=False):
    by_chr = []
    bins_per_chr = []
    sample_count = len(samples)

    for chrom in natural_sort(samples[0].keys()):
        max_len = max([sample[chrom].shape[0] for sample in samples])
        this_chr = np.zeros((max_len, sample_count), dtype=float)
        bins_per_chr.append(max_len)

        for i, sample in enumerate(samples):
            this_chr[:, i] = sample[chrom]
        by_chr.append(this_chr)

    all_data = np.concatenate(by_chr, axis=0)

    if (rc):
        sum_per_sample = np.sum(all_data, 0)
        all_data = all_data / sum_per_sample

    sum_per_bin = np.sum(all_data, 1)
    mask = sum_per_bin > 0

    return mask, bins_per_chr


def train_pca(ref_data, pcacomp=5):
    t_data = ref_data.T
    pca = PCA(n_components=pcacomp)
    pca.fit(t_data)
    PCA(copy=True, whiten=False)
    transformed = pca.transform(t_data)
    inversed = pca.inverse_transform(transformed)
    corrected = t_data / inversed
    return corrected.T, pca


def normalize_and_mask(samples, chrs, mask, rc=False):
    by_chr = []
    sample_count = len(samples)

    for chrom in chrs:
        max_len = max([sample[str(chrom)].shape[0] for sample in samples])
        this_chr = np.zeros((max_len, sample_count), dtype=float)
        for i, sample in enumerate(samples):
            this_chr[:, i] = sample[str(chrom)]
        by_chr.append(this_chr)

    all_data = np.concatenate(by_chr, axis=0)

    if rc:
        sum_per_sample = np.sum(all_data, 0)
        all_data = all_data / sum_per_sample

    masked_data = all_data[mask, :]

    return masked_data

def reference_prep(binsize, samples, mask, bins_per_chr, rc=False):
    bins_per_chr = bins_per_chr[:22]
    mask = mask[:np.sum(bins_per_chr)]

    masked_data = normalize_and_mask(samples, range(1, 23), mask, rc)
    pca_corrected_data, pca = train_pca(masked_data)

    masked_bins_per_chr = [sum(mask[sum(bins_per_chr[:i]):sum(bins_per_chr[:i]) + x]) for i, x in enumerate(bins_per_chr)]
    masked_bins_per_chr_cum = [sum(masked_bins_per_chr[:x + 1]) for x in range(len(masked_bins_per_chr))]

    return {'binsize' : binsize,
            'mask' : mask,
            'masked_data' : masked_data,
            'bins_per_chr' : bins_per_chr,
            'masked_bins_per_chr' : masked_bins_per_chr,
            'masked_bins_per_chr_cum' : masked_bins_per_chr_cum,
            'pca_corrected_data' : pca_corrected_data,
            'pca_components' : pca.components_,
            'pca_mean' : pca.mean_}

def get_reference(pca_corrected_data, masked_bins_per_chr, masked_bins_per_chr_cum, ref_size):
    big_indexes = []
    big_distances = []

    regions = split_by_chr(0, masked_bins_per_chr_cum[-1], masked_bins_per_chr_cum)
    for (chrom, start, end) in regions:
        chr_data = np.concatenate((pca_corrected_data[:masked_bins_per_chr_cum[chrom] - masked_bins_per_chr[chrom], :], pca_corrected_data[masked_bins_per_chr_cum[chrom]:, :]))
        part_indexes, part_distances = get_ref_for_bins(ref_size, start, end, pca_corrected_data, chr_data)
        big_indexes.extend(part_indexes)
        big_distances.extend(part_distances)

    index_array = np.array(big_indexes)
    distance_array = np.array(big_distances)
    null_ratio_array = np.zeros((len(distance_array), min(len(pca_corrected_data[0]), 100)))  # TODO: make parameter
    samples = np.transpose(pca_corrected_data)

    for null_i, case_i in enumerate(random.sample(range(len(pca_corrected_data[0])), min(len(pca_corrected_data[0]), 100))):
        sample = samples[case_i]
        for bin_i in list(range(len(sample))):
            r = np.log2(sample[bin_i] / np.median(sample[index_array[bin_i]]))
            null_ratio_array[bin_i][null_i] = r

    return index_array, distance_array, null_ratio_array


def get_ref_for_bins(ref_size, start, end, pca_corrected_data, chr_data):
    ref_indexes = np.zeros((end - start, ref_size), dtype=np.int32)
    ref_distances = np.ones((end - start, ref_size))
    for cur_bin in range(start, end):
        bin_distances = np.sum(np.power(chr_data - pca_corrected_data[cur_bin, :], 2), 1)

        unsrt_ranked_idx = np.argpartition(bin_distances, ref_size)[:ref_size]
        ranked_idx = unsrt_ranked_idx[np.argsort(bin_distances[unsrt_ranked_idx])]
        ranked_distances = bin_distances[ranked_idx]

        ref_indexes[cur_bin - start, :] = ranked_idx
        ref_distances[cur_bin - start, :] = ranked_distances

    return ref_indexes, ref_distances

def split_by_chr(start, end, chr_bin_sums):
    areas = []
    tmp = [0, start, 0]
    for i, val in enumerate(chr_bin_sums):
        tmp[0] = i
        if val >= end:
            break
        if start < val < end:
            tmp[2] = val
            areas.append(tmp)
            tmp = [i, val, 0]
        tmp[1] = val
    tmp[2] = end
    areas.append(tmp)
    return areas

def reference_construct(ref_dict, ref_size=300):
    indexes, distances, null_ratios = get_reference(ref_dict['pca_corrected_data'],
                                                    ref_dict['masked_bins_per_chr'],
                                                    ref_dict['masked_bins_per_chr_cum'],
                                                    ref_size=ref_size)
    ref_dict['indexes'] = indexes
    ref_dict['distances'] = distances
    ref_dict['null_ratios'] = null_ratios


@profile
def wcr_reference(args):
    split_path = list(os.path.split(args.outref))
    if split_path[-1][-4:] == '.npz':
        split_path[-1] = split_path[-1][:-4]
    base_path = os.path.join(split_path[0], split_path[1])

    args.basepath = base_path
    args.prepfile = '{}_prep.npz'.format(base_path)
    args.partfile = '{}_part'.format(base_path)

    rc_samples = []
    fs_samples = []

    for npz in args.in_npzs:
        logging.info(f"Loading: {npz}")
        npzdata = np.load(npz, encoding='latin1', allow_pickle=True)
        sample = npzdata['sample'].item()
        sample_args = npzdata['args'].item()
        sample_binsize = int(sample_args['binsize'])

        fs_sample = scale_sample_counter(sample['FS'], sample_binsize, args.binsize)
        clip_sample(fs_sample, clip_lo=0, clip_hi=300) # TODO: add params
        norm_freq_mask_sample(fs_sample, cutoff=0.0001, min_cutoff=500) # TODO: add params
        freq_to_mean(fs_sample)

        fs_samples.append(fs_sample)
        rc_samples.append(scale_sample_array(sample['RC'], sample_binsize, args.binsize))

    fs_samples = np.array(fs_samples)
    rc_samples = np.array(rc_samples)

    fs_total_mask, fs_bins_per_chr = get_mask(fs_samples, rc=False)
    rc_total_mask, rc_bins_per_chr = get_mask(rc_samples, rc=True)

    fs_auto = reference_prep(args.refsize, fs_samples, fs_total_mask, fs_bins_per_chr, rc=False)
    rc_auto = reference_prep(args.refsize, rc_samples, rc_total_mask, rc_bins_per_chr, rc=True)

    reference_construct(fs_auto, args.refsize)
    reference_construct(rc_auto, args.refsize)

    np.savez_compressed(args.outref, reference = {"RC" : rc_auto, "FS" : fs_auto})




def make_generic(cvalue, func_xs, comp_xs, typecast, message):
    """Function generator that can handle most of the different types needed by argparse

    Args:
        cvalue (T): Optional value to compare to with the argument of the inner function
        func_xs (List[func(x: T) -> bool]): Optional list value of functions that are applied onto the argument of the inner function
        comp_xs (List[func(x: T, y: T) -> bool] or List[func(x: T) -> bool]): List value of functions (two-argument if cvalue is set, otherwise only applied on the argument of the inner function)
        typecast (T): Typecast argument to cast the argument of the inner function
        message (str): fstring used in error messages should follow the format of {tvalue} {cvalue}
    Returns:
        f(x: T) -> x: T
    """

    def check_generic(value):
        if func_xs:
            tvalue = functools.reduce(lambda res, f: f(res), func_xs, typecast(value))
        else:
            tvalue = typecast(value)

        if cvalue is None:
            if functools.reduce(lambda res, f: f(res), comp_xs, tvalue):
                raise argparse.ArgumentTypeError(message.format(tvalue, cvalue))
        else:
            if any([func(tvalue, cvalue) for func in comp_xs]):
                raise argparse.ArgumentTypeError(message.format(tvalue, cvalue))

        return typecast(value)

    return check_generic



REGION_MIN = 5000
REGION_MAX = 20000000

QUAL_MIN = 0
QUAL_MAX = 30

check_bins = make_generic(
    REGION_MAX,
    None,
    [operator.ge, lambda *x: x[0] < REGION_MIN],
    int,
    '"{}" invalid region size set, should be between 5000 bp to {} MB',
)

check_qual = make_generic(
    QUAL_MAX,
    None,
    [operator.ge, lambda *x: x[0] < QUAL_MIN],
    int,
    '"{}" invalid mapping quality set, should be between 0 bp to 30',
)


@profile
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='[%(levelname)s - %(asctime)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='wisecondorFF')
    subparsers = parser.add_subparsers(dest="subcommand")

    parser_logger = argparse.ArgumentParser(add_help=False)

    parser_logger.add_argument(
        "-l",
        "--log",
        dest="log_level",
        help="Set the logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser_convert = subparsers.add_parser("convert",
                                           help="Read and process a BAM file.",
                                           parents=[parser_logger],
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_convert.add_argument("-i",
                                "--in_file",
                                dest="infile",
                                help="Input BAM file for conversion.",
                                type=str,
                                required=True)
    parser_convert.add_argument("-o",
                                "--out_file",
                                dest="outfile",
                                help="Output .npz file.",
                                type=str,
                                required=True)
    parser_convert.add_argument("-b",
                                "--binsize",
                                dest="binsize",
                                help="Bin size (bp).",
                                default=5e4,
                                type=check_bins,
                                required=False)
    parser_convert.add_argument("-q",
                                "--quality",
                                dest="map_quality",
                                help="Mapping quality.",
                                default=1,
                                type=check_qual,
                                required=False)
    parser_convert.set_defaults(func=wcr_convert)

    ####

    parser_reference = subparsers.add_parser("reference",
                                          help="Create a reference set using healthy controls",
                                          parents=[parser_logger],
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_reference.add_argument("-i",
                                  "--in_npz",
                                  dest="in_npzs",
                                  nargs="+",
                                  help="Path to all NPZ files",
                                  type=str,
                                  required=True)
    parser_reference.add_argument("-o",
                                  "-out_ref",
                                  dest="outref",
                                  help="Path and filename for the reference output (e.g. path/to/myref.npz)",
                                  type=str,
                                  required=True)
    parser_reference.add_argument("-r"
                                  "--refsize",
                                  dest="refsize",
                                  help="Amount of reference regions per region",
                                  default=300,
                                  type=int,
                                  required=False)
    parser_reference.add_argument("-b"
                                  "--binsize",
                                  dest="binsize",
                                  help="Scale samples to this region size, multiples of existing region size only",
                                  default=500000,
                                  type=int,
                                  required=False)
    parser_reference.set_defaults(func=wcr_reference)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    args.func(args)

if __name__ == '__main__':
    main()
