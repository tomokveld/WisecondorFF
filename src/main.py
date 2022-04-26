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
import uuid
import copy
import random
import json
import subprocess

from pathlib import Path
from time import time
from sklearn.decomposition import PCA
from scipy.stats import norm
from collections import defaultdict, Counter

import numpy as np

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def profile(f):
    def wrap(*args, **kwargs):
        fname = f.__name__
        argnames = f.__code__.co_varnames[: f.__code__.co_argcount]
        filled_args = ", ".join(
            "%s=%r" % entry
            for entry in list(zip(argnames, args[: len(argnames)]))
            + [("args", list(args[len(argnames) :]))]
            + [("kwargs", kwargs)]
        )
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
            rc_counts = np.zeros(
                int(bam_file.lengths[index] / float(binsize) + 1), dtype=np.int32
            )

            for ((read1, read1_prevpos), (read2, read2_prevpos)) in read_pair_gen(
                bam_file, stats_map, chrom
            ):
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

    return {"RC": rc_chr_map, "FS": fs_chr_map}, qual_info


@profile
def wcr_convert(args):
    sample, qual_info = convert_bam(args.infile, args.binsize, args.map_quality)
    np.savez_compressed(
        args.outfile,
        quality=qual_info,
        args={
            "binsize": args.binsize,
            "map_quality": args.map_quality,
            "infile": args.infile,
            "outfile": args.outfile,
        },
        sample=sample,
    )


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

    logging.info(
        f"Scaling up by a factor of {scale} from {from_size:,} to {to_size:,}."
    )

    for chrom, data in sample.items():
        new_len = int(np.ceil(len(data) / float(scale)))
        scaled_chrom = []
        for i in range(new_len):
            scaled_chrom.append(scaling_function(data[int(i * scale) : int((i * scale) + scale)]))
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
                mean = sum(key * count for key, count in counter.items()) / sum(
                    counter.values()
                )
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
    return sorted(xs, key=lambda key: [convert(c) for c in re.split("([0-9]+)", key)])


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

    if rc:
        sum_per_sample = np.sum(all_data, 0)
        all_data = all_data / sum_per_sample

    sum_per_bin = np.sum(all_data, 1)
    mask = sum_per_bin > 0

    return mask, bins_per_chr


def train_pca(ref_data, pcacomp=1):
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


def reference_prep(binsize, refsize, samples, mask, bins_per_chr, rc=False):
    bins_per_chr = bins_per_chr[:22]
    mask = mask[: np.sum(bins_per_chr)]

    masked_data = normalize_and_mask(samples, range(1, 23), mask, rc)
    pca_corrected_data, pca = train_pca(masked_data)

    masked_bins_per_chr = [
        sum(mask[sum(bins_per_chr[:i]) : sum(bins_per_chr[:i]) + x])
        for i, x in enumerate(bins_per_chr)
    ]
    masked_bins_per_chr_cum = [
        sum(masked_bins_per_chr[: x + 1]) for x in range(len(masked_bins_per_chr))
    ]

    return {
        "binsize": binsize,
        "refsize": refsize,
        "mask": mask,
        "masked_data": masked_data,
        "bins_per_chr": bins_per_chr,
        "masked_bins_per_chr": masked_bins_per_chr,
        "masked_bins_per_chr_cum": masked_bins_per_chr_cum,
        "pca_corrected_data": pca_corrected_data,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
    }


def get_reference(
    pca_corrected_data, masked_bins_per_chr, masked_bins_per_chr_cum, ref_size
):
    big_indexes = []
    big_distances = []

    regions = split_by_chr(0, masked_bins_per_chr_cum[-1], masked_bins_per_chr_cum)
    for (chrom, start, end) in regions:
        chr_data = np.concatenate(
            (
                pca_corrected_data[
                    : masked_bins_per_chr_cum[chrom] - masked_bins_per_chr[chrom], :
                ],
                pca_corrected_data[masked_bins_per_chr_cum[chrom] :, :],
            )
        )

        part_indexes, part_distances = get_ref_for_bins(
            ref_size, start, end, pca_corrected_data, chr_data
        )

        big_indexes.extend(part_indexes)
        big_distances.extend(part_distances)

    index_array = np.array(big_indexes)
    distance_array = np.array(big_distances)
    null_ratio_array = np.zeros(
        (len(distance_array), min(len(pca_corrected_data[0]), 100))
    )  # TODO: make parameter
    samples = np.transpose(pca_corrected_data)

    for null_i, case_i in enumerate(
        random.sample(
            range(len(pca_corrected_data[0])), min(len(pca_corrected_data[0]), 100)
        )
    ):
        sample = samples[case_i]
        for bin_i in list(range(len(sample))):
            r = np.log2(sample[bin_i] / np.median(sample[index_array[bin_i]]))
            null_ratio_array[bin_i][null_i] = r

    return index_array, distance_array, null_ratio_array


def get_ref_for_bins(ref_size, start, end, pca_corrected_data, chr_data):
    ref_indexes = np.zeros((end - start, ref_size), dtype=np.int32)
    ref_distances = np.ones((end - start, ref_size))
    for cur_bin in range(start, end):
        bin_distances = np.sum(
            np.power(chr_data - pca_corrected_data[cur_bin, :], 2), 1
        )

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
    indexes, distances, null_ratios = get_reference(
        ref_dict["pca_corrected_data"],
        ref_dict["masked_bins_per_chr"],
        ref_dict["masked_bins_per_chr_cum"],
        ref_size=ref_size,
    )
    ref_dict["indexes"] = indexes
    ref_dict["distances"] = distances
    ref_dict["null_ratios"] = null_ratios


@profile
def wcr_reference(args):
    split_path = list(os.path.split(args.outref))
    if split_path[-1][-4:] == ".npz":
        split_path[-1] = split_path[-1][:-4]
    base_path = os.path.join(split_path[0], split_path[1])
    args.basepath = base_path

    rc_samples = []
    fs_samples = []

    for npz in args.in_npzs:
        logging.info(f"Loading: {npz}")
        npzdata = np.load(npz, encoding="latin1", allow_pickle=True)
        sample = npzdata["sample"].item()
        sample_args = npzdata["args"].item()
        sample_binsize = int(sample_args["binsize"])

        fs_sample = scale_sample_counter(sample["FS"], sample_binsize, args.binsize)
        clip_sample(fs_sample, clip_lo=args.fs_clip_low, clip_hi=args.fs_clip_high)
        norm_freq_mask_sample(fs_sample, cutoff=args.rc_clip_norm, min_cutoff=args.rc_clip_abs)
        freq_to_mean(fs_sample)

        fs_samples.append(fs_sample)
        rc_samples.append(scale_sample_array(sample["RC"], sample_binsize, args.binsize))

    fs_samples = np.array(fs_samples)
    rc_samples = np.array(rc_samples)

    fs_total_mask, fs_bins_per_chr = get_mask(fs_samples, rc=False)
    rc_total_mask, rc_bins_per_chr = get_mask(rc_samples, rc=True)

    fs_auto = reference_prep(
        args.binsize, args.refsize, fs_samples, fs_total_mask, fs_bins_per_chr, rc=False
    )
    rc_auto = reference_prep(
        args.binsize, args.refsize, rc_samples, rc_total_mask, rc_bins_per_chr, rc=True
    )

    reference_construct(fs_auto, args.refsize)
    reference_construct(rc_auto, args.refsize)

    np.savez_compressed(args.outref, reference={"RC": rc_auto, "FS": fs_auto})


######


def coverage_normalize_and_mask(sample, ref_file, rc=False):
    by_chr = []
    chromosomes = range(1, len(ref_file["bins_per_chr"]) + 1)

    for chrom in chromosomes:
        this_chr = np.zeros(ref_file["bins_per_chr"][chrom - 1], dtype=float)
        min_len = min(ref_file["bins_per_chr"][chrom - 1], len(sample[str(chrom)]))
        this_chr[:min_len] = sample[str(chrom)][:min_len]
        by_chr.append(this_chr)
    all_data = np.concatenate(by_chr, axis=0)

    if rc:
        all_data = all_data / np.sum(all_data)

    masked_data = all_data[ref_file["mask"]]

    return masked_data


def project_pc(sample_data, ref_file):
    pca = PCA(n_components=ref_file["pca_components"].shape[0])
    pca.components_ = ref_file["pca_components"]
    pca.mean_ = ref_file["pca_mean"]

    transform = pca.transform(np.array([sample_data]))

    reconstructed = np.dot(transform, pca.components_) + pca.mean_
    reconstructed = reconstructed[0]
    return sample_data / reconstructed


def get_weights(ref_file):
    inverse_weights = [np.mean(np.sqrt(x)) for x in ref_file["distances"]]
    weights = np.array([1 / x for x in inverse_weights])
    return weights


def get_optimal_cutoff(ref_file, repeats):
    distances = ref_file["distances"]
    cutoff = float("inf")
    for i in range(0, repeats):
        mask = distances < cutoff
        average = np.average(distances[mask])
        stddev = np.std(distances[mask])
        cutoff = average + 3 * stddev
    return cutoff


def normalize_repeat(test_data, ref_file, optimal_cutoff):
    test_copy = np.copy(test_data)
    for i in range(3): # TODO: Parameter
        results_z, results_r, ref_sizes = _normalize_once(
            test_data, test_copy, ref_file, optimal_cutoff
        )

        test_copy[np.abs(results_z) >= norm.ppf(0.99)] = -1
    m_lr = np.nanmedian(np.log2(results_r))
    m_z = np.nanmedian(results_z)

    return results_z, results_r, ref_sizes, m_lr, m_z


def _normalize_once(test_data, test_copy, ref_file, optimal_cutoff):
    masked_bins_per_chr = ref_file["masked_bins_per_chr"]
    masked_bins_per_chr_cum = ref_file["masked_bins_per_chr_cum"]
    results_z = np.zeros(masked_bins_per_chr_cum[-1])
    results_r = np.zeros(masked_bins_per_chr_cum[-1])
    ref_sizes = np.zeros(masked_bins_per_chr_cum[-1])
    indexes = ref_file["indexes"]
    distances = ref_file["distances"]

    i = 0
    i2 = 0
    for chrom in list(range(len(masked_bins_per_chr))):
        start = masked_bins_per_chr_cum[chrom] - masked_bins_per_chr[chrom]
        end = masked_bins_per_chr_cum[chrom]
        chr_data = np.concatenate(
            (
                test_copy[
                    : masked_bins_per_chr_cum[chrom] - masked_bins_per_chr[chrom]
                ],
                test_copy[masked_bins_per_chr_cum[chrom] :],
            )
        )
        for index in indexes[start:end]:
            ref_data = chr_data[index[distances[i] < optimal_cutoff]]
            ref_data = ref_data[ref_data >= 0]
            ref_stdev = np.std(ref_data)

            results_z[i2] = (test_data[i] - np.mean(ref_data)) / ref_stdev
            results_r[i2] = test_data[i] / np.median(ref_data)
            ref_sizes[i2] = ref_data.shape[0]
            i += 1
            i2 += 1

    return results_z, results_r, ref_sizes


def normalize(maskrepeats, sample, final_dict, rc=False):
    sample = coverage_normalize_and_mask(sample, final_dict, rc)
    sample = project_pc(sample, final_dict)
    results_w = get_weights(final_dict)
    optimal_cutoff = get_optimal_cutoff(final_dict, maskrepeats)
    results_z, results_r, ref_sizes, m_lr, m_z = normalize_repeat(
        sample, final_dict, optimal_cutoff
    )
    return results_r, results_z, results_w, ref_sizes, m_lr, m_z


def inflate_results(results, rem_input):
    temp = [0 for x in rem_input["mask"]]
    j = 0
    for i, val in enumerate(rem_input["mask"]):
        if val:
            temp[i] = results[j]
            j += 1
    return temp


def get_post_processed_result(minrefbins, result, ref_sizes, rem_input):
    infinite_mask = ref_sizes < minrefbins
    result[infinite_mask] = 0
    inflated_results = inflate_results(result, rem_input)

    final_results = []
    for chr in range(len(rem_input["bins_per_chr"])):
        chr_data = inflated_results[
            sum(rem_input["bins_per_chr"][:chr]) : sum(
                rem_input["bins_per_chr"][: chr + 1]
            )
        ]
        final_results.append(chr_data)

    return final_results


def log_trans(results, log_r_median):
    for chr in range(len(results["results_r"])):
        results["results_r"][chr] = np.log2(results["results_r"][chr])

    results["results_r"] = [x.tolist() for x in results["results_r"]]

    for c in range(len(results["results_r"])):
        for i, rR in enumerate(results["results_r"][c]):
            if not np.isfinite(rR):
                results["results_r"][c][i] = 0
                results["results_z"][c][i] = 0
                results["results_w"][c][i] = 0
            if results["results_r"][c][i] != 0:
                results["results_r"][c][i] = results["results_r"][c][i] - log_r_median


def _wcr_detect_wrap(sample, final_dict, rc=False, maskrepeats=5, minrefbins=150, zscore=5):
    results_r, results_z, results_w, ref_sizes, m_lr, m_z = normalize(
        maskrepeats, sample, final_dict, rc=rc
    )

    null_ratios_aut_per_bin = final_dict["null_ratios"]

    rem_input = {
        "binsize": int(final_dict["binsize"]),
        "zscore": zscore,
        "mask": final_dict["mask"],
        "bins_per_chr": np.array(final_dict["bins_per_chr"]),
        "masked_bins_per_chr": np.array(final_dict["masked_bins_per_chr"]),
        "masked_bins_per_chr_cum": np.array(final_dict["masked_bins_per_chr_cum"]),
    }

    results_z = results_z - m_z
    results_w = results_w / np.nanmean(results_w)

    if np.isnan(results_w).any() or np.isinf(results_w).any():
        logging.warning(
            "Non-numeric values found in weights -- reference too small. Circular binary segmentation and z-scoring will be unweighted"
        )
        results_w = np.ones(len(results_w))

    null_ratios = np.array([x.tolist() for x in null_ratios_aut_per_bin])

    results = {
        "results_r": results_r,
        "results_z": results_z,
        "results_w": results_w,
        "results_nr": null_ratios,
        "ref_sizes": ref_sizes,
    }

    for result in results.keys():
        results[result] = get_post_processed_result(
            minrefbins, results[result], ref_sizes, rem_input
        )

    log_trans(results, m_lr)

    return results, rem_input


def get_res_to_nparray(results):
    new_results = {}
    new_results["results_z"] = np.array([np.array(x) for x in results["results_z"]])
    new_results["results_w"] = np.array([np.array(x) for x in results["results_w"]])
    new_results["results_nr"] = np.array([np.array(x) for x in results["results_nr"]])
    new_results["results_r"] = np.array([np.array(x) for x in results["results_r"]])
    return new_results


def res_to_nestedlist(results):
    for k, v in results.items():
        results[k] = [list(i) for i in v]


def get_z_score(results_c, results):
    results_nr, results_r, results_w = (
        results["results_nr"],
        results["results_r"],
        results["results_w"],
    )
    zs = []
    for segment in results_c:
        segment_nr = results_nr[segment[0]][segment[1] : segment[2]]
        segment_rr = results_r[segment[0]][segment[1] : segment[2]]
        segment_nr = [
            segment_nr[i] for i in range(len(segment_nr)) if segment_rr[i] != 0
        ]
        segment_w = results_w[segment[0]][segment[1] : segment[2]]
        segment_w = [segment_w[i] for i in range(len(segment_w)) if segment_rr[i] != 0]
        null_segments = [
            np.ma.average(x, weights=segment_w) for x in np.transpose(segment_nr)
        ]
        null_mean = np.ma.mean([x for x in null_segments if np.isfinite(x)])
        null_sd = np.ma.std([x for x in null_segments if np.isfinite(x)])
        z = (segment[3] - null_mean) / null_sd
        z = min(z, 1000)
        z = max(z, -1000)
        zs.append(z)
    return zs


def _get_processed_cbs(cbs_data):
    results_c = []
    for i, segment in enumerate(cbs_data):
        chr = int(segment["chr"]) - 1
        s = int(segment["s"])
        e = int(segment["e"])
        r = segment["r"]
        results_c.append([chr, s, e, r])
    return results_c


def exec_R(json_dict):
    json.dump(json_dict, open(json_dict["infile"], "w"))

    r_cmd = ["Rscript", json_dict["R_script"], "--infile", json_dict["infile"]]
    logging.debug(f"CBS cmd: {r_cmd}")

    try:
        subprocess.check_call(r_cmd)
    except subprocess.CalledProcessError as e:
        logging.critical(f"Rscript failed: {e}")
        sys.exit()

    os.remove(json_dict["infile"])
    if "outfile" in json_dict.keys():
        json_out = json.load(open(json_dict["outfile"]))
        os.remove(json_dict["outfile"])
        return json_out


def exec_cbs(rem_input, results):
    json_cbs_dir = os.path.abspath('test_' + str(uuid.uuid4()) + '_CBS_tmp')

    json_dict = {
        "R_script": str(Path(os.path.dirname(os.path.realpath(__file__))).joinpath("include/CBS.R")),
        "ref_gender": "A",
        "alpha": str(1e-4),
        "binsize": str(rem_input["binsize"]),
        "results_r": results["results_r"],
        "results_w": results["results_w"],
        "infile": str(f"{json_cbs_dir}_01.json"),
        "outfile": str(f"{json_cbs_dir}_02.json"),
    }

    results_c = _get_processed_cbs(exec_R(json_dict))
    segment_z = get_z_score(results_c, results)
    results_c = [
        results_c[i][:3] + [segment_z[i]] + [results_c[i][3]]
        for i in range(len(results_c))
    ]
    return results_c


def generate_segments(rem_input, results):
    for segment in results["results_c"]:
        chr_name = str(segment[0] + 1)
        row = [
            chr_name,
            int(segment[1] * rem_input["binsize"] + 1),
            int(segment[2] * rem_input["binsize"]),
            segment[4],
            segment[3],
        ]
        if float(segment[3]) > rem_input["zscore"]:
            print("{}\tgain\n".format("\t".join([str(x) for x in row])))
        elif float(segment[3]) < -rem_input["zscore"]:
            print("{}\tloss\n".format("\t".join([str(x) for x in row])))


@profile
def wcr_detect(args):
    ref_npz = np.load(args.reference, encoding="latin1", allow_pickle=True)
    sample_npz = np.load(args.in_npz, encoding="latin1", allow_pickle=True)

    ref = ref_npz["reference"].item()

    sample_args = sample_npz["args"].item()
    sample_data = sample_npz["sample"].item()

    rc_sample = sample_data["RC"]
    fs_sample = sample_data["FS"]

    rc_sample = scale_sample_array(
        rc_sample, sample_args["binsize"], ref["RC"]["binsize"]
    )
    fs_sample = scale_sample_counter(
        fs_sample, sample_args["binsize"], ref["FS"]["binsize"]
    )

    clip_sample(fs_sample, clip_lo=args.fs_clip_low, clip_hi=args.fs_clip_high)
    norm_freq_mask_sample(fs_sample, cutoff=args.rc_clip_norm, min_cutoff=args.rc_clip_abs)
    freq_to_mean(fs_sample)

    rc_results, rc_rem_input = _wcr_detect_wrap(rc_sample, ref["RC"], rc=True, maskrepeats=args.maskrepeats, minrefbins=args.minrefbins, zscore=args.zscore)
    fs_results, fs_rem_input = _wcr_detect_wrap(fs_sample, ref["FS"], rc=False, maskrepeats=args.maskrepeats, minrefbins=args.minrefbins, zscore=args.zscore)

    _rc_results = get_res_to_nparray(rc_results)
    _fs_results = get_res_to_nparray(fs_results)

    comb_results = {}
    comb_results["results_z"] = (
        _rc_results["results_z"] + (-_fs_results["results_z"])
    ) / np.sqrt(2)
    comb_results["results_r"] = (
        _rc_results["results_r"] + (-_fs_results["results_r"])
    ) / np.sqrt(2)
    comb_results["results_w"] = (
        _rc_results["results_w"] + (_fs_results["results_w"])
    ) / 2
    comb_results["results_nr"] = (
        _rc_results["results_nr"] + (_fs_results["results_nr"])
    ) / 2

    res_to_nestedlist(comb_results)

    comb_results["results_c"] = exec_cbs(fs_rem_input, comb_results)

    generate_segments(fs_rem_input, comb_results)


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
    logging.basicConfig(
        format="[%(levelname)s - %(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="wisecondorFF")
    subparsers = parser.add_subparsers(dest="subcommand")

    parser_logger = argparse.ArgumentParser(add_help=False)

    parser_logger.add_argument(
        "-l",
        "--log",
        dest="log_level",
        help="Set the logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    parser_convert = subparsers.add_parser(
        "convert",
        help="Read and process a BAM file.",
        parents=[parser_logger],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_convert.add_argument(
        "-i",
        "--in_file",
        dest="infile",
        help="Input BAM file for conversion.",
        type=str,
        required=True,
    )
    parser_convert.add_argument(
        "-o",
        "--out_file",
        dest="outfile",
        help="Output .npz file.",
        type=str,
        required=True,
    )
    parser_convert.add_argument(
        "-b",
        "--binsize",
        dest="binsize",
        help="Bin size (bp).",
        default=5e4,
        type=check_bins,
        required=False,
    )
    parser_convert.add_argument(
        "-q",
        "--quality",
        dest="map_quality",
        help="Mapping quality.",
        default=1,
        type=check_qual,
        required=False,
    )
    parser_convert.set_defaults(func=wcr_convert)

    ####

    # TODO: Clean these arguments up!
    parser_shared_args = argparse.ArgumentParser(add_help=False)

    parser_shared_args.add_argument(
        "-cl",
        "--fs_clip_low",
        dest="fs_clip_low",
        help="Lower bound for the inclusion range of the fragment size distribution",
        type=int,
        default=0,
    )

    parser_shared_args.add_argument(
        "-ch",
        "--fs_clip_high",
        dest="fs_clip_high",
        help="Upper bound for the inclusion range of the fragment size distribution",
        type=int,
        default=300,
    )

    parser_shared_args.add_argument(
        "-cn",
        "--rc_clip_norm",
        dest="rc_clip_norm",
        help="Lower bound cutoff based on the normalized read count across regions (regions that fall below this cutoff are masked)",
        type=float,
        default=0.0001,
    )

    parser_shared_args.add_argument(
        "-ca",
        "--rc_clip_abs",
        dest="rc_clip_abs",
        help="Lower bound cutoff based on the absolute read count across regions (regions that fall below this cutoff are masked)",
        type=float,
        default=500,
    )

    parser_reference = subparsers.add_parser(
        "reference",
        help="Construct a reference set using healthy controls",
        parents=[parser_logger, parser_shared_args],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_reference.add_argument(
        "-i",
        "--in_npz",
        dest="in_npzs",
        nargs="+",
        help="Path to all NPZ files",
        type=str,
        required=True,
    )
    parser_reference.add_argument(
        "-o",
        "-out_ref",
        dest="outref",
        help="Path and filename for the reference output (e.g. path/to/myref.npz)",
        type=str,
        required=True,
    )
    parser_reference.add_argument(
        "-r",
        "--refsize",
        dest="refsize",
        help="Number of reference regions per region",
        default=300,
        type=int,
        required=False,
    )
    parser_reference.add_argument(
        "-b",
        "--binsize",
        dest="binsize",
        help="Scale samples to this region size (multiples of existing region size only)",
        default=500000,
        type=int,
        required=False,
    )
    parser_reference.set_defaults(func=wcr_reference)

    ####

    parser_test = subparsers.add_parser(
        "detect",
        help="Detect CNVs",
        parents=[parser_logger, parser_shared_args],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_test.add_argument(
        "-i",
        "--in-npz",
        dest="in_npz",
        type=str,
        help="Input sample npz file",
        required=True,
    )
    parser_test.add_argument(
        "-r",
        "--reference",
        dest="reference",
        type=str,
        help="Reference .npz, as previously created with reference",
        required=True,
    )
    parser_test.add_argument(
        "-o",
        "-out-id",
        dest="outid",
        type=str,
        help="Basename (w/o extension) of output files (paths are allowed, e.g. path/to/ID_1)",
        required=True,
    )
    parser_test.add_argument(
        "-mrb",
        "--min-ref-bins",
        dest="minrefbins",
        type=int,
        default=150,
        help="Minimum amount of sensible reference bins per target bin.",
        required=False,
    )
    parser_test.add_argument(
        "-m",
        "--maskrepeats",
        dest="maskrepeats",
        type=int,
        default=5,
        help="Regions with distances > mean + sd * 3 will be masked. Number of masking cycles.",
        required=False,
    )
    # parser_test.add_argument(
    #     "-a",
    #     "--alpha",
    #     dest="alpha",
    #     type=float,
    #     default=1e-4,
    #     help="P-value cut-off for calling a CBS breakpoint.",
    #     required=False,
    # )
    parser_test.add_argument(
        "-z",
        "--zscore",
        dest="zscore",
        type=float,
        default=5,
        help="Z-score cut-off for aberration calling.",
        required=False,
    )
    # parser_test.add_argument(
    #     "-b",
    #     "--beta",
    #     dest="beta",
    #     type=float,
    #     default=None,
    #     help="When beta is given, --zscore is ignored and a ratio cut-off is used to call aberrations. Beta is a number between 0 (liberal) and 1 (conservative) and is optimally close to the purity.",
    #     required=False,
    # )
    parser_test.set_defaults(func=wcr_detect)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    args.func(args)


if __name__ == "__main__":
    main()
