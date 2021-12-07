import argparse
import logging
import os
import sys
import warnings
import numpy as np
import pysam
import re
import signal

from time import time
from collections import defaultdict

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

def scale_sample(sample, from_size, to_size):
    if not to_size or from_size == to_size:
        return sample

    if to_size == 0 or from_size == 0 or to_size < from_size or to_size % from_size > 0:
        logging.critical(f"Impossible binsize scaling requested: {int(from_size)} to {int(to_size)}")
        sys.exit()

    return_sample = dict()
    scale = to_size / from_size
    for chrom_name in sample:
        chr_data = sample[chrom_name]
        new_len = int(np.ceil(len(chr_data) / float(scale)))
        scaled_chr = np.zeros(new_len, dtype=np.int32)
        for i in range(new_len):
            scaled_chr[i] = np.sum(chr_data[int(i * scale):int(i * scale + scale)])
            return_sample[chrom_name] = scaled_chr
    return return_sample


def read_pair_gen(bam_pointer, stats_dict, region_string=None):
    """
    Return read pairs as they are encountered, includes the position of the first previously aligned reads
    """

    read_dict = defaultdict(lambda: [None, None])
    lr_pos = -1

    for read in bam_pointer.fetch(region=region_string):
        stats_dict["reads_seen"] += 1

        if not read.is_proper_pair:
            stats_dict["reads_pairf"] += 1
            continue

        if read.is_secondary or read.is_supplementary:
            stats_dict["reads_secondary"] += 1
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
    stats_dict = defaultdict(int)
    chr_re = re.compile("chr", re.IGNORECASE)
    bins_per_chr = {}
    for chrom in map(str, range(1, 23)):
        bins_per_chr[chrom] = None

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
            if chrom_name not in bins_per_chr:
                continue

            chr_size = bam_file.lengths[index]
            n_bins = (int(chr_size / float(binsize))) + 1

            logging.info(f"Processing: {chrom}, filling: {n_bins} bins")

            fs_dict = [defaultdict(int) for __ in range(n_bins)]
            rc_counts = np.zeros(int(bam_file.lengths[index] / float(binsize) + 1), dtype=np.int32)

            for ((read1, read1_prevpos), (read2, read2_prevpos)) in read_pair_gen(bam_file, stats_dict, chrom):
                if read1.pos == read1_prevpos or read2.pos == read2_prevpos:
                    stats_dict["reads_rmdup"] += 2
                    continue

                if read1.mapping_quality < min_mapq or read2.mapping_quality < min_mapq:
                    stats_dict["reads_mapq"] += 2
                    continue

                stats_dict["reads_kept"] += 2

                rc_counts[int(read1.pos / binsize)] += 1
                rc_counts[int(read2.pos / binsize)] += 1

                if read1.template_length > 0:
                    mid, insert_size = get_midpoint(read1, read2)
                else:
                    mid, insert_size = get_midpoint(read2, read1)
                fs_dict[int(mid // binsize)][insert_size] += 1

            bins_per_chr[chrom_name] = (rc_counts, fs_dict)

        qual_info = {
            "mapped": bam_file.mapped,
            "unmapped": bam_file.unmapped,
            "no_coordinate": bam_file.nocoordinate,
            "filter_rmdup": stats_dict["reads_rmdup"],
            "filter_mapq": stats_dict["reads_mapq"],
            "pre_retro": stats_dict["reads_seen"],
            "post_retro": stats_dict["reads_kept"],
            "pair_fail": stats_dict["reads_pairf"],
        }

    return bins_per_chr, qual_info


@profile
def wcr_convert(args):
    sample, qual_info = convert_bam(args.infile, args.binsize)
    np.savez_compressed(args.outfile,
                        binsize=args.binsize,
                        sample=sample,
                        quality=qual_info)


@profile
def main():
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='wisecondorFF')
    parser.add_argument('--loglevel',
                        type=str,
                        default='INFO',
                        choices=['info', 'warning', 'debug', 'error', 'critical'])
    subparsers = parser.add_subparsers()

    parser_convert = subparsers.add_parser('convert',
                                           description='Convert and filter a .bam file to a .npz',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_convert.add_argument('infile', type=str, help='.bam input file for conversion')
    parser_convert.add_argument('outfile', type=str, help='Output .npz file')
    parser_convert.add_argument('--binsize', type=float, default=5e4, help='Bin size (bp)')
    parser_convert.set_defaults(func=wcr_convert)

    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(format='[%(levelname)s - %(asctime)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=getattr(logging, "INFO", None))  # args.loglevel.upper()
    logging.debug('args are: {}'.format(args))

    args.func(args)


if __name__ == '__main__':
    main()
