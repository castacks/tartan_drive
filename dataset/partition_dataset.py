import os
import argparse
import itertools

from os_util import maybe_mkdir

if __name__ == '__main__':
    """
    Generate test/train from the paper
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_fp', type=str, required=True, help='dir containing dataset')
    parser.add_argument('--save_to', type=str, required=True, help='dir to save to')
    parser.add_argument('--partition_fp', type=str, required=False, default='data_partition', help='The path to look for the dataset partition')
    args = parser.parse_args()

    maybe_mkdir(args.save_to)

    partitions = {}
    for tfp in os.listdir(args.partition_fp):
        maybe_mkdir(os.path.join(args.save_to, tfp), force=True)
        with open(os.path.join(args.partition_fp, tfp), 'r') as fp:
            s = fp.read()
            tfps = s.split(', ')[:-1]
            partitions[os.path.splitext(tfp)[0]] = tfps

    all_files = list(itertools.chain(partitions.values()))[0]
    processed_files = []
    missing_files = []
    extra_files = []
    for k,v in partitions.items():
        for tfp in v:
            full_tfp = os.path.join(args.dataset_fp, tfp)
            if os.path.exists(full_tfp + '.pt'):
                os.rename(full_tfp + '.pt', os.path.join(args.save_to, k, tfp) + '.pt')
                processed_files.append(tfp)
            elif os.path.exists(full_tfp + '.bag'):
                os.rename(full_tfp + '.pt', os.path.join(args.save_to, k, tfp) + '.bag')
                processed_files.append(tfp)
            else:
                missing_files.append(tfp)

    extra_files = list(set(processed_files) - set(all_files))
    if len(missing_files) == 0 and len(extra_files) == 0:
        print('Partitioning done. No errors')
    else:
        if len(missing_files) > 0:
            print('These files were in the partitioning description, but not foud in the data folder:')
            for fp in missing_files:
                print('\t', fp)
        if len(extra_files) > 0:
            print('These files were in the data folder, but not foud in the partitioning description:')
            for fp in extra_files:
                print('\t', fp)
