'''
Author: Wenshan Wang
Date: 2024-09-06

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os
# import sys

from colorama import Fore, Style

# Local imports.
from os.path import isdir, isfile, join
import argparse

def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)

def print_warn(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)

def print_highlight(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)

class AirLabDownloader(object):
    def __init__(self, bucket_name = 'tartandrive') -> None:
        from minio import Minio
        endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"

        # public key (for downloading): 
        access_key = "m7sTvsz28Oq3AicEDHFo"
        secret_key = "YVPGh367RnrT7G33lG6DtbaeuFZCqTE6KabMQClw"

        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    def download(self, filelist, destination_path):
        target_filelist = []
        for source_file_name in filelist:
            target_file_name = join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, None

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")

        return True, target_filelist


class TartandriveDownloader():
    def __init__(self, ):
        super().__init__()

        self.downloader = AirLabDownloader()


    def unzip_files(self, zipfilelist, target_folder):
        print_warn('Note unzipping will overwrite existing files ...')
        for zipfile in zipfilelist:
            if not isfile(zipfile) or (not zipfile.endswith('.zip')):
                print_error("The zip file is missing {}".format(zipfile))
                return False
            print('  Unzipping {} ...'.format(zipfile))
            cmd = 'unzip -q -o ' + zipfile + ' -d ' + target_folder
            os.system(cmd)
        print_highlight("Unzipping Completed! ")
            
    def download(self, target_path, unzip = False, **kwargs):
        """
        """
        with open('azfiles.txt', 'r') as f:
            lines = f.readlines()

        zipfilelist = [ll.strip() for ll in lines] 

        suc, targetfilelist = self.downloader.download(zipfilelist, target_path)
        if suc:
            print_highlight("Download completed! Enjoy using Tartandrive!")

        if unzip:
            self.unzip_files(targetfilelist)

        return True

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--download-dir', default='./',
                        help='root directory for downloaded files')

    args = parser.parse_args()

    downloader = TartandriveDownloader()
    downloader.download(args.download_dir)