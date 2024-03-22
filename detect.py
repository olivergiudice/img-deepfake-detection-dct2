import DCT2
import time
import argparse
from PIL import Image
import datetime
import hashlib

def image2pil(file):
    '''
    Converts a file to pillow format.

    :param file: file to convert
    :type file: FileStorage

    :return: pillow representation
    :rtype: PIL.image
    '''
    img = Image.open(file)
    #if ext[1] == 'png':
    return img.convert("RGB")

def readable_timestamp(timestamp):
    '''
    Converts timestamp to readable. 

    :param timestamp: timestamp
    :type timestamp: float

    :return: human readable timestamp
    :rtype: string
    '''
    return str(datetime.datetime.fromtimestamp(timestamp))


def hash_FileStorage(file_storage, alg='sha1'):
    '''
    Compute the hash string for a given file.

    :param file_storage: file instance
    :type file_storage: FileStorage
    :param alg: hash function uses 'sha1' or 'md5'. Default: sha1
    :type alg: string

    :returns: a tuple containing the hashing algorithm used and the computed hash
    :rtype: tuple of string
    '''
    assert alg in ('sha1', 'md5')

    file_storage.seek(0)
    return ("sha1", hashlib.sha1(file_storage.read()).hexdigest()) if alg == "sha1" else ("md5", hashlib.md5(file_storage.read()).hexdigest())



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Runs the DCT2 based deepfake detector for faces.')
    ap.add_argument('filename')
    args = ap.parse_args()

    filename = args.filename

    start_time = time.time()

    image = image2pil(filename)
    result = DCT2.execute(image)
    hash_alg, hash_value = hash_FileStorage(open(filename,'rb'))

    result_dict = {
                "timestamp_start": readable_timestamp(start_time),
                "file": {
                    'hash': hash_value,
                    'hash_alg': hash_alg,
                    'name': filename
                },
                "result": result
            }
    
    print(result_dict)





