## This is course material for Introduction to Python Scientific Programming
## Example code: save_hashing_bytes.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import os
import hashlib

source_filename = 'nasdaqlisted.txt'
hashtag_filename = 'md5'

try:
    # Obtain current python file's path
    path = os.path.dirname(os.path.abspath(__file__))
    # Open source file and the result file
    source_handle = open(path+'/'+source_filename,'r')
    #w+ allows you to read and write 
    hashtag_handle = open(path+'/'+hashtag_filename,'wb+')
    #text file
    '''
    hashtag_handle = open(path+'/'+hashtag_filename,'w+')
    '''

    #read source_filename
    md5 = hashlib.md5()
    for line in source_handle:
        byte_array = line.encode() # make sure getting a byte array
        md5.update(byte_array)


    # Write the binary digest values
    hashtag_handle.write(md5.digest())
    # hexdigest() = text file (sting return)
    '''
    hashtag_handle.write(md5.hexdigest())
    '''
    #when you finish writing, the pointer is at the end


    # Re-read from the beginning to verify
    # repositioning the pointer to the 0 position
    hashtag_handle.seek(0)
    hashtag_bytes = hashtag_handle.read()
    print(hashtag_bytes)
except IOError:
    print('IO Error! Please check valid file names and paths')
    exit
finally:
    source_handle.close()
    hashtag_handle.close()