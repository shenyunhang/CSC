from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
import os


def binaryfile_to_blobproto_to_array(file_path):
    # input the filepath save by function WriteProtoToBinaryFile in caffe
    # output the array data

    assert os.path.exists(file_path),'File does not exists: %s'%file_path

    binary_data = open(file_path, 'rb').read()
    blob_proto = caffe_pb2.BlobProto()
    blob_proto.ParseFromString(binary_data)
    array_data=blobproto_to_array(blob_proto)

    return array_data
