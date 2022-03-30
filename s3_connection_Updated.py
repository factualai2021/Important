# -*- coding: utf-8 -*-
# Copyright (c) 2020, IBM and/or its affiliates.
# All rights reserved.
"""
Class to do operations on predefined/assigned s3 buckets/Objects
Available Methods:
create_bucket
list_all_buckets
list_my_bucket
retrieve_from_bucket
get_csv
save_as_csv
save_model
examples:
----------
import s3_connection as s3
s3_obj = s3.Connection()
s3_obj.create_bucket("abc")
@author: Asad, Jawed
Kumar S G, Khanij - added the following modules - get_csv,save_as_csv,save_model,load_model
"""
#load libraries

from minio import Minio
#from minio.error import BucketAlreadyOwnedByYou,BucketAlreadyExists
import pandas as pd
import json
import os
import pickle
import io
from keras.models import *
from keras.layers import *

class Connection():
    """
        Class to perform different operations on S3 bucket's objects
    """
    def __init__(self):
        self.client = self.make_connection()

    def make_connection(self):
        """
        Class method to be used internally to make connection to s3 storage
        Ensure that file "configuration.json" is either present in the PATH or is added to current directory
        """
        with open('configuration.json') as conf_file:
            conf = json.load(conf_file)
        client = Minio(
        endpoint=conf["endpoint"],
        access_key=conf["access_key"],
        secret_key=conf["secret_key"],
        secure=False
        )
        return client


    def create_bucket(self, bucket_name):
        """
        Create bucket/folder in s3 storage space
        Parameters
        ----------
        bucket_name : any string in lowercase
        Returns
        -------
        None
        """
        found = self.client.bucket_exists(bucket_name)
        if not found:
            # Make a bucket with the make_bucket API call.
            try:
                self.client.make_bucket(bucket_name, location="us-east-1")
                print("Bucket ", bucket_name, " created")
            except BucketAlreadyOwnedByYou as err:
                pass
            except BucketAlreadyExists as err:
                pass
            except ResponseError as err:
                raise
        else:
            print("Bucket ", bucket_name, " already exists")


    def list_all_buckets(self):
        """
        List all the available buckets/folders in s3 storage space
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        buckets = self.client.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)


    def list_my_bucket(self, name):
        """
        List all the objects present within mentioned bucket
        Parameters
        ----------
        Name : Name of the bucket as string
        Returns
        -------
        None
        """
        objects = self.client.list_objects(name)
        for obj in objects:
            print(obj)


    def retrieve_from_bucket(self, bucket_name, remote_object, local_object):
        """
        Retrieve required object from mentioned buckets/folders in s3 storage space
        Parameters
        ----------
        bucket_name : Name of the bucket as string
        remote_object : Name of the remote objectwhich is to be retrieved as string
        local_object : Name by which the object will be received on the local system in string
        Returns
        -------
        None
        """
        self.client.fget_object(bucket_name, remote_object, local_object)

    def retrieve_from_bucket(self, bucket_name, remote_object, local_object):
        """
        Retrieve required object from mentioned buckets/folders in s3 storage space

        Parameters
        ----------
        bucket_name : Name of the bucket as string
        remote_object : Name of the remote objectwhich is to be retrieved as string
        local_object : Name by which the object will be received on the local system in string

        Returns
        -------
        None
        """
        self.client.fget_object(bucket_name, remote_object, local_object)


    def upload_to_bucket(self, bucket_name, upload_object_path, local_object):
        """
        Uploads required object from local to mentioned buckets/folders in s3 storage space

        Parameters
        ----------
        bucket_name : Name of the bucket as string (Very first level of bucket, e.g., 'customer1')
        remote_object : Name of the complete path of s3 remote location (with name of object) where
                        object will be saved e.g., 'iot/battery/models/abc.png'. Here the first part
                        i.e, 'iot/battery/models/' refers to folder structure within outur bucket
                        'customer1' and second part i.e, 'abc.png' refers to name of the object itself
        local_object : Name by which the object is available locally as string

        Returns
        -------
        None
        """
        self.client.fput_object(bucket_name, upload_object_path, local_object)

    def get_csv(self,bucket_name, csv_location):
        """
        Retrieve required a dataframe from S3 bucket
        Parameters
        ----------
        bucket_name : Name of the S3 bucket as string
        csv_location : Location of the csv to be read from S3.
        Returns
        -------
        Data frame


        examples:
        ----------
        import s3_connection as s3
        s3_obj = s3.Connection()
        df = s3_obj.get_csv('customer1','bais/telecom/input/cell2celltrain.csv' )

        """
        obj = self.client.get_object(bucket_name,csv_location)
        df = pd.read_csv(obj)

        return df

    def save_as_csv(self, bucket_name, csv_location, df):
        """
        Saves the given dataframe as a csv in the location bucket_name/csv_location
        Parameters
        ----------
        bucket_name : Name of the S3 bucket as string
        csv_location : Location where the csv needs to be saved
        df : Pandas Dataframe which needs to be saved
        Returns
        -------
        None

        examples:
        ----------
        import s3_connection as s3
        s3_obj = s3.Connection()
        s3_obj.save_as_csv('customer1','bais/telecom/input/cell2celltrain.csv', df)

        """
        csv = df.to_csv(index = False).encode('utf-8')
        self.client.put_object(
            bucket_name,
            csv_location,
            data=io.BytesIO(csv),
            length=len(csv),
            content_type='application/csv'
        )

    def save_model(self, bucket_name, model_location, model):
        """
        Saves the given model at  buckets/model_location in s3 storage space
        Parameters
        ----------
        bucket_name : Name of the bucket as string
        model_location :  Location where the model needs to be saved as a pickle object
        model : Model object to be saved as a pickle object

        Returns
        -------
        None

        examples:
        ----------
        import s3_connection as s3
        s3_obj = s3.Connection()
        s3_obj.save_model('customer1','bais/telecom/models/ChurnModel.p', model_object)

        """
        bytes_file = pickle.dumps(model)

        self.client.put_object(
            bucket_name=bucket_name,
            object_name=model_location,
            data=io.BytesIO(bytes_file),
            length=len(bytes_file)
        )

    def load_model(self, bucket_name, model_location):
        """
        return the model object from the s3 location bucket_name/model_location
        Parameters
        ----------
        bucket_name : Name of the bucket as string
        model_location :  Location from where the model needs to be loaded

        Returns
        -------
        model object

        examples:
        ----------
        import s3_connection as s3
        s3_obj = s3.Connection()
        model_object = s3_obj.load_model('customer1','bais/telecom/models/ChurnModel.p')

        """
        model = pickle.loads(self.client.get_object(bucket_name=bucket_name, object_name=model_location).read())
        #model = load_model(self.client.get_object(bucket_name=bucket_name, object_name=model_location).read())

        return model
