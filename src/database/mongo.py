import os
import sys

import pandas as pd
import numpy as np
import pymongo
import bson

from datetime import date, datetime, timedelta

from sshtunnel import SSHTunnelForwarder


# keys = ['249i4D', ]
# Poner comas en keys
keys = [
]

from_days = 10000

pd.options.mode.chained_assignment = None

MONGO_CONNECTION = {
    'MONGO_SSH_HOST': os.getenv('MONGO_SSH_HOST'),
    'MONGO_SSH_PORT': int(os.getenv('MONGO_SSH_PORT', 22)),
    'MONGO_SSH_USER': os.getenv('MONGO_SSH_USER'),
    'MONGO_SSH_PASSWORD': os.getenv('MONGO_SSH_PASSWORD'),
    'MONGO_SSH_LOCAL_PORT': int(os.getenv('MONGO_SSH_LOCAL_PORT', 27017)),

    'MONGO_HOST': os.getenv('MONGO_HOST', 'localhost'),
    'MONGO_PORT': int(os.getenv('MONGO_PORT', 27017)),
    'MONGO_USER': os.getenv('MONGO_USER'),
    'MONGO_PASSWORD': os.getenv('MONGO_PASSWORD'),

    'MONGO_DATABASE': os.getenv('MONGO_DATABASE', 'emcaredb'),
}



class EMCareDataAnalizer:

    def __init__(self, dbName, collectionRaw, keys=None, exclude_keys=None):
        self.dbName = dbName
        self.collectionRaw = collectionRaw

        self.keys = keys
        self.exclude_keys = exclude_keys

        self.ssh_host = MONGO_CONNECTION['MONGO_SSH_HOST']
        self.ssh_port = MONGO_CONNECTION['MONGO_SSH_PORT']
        self.ssh_username = MONGO_CONNECTION['MONGO_SSH_USER']
        self.ssh_password = MONGO_CONNECTION['MONGO_SSH_PASSWORD']
        self.ssh_local_port = MONGO_CONNECTION['MONGO_SSH_LOCAL_PORT']

        self.host = 'localhost'
        self.port = MONGO_CONNECTION['MONGO_PORT']
        self.username = None
        self.password = None

        self.tunnel = self._start_ssh_tunnel()
        self.db = self._connect_mongo_through_tunnel()

        self.processors = [
            AddedAndRemovedCountProcessor(self.db),
            #StepsProcessor(self.db)
        ]


    def get_keys(self):
        return list(self.db[self.collectionRaw].distinct("key"))


    def process_user_activities(self, key, lenght = None):
        n = 1

        if self.keys is None:
            self.keys = self.get_keys()
        n_users = len(self.keys)

        print("Starting process...")

        print(" ================================================================================ ")
        print("Key " + key + " - " + n.__str__() + "/" + n_users.__str__())

        for processor in self.processors:
            print(" ::::::: " + processor.collection + " ::::::: ")
            df = self._get_user_raw(key, processor.data_type)

            print("Obtained " + len(df).__str__() + " raw activities")
            if len(df) <= 0:
                continue

            process_df = processor.process(df)
            
            if lenght is not None:
                process_df = process_df.tail(lenght)

            print ("Generated " + len(process_df).__str__() + " rows after processing")
            try:
                #process_df.to_csv("./" + processor.collection + "/" + k + ".csv", index=False)
                # df_last.to_csv("./" + key + ".csv", index=False)
                return process_df
            except Exception as e:
                print("Error saving CSV: " + str(e))
            #self._save_dataframe(process_df, processor.collection)
    
    def _start_ssh_tunnel(self):
        tunnel = SSHTunnelForwarder(
                (self.ssh_host, self.ssh_port),
                ssh_username=self.ssh_username,
                ssh_password=self.ssh_password,
                remote_bind_address=(self.host, self.port),
                local_bind_address=('localhost', self.ssh_local_port)
        )
        tunnel.start()
        return tunnel

    def _connect_mongo_through_tunnel(self):
        if self.username and self.password:
            mongo_uri = f'mongodb://{self.username}:{self.password}@localhost:{self.tunnel.local_bind_port}/{self.dbName}'
            client = pymongo.MongoClient(mongo_uri)
        else:
            client = pymongo.MongoClient('localhost', self.tunnel.local_bind_port)

        #client = pymongo.MongoClient('localhost', self.tunnel.local_bind_port)
        
        return client[self.dbName]
    
    def _get_user_raw(self, key, data_type):
        """ Read from Mongo and Store into DataFrame """

        print("Reading Data")
        print({'key': key, 'em_type': data_type})

        from_time = date.today() - timedelta(days=from_days)
        from_time = datetime.combine(from_time, datetime.min.time())
        from_time = int(datetime.timestamp(from_time) * 1000)

        to_time = date.today()
        to_time = datetime.combine(to_time, datetime.min.time())
        to_time = int(datetime.timestamp(to_time) * 1000)

        # Make a query to the specific DB and Collection
        cursor = self.db[self.collectionRaw].find(
            {
                'key': key,
                'em_type': data_type,
                'em_time': {'$gte': from_time, '$lt': to_time}
            })

        print("Getting Dataframe...")
        # Expand the cursor and construct the DataFrame
        df = pd.DataFrame(list(cursor))

        return df
    
    def close(self):
        if self.tunnel:
            self.tunnel.stop()

class MetricProcessor:
    def __init__(self, db, collection, data_type):
        self.db = db
        self.collection = collection
        self.data_type = data_type

    def process(self, df):
        pass


class AddedAndRemovedCountProcessor(MetricProcessor):
    def __init__(self, db):
        MetricProcessor.__init__(self, db, 'keyboard_daily_stats', 16)

    def process(self, df):
        print("Processing " + str(self.data_type) + " data...")
        # print("Obtained " + len(df).__str__() + " raw activities")
        # print(df)
        df = df.loc[(df['em_type'] == self.data_type) & (df['em_added_count'] < 7)]
        print("Obtained " + len(df).__str__() + " type 16 activities")
        # print(df)
        #
        # print("Parsing time...")
        print("Saving data to CSV...")
        try:
            os.mkdir("./data/" + str(self.data_type))
        except:
            print("Directory already exists")
        try:
            df.to_csv("./data/" + str(self.data_type) + "/" + df['key'].iloc[0] + ".csv", index=False)
        except Exception as e:
            print("Error saving CSV: " + str(e))

        # Convertir 'em_time2' a tipo datetime
        df['em_time2'] = pd.to_datetime(df['em_time'], unit='ms')

        # Crear nuevas columnas
        df['date'] = df['em_time2'].dt.date
        df['year'] = df['em_time2'].dt.year
        df['month'] = df['em_time2'].dt.month
        df['week'] = df['em_time2'].dt.isocalendar().week
        df['day'] = df['em_time2'].dt.day
        df['hour'] = df['em_time2'].dt.hour
        df['minute'] = df['em_time2'].dt.minute
        df['second'] = df['em_time2'].dt.second

        # print(df)
        #
        # print("Grouping by second...")
        # Agrupar y sumar las columnas
        grouped_df = df.groupby(['key', 'date', 'year', 'month', 'week', 'day', 'hour', 'minute', 'second']).agg({
            'em_added_count': 'sum',
            'em_removed_count': 'sum'
        }).reset_index()

        # print(grouped_df)

        # print("Rename columns to minute...")
        # Renombrar las columnas resultantes
        grouped_df = grouped_df.rename(
            columns={'em_added_count': 'em_added_count_minute', 'em_removed_count': 'em_removed_count_minute'})
        # print(grouped_df)

        # print("Calc Index...")
        # Calcular el nuevo campo 'index'
        grouped_df['index'] = grouped_df['em_removed_count_minute'] / (
                    grouped_df['em_added_count_minute'] + grouped_df['em_removed_count_minute'])
        # print(grouped_df)

        # print("Filtering em_added_count_minute so is < 21...")
        # meto un filtro para coger solo valores menores de 21 (limite que pongo para evitar más datos aberrantes)
        grouped_df = grouped_df.loc[grouped_df['em_added_count_minute'] < 21]
        # print(grouped_df)

        # print("Group by day, getting statistical data...")
        # Agrupar por key, date, year, month, week, day y aplicar funciones estadísticas
        daily_stats_df = grouped_df.groupby(['key', 'date', 'year', 'month', 'week', 'day']).agg({
            'em_added_count_minute': ['mean', 'median', 'max', 'min'],
            'em_removed_count_minute': ['mean', 'median', 'max', 'min'],
            'index': ['mean', 'median', 'max', 'min'],  # También puedes calcular estadísticas para el campo 'index'
        }).reset_index()
        # print(daily_stats_df)

        # print("Rename stats...")
        # Renombrar las columnas resultantes
        daily_stats_df.columns = ['key', 'date', 'year', 'month', 'week', 'day',
                                  'mean_added_count', 'median_added_count', 'max_added_count', 'min_added_count',
                                  'mean_removed_count', 'median_removed_count', 'max_removed_count',
                                  'min_removed_count',
                                  'mean_index', 'median_index', 'max_index', 'min_index']

        min_date = daily_stats_df['date'].min()
        try:
            daily_stats_df['days_fu'] = (daily_stats_df['date'] - min_date).dt.days
        except:
            print('Exception generating days_fu')
            daily_stats_df['days_fu'] = 0

        daily_stats_df['date'] = pd.to_datetime(daily_stats_df['date'])
        # print(daily_stats_df)

        return daily_stats_df

def get_df_key(key, lenght=None):
    analyzer = EMCareDataAnalizer('emcaredb', 'activity', exclude_keys=exclude_keys, keys=keys)
    df = analyzer.process_user_activities(key, lenght)
    analyzer.close()
    return df