import re
import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA

# ======================================== REALDISP_HAR_DATA ========================================
class REALDISP_HAR_DATA(BASE_DATA):
    """
    https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset

    REAListic sensor DISPlacement dataset (REALDISP)

    Brief summary of the dataset:
    ---------------------------------
    #Activities: 33 
    #Sensors: 9
    #Subjects: 17
    #Scenarios: 3   

    There are 4 sensors: Acc Gyro Mag Quat (13 sensor channels)
    Amounted in 9 places 'LC', 'LT', 'RC', 'RT', 'BACK', 'LLA', 'LUA', 'RLA', 'RUA'
    --> In total 117 channels

    Sensor modalities (acceleration, rate of turn, magnetic field, and quaternions).

    Each sensor provides 3D acceleration (accX,accY,accZ), 3D gyro (gyrX,gyrY,gyrZ), 3D magnetic field orientation (magX,magY,magZ) and 4D quaternions (Q1,Q2,Q3,Q4).
    The sensors are identified according to the body part on which is placed respectively.
    """

    def __init__(self, args):
        """
        root_path : Root directory of the data set
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data
            wavelet : Methods of wavelet transformation
        """
		
        # [2, 118], i.e. 117 channels, the first two columns are timestamps, the last one is activity_id
        self.used_cols = [i for i in range(2, 119)]

        col_list       = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ', 'Q1', 'Q2', 'Q3', 'Q4']
        pos_list       = ['LC', 'LT', 'RC', 'RT', 'BACK', 'LLA', 'LUA', 'RLA', 'RUA']
        self.col_names = [item for sublist in [[col + '_' + pos for col in col_list] for pos in pos_list] for item in sublist]

        # These two variables represent whether all sensors can be filtered according to position and sensor type
        # pos_filter ------- >  filter according to position
        # sensor_filter ----->  filter according to the sensor type
        self.pos_filter         = ['LC', 'LT', 'RC', 'RT', 'BACK', 'LLA', 'LUA', 'RLA', 'RUA']
        self.sensor_filter      = ['acc', 'gyr', 'mag', 'Q']

        # Selected_cols will be updated according to user settings. User have to set -- args.pos_select, args.sensor_select---
        self.selected_cols      = None
        # Filtering channels according to the Position
        self.selected_cols      = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names, "position")
        # Filtering channels according to the Sensor Type
        if self.selected_cols is None:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names, "Sensor Type")
        else:
            self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.selected_cols, "Sensor Type")

        # Activities
        self.label_map = [
            (0, '00'),  # No activity
            (1, '01'),  # Walking (A1)
            (2, '02'),  # Jogging (A2)
            (3, '03'),  # Running (A3)
            (4, '04'),  # Jump up (A4)
            (5, '05'),  # Jump front & back (A5)
            (6, '06'),  # Jump sideways (A6)
            (7, '07'),  # Jump leg/arms open/closed (A7)
            (8, '08'),  # Jump rope (A8)
            (9, '09'),  # Trunk twist (arms outstretched) (A9)
            (10, '10'), # Trunk twist (elbows bent) (A10)
            (11, '11'), # Waist bends forward (A11)
            (12, '12'), # Waist rotation (A12)
            (13, '13'), # Waist bends (reach foot with opposite hand) (A13)
            (14, '14'), # Reach heels backwards (A14)
            (15, '15'), # Lateral bend (10_ to the left + 10_ to the right) (A15)
            (16, '16'), # Lateral bend with arm up (10_ to the left + 10_ to the right) (A16)
            (17, '17'), # Repetitive forward stretching (A17)
            (18, '18'), # Upper trunk and lower body opposite twist (A18)
            (19, '19'), # Lateral elevation of arms (A19)
            (20, '20'), # Frontal elevation of arms (A20)
            (21, '21'), # Frontal hand claps (A21)
            (22, '22'), # Frontal crossing of arms (A22)
            (23, '23'), # Shoulders high-amplitude rotation (A23)
            (24, '24'), # Shoulders low-amplitude rotation (A24)
            (25, '25'), # Arms inner rotation (A25)
            (26, '26'), # Knees (alternating) to the breast (A26)
            (27, '27'), # Heels (alternating) to the backside (A27)
            (28, '28'), # Knees bending (crouching) (A28)
            (29, '29'), # Knees (alternating) bending forward (A29)
            (30, '30'), # Rotation on the knees (A30)
            (31, '31'), # Rowing (A31)
            (32, '32'), # Elliptical bike (A32)
            (33, '33'), # Cycling (A33)
        ]    

        # Drop activities without a label (0 is the label for no activity)
        self.drop_activities = [0]

        # One group as the test data and the other three groups' subjects as the source domains (each subject forms a source domain)
        self.train_keys   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.vali_keys    = []
        self.test_keys    = [13, 14, 15, 16]

        # According to AALH data preprocessing
        self.LOCV_keys = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        self.all_keys  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.sub_ids_of_each_sub = {}

        self.exp_mode  = args.exp_mode
        self.split_tag = "sub"

        self.file_encoding = {}  # no use 

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(REALDISP_HAR_DATA, self).__init__(args)

    def load_all_the_data(self, root_path):
        print("=" * 16, " Load all the data ", "=" * 16)

        df_dict = {}
        # Regex pattern to extract the subject ID, scenario, and mutual displacement if any.
        pattern = re.compile(r"subject(\d+)_([a-z]+)(\d*)\.log")

        for i, filename in enumerate(os.listdir(root_path)):
            match = pattern.match(filename)
            if not match:
                continue  # If the filename doesn't match the pattern, skip it

            sub, _, _ = match.groups()

            # Skip files with scenarios not in the specified scenarios list
            # if scenario not in self.scenarios:
            #     continue

            # Identifies the sequences, continuous or not
            sub_id = i

            sub = int(sub)   # Convert subject to int
            
            # Read the file and process it
            file_path = os.path.join(root_path, filename)
            sub_data = pd.read_csv(file_path, header=None, delim_whitespace=True)

            # Last column is the activity_id
            activity_id = sub_data.iloc[:, -1]

            # Include all rows but only the columns specified by self.used_cols
            sub_data = sub_data.iloc[:, self.used_cols]

            # Assign column names to all except the last column from sub_data
            sub_data.columns = self.col_names
            
            # Include additional information
            sub_data['sub_id']      = sub_id
            sub_data['sub']         = sub
            sub_data['activity_id'] = activity_id

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)

            df_dict[sub_id] = sub_data

        # Concatenate all dataframes
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # Reorder the columns as sensor1, sensor2... sensorN, sub, activity_id
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"] + ["activity_id"]]
        else:
            df_all = df_all[self.col_names + ["sub"] + ["activity_id"]]

        # Split the dataframe into features (X) and labels (y)
        data_y = df_all.iloc[:, -1]
        data_x = df_all.iloc[:, :-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorN, sub, 
        
        return data_x, data_y
