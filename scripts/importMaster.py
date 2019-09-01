import numpy as np
import gzip
import io
import os
import pandas as pd


def importData(dataPath):
    def read_user_data(file_path):

        def parse_body_of_csv(csv_str, n_features):
            # Read the entire CSV body into a single numeric matrix:
            full_table = np.loadtxt(io.StringIO(csv_str), delimiter=',', skiprows=1);

            # Timestamp is the primary key for the records (examples):
            timestamps = full_table[:, 0].astype(int);

            # Read the sensor features:
            X = full_table[:, 1:(n_features + 1)];

            # Read the binary label values, and the 'missing label' indicators:
            trinary_labels_mat = full_table[:, (n_features + 1):-1];  # This should have values of either 0., 1. or NaN
            M = np.isnan(trinary_labels_mat);  # M is the missing label matrix
            Y = np.where(M, 0, trinary_labels_mat) > 0.;  # Y is the label matrix

            return (X, Y, M, timestamps);

        def parse_header_of_csv(csv_str):
            # Isolate the headline columns:
            headline = csv_str[:csv_str.index('\n')]
            columns = headline.split(',');

            # The first column should be timestamp:
            assert columns[0] == 'timestamp';
            # The last column should be label_source:
            assert columns[-1] == 'label_source';

            # Search for the column of the first label:
            for (ci, col) in enumerate(columns):
                if col.startswith('label:'):
                    first_label_ind = ci;
                    break;
                pass;

            # Feature columns come after timestamp and before the labels:
            feature_names = columns[1:first_label_ind];
            # Then come the labels, till the one-before-last column:
            label_names = columns[first_label_ind:-1];
            for (li, label) in enumerate(label_names):
                # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
                assert label.startswith('label:');
                label_names[li] = label.replace('label:', '');
                pass;

            return (feature_names, label_names);

        # Read the entire csv file of the user:
        with gzip.open(file_path, 'rb') as fid:
            csv_str = fid.read();
            csv_str = csv_str.decode("utf-8")
            pass;

        (feature_names, label_names) = parse_header_of_csv(csv_str);
        n_features = len(feature_names);
        (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features);

        data = np.concatenate((X, Y), axis=1)
        data = pd.DataFrame(data, columns=feature_names + label_names)
        data["Timestamp"] = timestamps

        return data.reset_index(drop=True);

    user_files = os.listdir(dataPath)
    data_list = []
    for user_file in user_files:
        user = user_file.split(".", 1)[0]
        print(user)
        user_file_path = dataPath + user_file

        user_data = read_user_data(user_file_path)
        user_data["User"] = user
        data_list.append(user_data)

    data = pd.concat(data_list, axis=0)

    return data


if __name__ == "__main__":
    dataPath = "../data/features_labels/"
    data = importData(dataPath)

