import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

class final_project:
    def __init__(self, data):
        self.data = data

########################################################################################################################
# introduction -

    def import_data(self):
        """
           import data.
           """
        col_names = ['id', 'classes', 'cap_shape', 'cap_surface', 'cap_color', 'odor', 'gill_attachment', 'gill_spacing', 'stalk_shape', 'veil_color', 'ring_number', 'population', 'latitude', 'longitude', 'veil_type']
        data = pd.read_csv(self.data, header=None, sep=',', names=col_names, index_col='id')
        # drop first row that contain numbers -
        data = data.drop(data.index[0])
        # show all columns -
        pd.set_option('display.max_columns', None)

        self.data = data

    def introduction(self):
        """
           import data and exploring features.
           """
        # import data from CSV -
        self.import_data()
        print('data table: ')
        print(self.data.head(), '\n')
        print('data info: ')
        print(self.data.info(), '\n')
        # there is no data in column 'veil-type', so we will drop that column -
        self.data = self.data.drop(columns=['veil_type'])
        print('data size: ')
        # equals to number of data's rows=8124 X number of data's columns=13
        print(self.data.size, '\n')

########################################################################################################################
# initial_data_analysis -

    def ordinal(self):
        """
           for all ordinal features we add a new column to the dataframe which
           holds the ordered values representing each original value of that column.
           """
        new_data = self.data[['gill_attachment', 'gill_spacing', 'ring_number']].copy()
        gill_attachment = ['f', 'n', 'a', 'd']
        gill_spacing = [0, -1, 1]
        ring_number = ['n', 'o', 't']

        new_data['gill_attachment_ord'] = (pd.Categorical(self.data.gill_attachment, ordered=True, categories=gill_attachment).codes + 1)
        new_data['gill_spacing_ord'] = pd.Categorical(self.data.gill_spacing, ordered=True, categories=gill_spacing).codes + 1
        new_data['ring_number_ord'] = pd.Categorical(self.data.ring_number, ordered=True, categories=ring_number).codes + 1

        return new_data

    def missing_data(self):
        """
           check for missing data in dataframe columns.
           """
        columns = []
        for column in self.data:
            """
               iterate over data columns to check if there is data missing.
               """
            count = 0
            features_value = pd.isna(self.data[column])
            for index, value in features_value.items():
                if value == True:
                    count += 1

            if count > 0:
                columns.append(column)
        print('dataframe columns with missing information: ', columns, '\n')

    def initial_data_analysis(self):
        """
           analysis data and the connection between features.
           """
        # gives us initial analyze of the data -
        print('data describe: ')
        describe = self.data.describe(include='all')
        print(describe, '\n')
        # import data description to CSV file -
        # describe.to_csv('description.csv')//open comment when submit

        # check if and which column has missing data -
        self.missing_data()

        # 3 columns have missing information. Those columns are categorical + ordinal
        # in order to replace the missing data with the mean we will convert those
        # ordinal columns to hold ordered values, using method 'ordinal()'.
        missing_data_columns = self.ordinal()
        print('data table: ')
        print(missing_data_columns.describe(include='all'), '\n')

        # we can see from describe() the means for each new column -
        # 'gill_attachment_ord' mean ~ 1 = 'f'
        # 'gill_spacing_ord' mean ~ 2 = '-1'
        # 'ring_number_ord' mean ~ 2 = 'o'


        # replace NaN with a scalar value -
        self.data['gill_attachment'] = self.data['gill_attachment'].fillna('f')
        self.data['gill_spacing'] = self.data['gill_spacing'].fillna(-1)
        self.data['ring_number'] = self.data['ring_number'].fillna('o')

        # check, to see that we fixed all NaN values -
        self.missing_data()
        # save fixed data to CSV file -
        # self.data.to_csv("./fixed_data")//open comment when submit

########################################################################################################################
# exploratory_data_analysis -

    def nominal(self):
        """
           for all nominal features we create a new column which
           holds the binary or one-hot encoding values representing each nominal value of that column.
           """
        classes_n = pd.get_dummies(self.data['classes'])
        cap_shape_n = pd.get_dummies(self.data['cap_shape'])
        cap_surface_n = pd.get_dummies(self.data['cap_surface'])
        cap_color_n = pd.get_dummies(self.data['cap_color'])
        stalk_shape_n = pd.get_dummies(self.data['stalk_shape'])
        veil_color_n = pd.get_dummies(self.data['veil_color'])

        nominal_features = {}
        nominal_features['classes_n'] = classes_n
        nominal_features['cap_shape_n'] = cap_shape_n
        nominal_features['cap_surface_n'] = cap_surface_n
        nominal_features['cap_color_n'] = cap_color_n
        nominal_features['stalk_shape_n'] = stalk_shape_n
        nominal_features['veil_color_n'] = veil_color_n

        return nominal_features

    def create_plot(self, corr_):
        """
            generate heatmap diagram.
            """

        mask = np.triu(np.ones_like(corr_, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr_, mask=mask, cmap=cmap, center=0,square=True, linewidths=.5)
        plt.show()
        plt.close()

    def features_relation(self):
        pass

    def exploratory_data_analysis(self):
        """
            pass.
            """

        # data_corr = self.data.select_dtypes(np.number)
        # corrMatrix = data_corr.corr()
        # fig, ax = plt.subplots(figsize=(5, 5))
        # mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
        # cmap = sns.diverging_palette(200, 10, as_cmap=True)
        #
        # sns.heatmap(corrMatrix, mask=mask, cmap=cmap, annot=True, center=0, linewidth=.5, ax=ax)
        # plt.show()
        # plt.close()

        # we would like to explore the feature 'classes', which indicates whether the mushroom is poisonous or edible -
        print('number of poisonous & edible mushrooms: ', self.data.classes.value_counts().head(), '\n')

        nominal_features = self.nominal()

        # check dependencies between features -
        corr_ = pd.concat([nominal_features['classes_n'], nominal_features['veil_color_n']], axis=1).corr()
        self.create_plot(corr_)


def main():
    data = "mushrooms3.csv"
    f = final_project(data)
    # introduction -
    f.introduction()
    # initial_data_analysis -
    f.initial_data_analysis()
    # exploratory_data_analysis -
    f.exploratory_data_analysis()


# Calling main function
if __name__ == "__main__":
    main()
