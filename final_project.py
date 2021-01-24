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
           analysis data, the connection between features and fill missing information.
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

        classes_bin = np.where(self.data['classes'] == "e", 1, 0)
        stalk_shape_bin = np.where(self.data['stalk_shape'] == "e", 1, 0)

        nominal_features = {}
        nominal_features['classes_n'] = classes_n
        nominal_features['cap_shape_n'] = cap_shape_n
        nominal_features['cap_surface_n'] = cap_surface_n
        nominal_features['cap_color_n'] = cap_color_n
        nominal_features['stalk_shape_n'] = stalk_shape_n
        nominal_features['veil_color_n'] = veil_color_n
        nominal_features['classes_bin'] = classes_bin
        nominal_features['stalk_shape_bin'] = stalk_shape_bin

        return nominal_features

    def create_plot(self, corrMatrix):
        """
            generate heatmap diagram.
            """
        fig, ax = plt.subplots(figsize=(5, 5))
        mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)

        sns.heatmap(corrMatrix, mask=mask, annot=True, cmap=cmap, center=0, linewidth=.5, ax=ax)
        plt.show()
        plt.close()

    def create_nominal_df(self, df, column, nominal_features):
        """
          create nominal dataframe.
          """

        for feature in nominal_features[column].columns:
            df[column + '_' + feature] = nominal_features[column][feature]

        return df

    def features_relation(self, nominal_features):
        """
           check dependencies between features.
           """
        # numerical features correlation heatmap -
        corrMatrix = self.data.corr()
        self.create_plot(corrMatrix)

        # ordinal features correlation heatmap -
        ordinal_data = (self.ordinal())[['gill_attachment_ord', 'gill_spacing_ord', 'ring_number_ord']].copy()
        corrMatrix = ordinal_data.corr()
        self.create_plot(corrMatrix)

        # create nominal features dataframe -
        nominal_data = pd.DataFrame()
        for column in nominal_features:
            if column != 'classes_bin' and column != 'stalk_shape_bin':
                self.create_nominal_df(nominal_data, column, nominal_features)

        # numerical with nominal features correlation heatmap -
        new_data = nominal_data.copy()
        new_data['population'] = self.data['population']
        new_data['odor'] = self.data['odor']

        corrMatrix = new_data.corr()
        fig, ax = plt.subplots(figsize=(13, 13))
        mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)

        sns.heatmap(corrMatrix, mask=mask, annot=True, cmap=cmap, center=0, linewidth=.5, ax=ax, fmt=".1f")
        plt.show()
        plt.close()

        # ordinal with nominal features correlation heatmap -
        new_data = nominal_data.copy()
        new_data['gill_attachment_ord'] = ordinal_data['gill_attachment_ord']
        new_data['gill_spacing_ord'] = ordinal_data['gill_spacing_ord']
        new_data['ring_number_ord'] = ordinal_data['ring_number_ord']

        corrMatrix = new_data.corr()
        fig, ax = plt.subplots(figsize=(13, 13))
        mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)

        sns.heatmap(corrMatrix, mask=mask, annot=True, cmap=cmap, center=0, linewidth=.5, ax=ax, fmt=".1f")
        plt.show()
        plt.close()

        # ordinal with numerical features correlation heatmap -
        new_data = ordinal_data.copy()
        new_data['population'] = self.data['population']
        new_data['odor'] = self.data['odor']
        corrMatrix = new_data.corr()
        self.create_plot(corrMatrix)

    def feature_graphs(self):
        """
           understand the numeric features distribution.
           """

        sns.histplot(self.data.population, bins=100, kde=True)
        plt.show()
        plt.close()

        sns.histplot(self.data.odor, bins=100, kde=True)
        plt.show()
        plt.close()

        sns.histplot(self.data.longitude, bins=100, kde=True)
        plt.show()
        plt.close()

        sns.histplot(self.data.latitude, bins=100, kde=True)
        plt.show()
        plt.close()

        sns.histplot(self.data.global_address, bins=100, kde=True)
        plt.show()
        plt.close()

    def classes_relations(self, nominal_features):
        """
          explore classes relations with other features.
          """
        # we can see that when the odor is > 15 the chances for the mushroom to be poisonous are higher -
        sns.jointplot(x='classes', y='odor', kind='scatter', data=self.data)
        plt.show()
        plt.close()

        new_data = self.data[['odor']].copy()
        new_data['classes_e'] = nominal_features['classes_n']['e']
        new_data['classes_p'] = nominal_features['classes_n']['p']
        corrMatrix = new_data.corr()
        self.create_plot(corrMatrix)

    def population_relations(self, nominal_features):
        """
          explore population relations with other features.
          """
        new_data = self.data[['population']].copy()
        # gill_spacing = [0, -1, 1] -> 1 = crowded, 2 = close, 3 = distant
        new_data['gill_spacing_ord'] = (self.ordinal())[['gill_spacing_ord']].copy()
        # edible = 1, poisonous = 0
        new_data['classes_bin'] = nominal_features['classes_bin']
        sns.pairplot(new_data)
        plt.show()
        plt.close()

        adress, classes = self.data['global_address'], new_data['classes_bin']
        population = new_data['population']
        gill_spacing = new_data['gill_spacing_ord']

        for ar in [1, 2, 3]:
            plt.scatter([], [], c='k', alpha=0.3, s=ar * 100, label=str(ar) + ' gill_spacing')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='gill_spacing')

        plt.scatter(population, adress, label=None, c=classes, cmap='viridis', s=gill_spacing * 200, linewidth=0,
                    alpha=0.5)

        plt.axis("equal")
        plt.xlabel('population')
        plt.ylabel('global_address')
        plt.colorbar(label="classes")

        plt.title("feature: population, gill_spacing_ord X classes_bin")
        plt.show()
        plt.close()

    def gill_attachment_relations(self, nominal_features):
        """
          explore gill_attachment relations with other features.
          """
        # gill_attachment = ['f', 'n', 'a', 'd'] 1 = free, 2 = notched , 3 = attached, 4 = descending
        new_data = (self.ordinal())[['gill_attachment_ord']].copy()
        # edible = 1, poisonous = 0
        new_data['classes_bin'] = nominal_features['classes_bin']
        adress, classes = self.data['global_address'], new_data['classes_bin']
        gill_attachment_ord = new_data['gill_attachment_ord']
        # w = white, n = brown, o = orange, y = yellow
        veil_color = self.data['veil_color']
        for ar in [1, 2, 3, 4]:
            plt.scatter([], [], c='k', alpha=0.3, s=ar * 100, label=str(ar) + ' gill_attachment')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='gill_attachment')

        plt.scatter(veil_color, adress, label=None, c=classes, cmap='viridis', s=gill_attachment_ord * 100, linewidth=0,
                    alpha=0.5)

        plt.axis("equal")
        plt.xlabel('veil_color')
        plt.ylabel('global_address')
        plt.colorbar(label="classes")

        plt.title("feature: gill_attachment, veil_color X classes_bin")
        plt.show()
        plt.close()

    def exploratory_data_analysis(self):
        """
            explore features graphs and features relationship.
            """
        # create new feature based on Latitude and Longitude -
        self.data['global_address'] = self.data['latitude'] / self.data['longitude']

        # self.feature_graphs()//open
        nominal_features = self.nominal()

        # explore features relations -
        # self.features_relation(nominal_features)//open

        # feature classes -
        # explore classes (edible, poisonous) relations with other features we saw high correlation with -
        # self.classes_relations(nominal_features)//open

        # feature population -
        # self.population_relations(nominal_features)//open

        # feature gill attachment -
        # self.gill_attachment_relations(nominal_features)//open

        # >> to do:
        # pivot table for classes and maybe for population - data exploration 2 -> 41:00


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
