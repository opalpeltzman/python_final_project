import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

class final_project:
    def __init__(self, data):
        self.data = data
        self.nominal_data = pd.DataFrame()
        self.ordinal_data = pd.DataFrame()

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

        # we can see from describe() the means\ top for each new\ original column -
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
        self.ordinal_data = ordinal_data
        corrMatrix = ordinal_data.corr()
        self.create_plot(corrMatrix)

        # create nominal features dataframe -
        nominal_data = pd.DataFrame()
        for column in nominal_features:
            if column != 'classes_bin' and column != 'stalk_shape_bin':
                self.create_nominal_df(nominal_data, column, nominal_features)

        self.nominal_data = nominal_data
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
        # odor -
        # we can see that when the odor is > 15 the chances for the mushroom to be poisonous are higher -
        sns.jointplot(x='classes', y='odor', kind='scatter', data=self.data)
        plt.show()
        plt.close()

        new_data = self.data[['odor']].copy()
        new_data['classes_e'] = nominal_features['classes_n']['e']
        new_data['classes_p'] = nominal_features['classes_n']['p']
        corrMatrix = new_data.corr()
        self.create_plot(corrMatrix)

        # odor, population, cap surface -
        data = self.data[['classes', 'population', 'odor']].copy()
        data['cap_surfaces'] = self.create_nominal_df_ord(nominal_features)['cap_surfaces'].copy()
        data['gill_spacing_ord'] = (self.ordinal())[['gill_spacing_ord']].copy()
        data['ring_number_ord'] = (self.ordinal())[['ring_number_ord']].copy()
        data['classes_bin'] = nominal_features['classes_bin']

        odor, classes = data['odor'], data['classes_bin']
        population, cap_surface = data['population'], data['cap_surfaces']
        ring_number = data['ring_number_ord']


        for ar in [1, 2, 3]:
            plt.scatter([], [], c='k', alpha=0.3, s=(ar * 3) ** 2, label=str(ar) + ' ring_number')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='ring_number')

        plt.scatter(cap_surface, population, label=None, c=classes, cmap='vlag', s=(ring_number * 3) ** 2, linewidth=0,
                    alpha=0.5)
        plt.xlim(0, 5)
        plt.xlabel('cap_surface')
        plt.ylabel('population')
        plt.colorbar(label="classes")

        plt.title("classes relations")
        plt.show()
        plt.close()

    def population_relations(self, nominal_features):
        """
          explore population relations with other features.
          """
        new_data = self.data[['population', 'odor']].copy()
        # gill_spacing = [0, -1, 1] -> 1 = crowded, 2 = close, 3 = distant
        new_data['gill_spacing_ord'] = (self.ordinal())[['gill_spacing_ord']].copy()
        # edible = 1, poisonous = 0
        new_data['classes_bin'] = nominal_features['classes_bin']
        sns.pairplot(new_data)
        plt.show()
        plt.close()

        classes = new_data['classes_bin']
        population = new_data['population']
        gill_spacing = new_data['gill_spacing_ord']
        odor = new_data['odor']

        for ar in [1, 2, 3]:
            plt.scatter([], [], c='k', alpha=0.3, s=ar ** 5, label=str(ar) + ' gill_spacing')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='gill_spacing')

        plt.scatter(population, odor, label=None, c=classes, cmap='vlag', s=gill_spacing ** 5, linewidth=0,
                    alpha=0.5)


        plt.xlabel('population')
        plt.ylabel('odor')
        plt.colorbar(label="classes")

        plt.title("classes relations")
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
        odor, classes = self.data['odor'], new_data['classes_bin']
        gill_attachment_ord = new_data['gill_attachment_ord']
        # w = white, n = brown, o = orange, y = yellow
        veil_color = self.data['veil_color']
        for ar in [1, 2]:
            plt.scatter([], [], c='k', alpha=0.3, s=(ar * 5) ** 2, label=str(ar) + ' classes')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='classes')

        plt.scatter(veil_color, odor, label=None, c=gill_attachment_ord, cmap='Paired', s=(classes * 5) ** 2, linewidth=0,
                    alpha=0.5)

        plt.xlabel('veil_color')
        plt.ylabel('odor')
        plt.colorbar(label="gill_attachment_ord")

        plt.title("gill_attachment relations")
        plt.show()
        plt.close()

    def create_nominal_df_ord(self, nominal_features):
        """
            create dataframe that holds one column for each nominal feature with its converted values to numbers.
            """
        veil_color_values = self.data.veil_color.unique().tolist()
        cap_color_values = self.data.cap_color.unique().tolist()
        cap_surface_values = self.data.cap_surface.unique().tolist()
        cap_shape_values = self.data.cap_shape.unique().tolist()

        new_data = pd.DataFrame()
        # e=1, t=0
        new_data['stalk_shape_bin'] = nominal_features['stalk_shape_bin']
        # veil_color_values:  ['w', 'n', 'o', 'y'] -> w=1, n=2, o=3, y=4
        new_data['veil_colors'] = (
                pd.Categorical(self.data.veil_color, ordered=True, categories=veil_color_values).codes + 1)
        # cap_color_values:  ['n'= 1, 'y'= 2, 'w'= 3, 'g'= 4, 'e'= 5, 'p'= 6, 'b'= 7, 'u'= 8, 'c'= 9, 'r'= 10]
        new_data['cap_colors'] = (
                pd.Categorical(self.data.cap_color, ordered=True, categories=cap_color_values).codes + 1)
        # cap_surface_values:  ['s', 'y', 'f', 'g'] -> s=1, y=2, f=3, g=4
        new_data['cap_surfaces'] = (
                pd.Categorical(self.data.cap_surface, ordered=True, categories=cap_surface_values).codes + 1)
        # cap_shape_values:  ['X'=1 , 'B'=2, 'S'=3, 'F'=4, 'K'=5, 'C'=6]
        new_data['cap_shapes'] = (
                pd.Categorical(self.data.cap_shape, ordered=True, categories=cap_shape_values).codes + 1)

        return new_data

    def classes_pivot_table(self, nominal_features):
        """
          generate pivot table for feature classes.
          """
        new_data = self.create_nominal_df_ord(nominal_features)
        # e=1, p=0
        new_data['classes'] = self.data['classes'].copy()

        # pivot table with numerical features -
        print('pivot table with numerical features -')
        platform_numerical = pd.pivot_table(self.data, columns="classes", aggfunc=np.mean)
        print(platform_numerical.head(6), '\n')

        print(new_data.groupby('classes')['cap_shapes'].value_counts())
        print(new_data.groupby('classes')['cap_surfaces'].value_counts())
        print(new_data.groupby('classes')['stalk_shape_bin'].value_counts())
        print(new_data.groupby('classes')['veil_colors'].value_counts())
        print(new_data.groupby('classes')['cap_colors'].value_counts())
        # pivot table with nominal features -
        # will get the highest values first -
        print('pivot table with nominal features -')
        platform_nominal = pd.pivot_table(new_data, columns="classes",
                                          aggfunc=lambda x: x.value_counts()[:1].sort_values(ascending=False).index)
        print(platform_nominal.head(), '\n')

        # pivot table with ordinal features -
        print('pivot table with ordinal features -')
        ordinal_data = (self.ordinal())[['gill_attachment_ord', 'gill_spacing_ord', 'ring_number_ord']].copy()
        ordinal_data['classes'] = self.data['classes'].copy()
        platform_ordinal = pd.pivot_table(ordinal_data, columns="classes",
                                          aggfunc=lambda x: x.value_counts()[:1].sort_values(ascending=False).index)
        print(platform_ordinal.head(), '\n')
        print('classes feature relation with ordinal features -')
        print(ordinal_data.groupby('classes')['gill_attachment_ord'].value_counts())
        print(ordinal_data.groupby('classes')['gill_spacing_ord'].value_counts())
        print(ordinal_data.groupby('classes')['ring_number_ord'].value_counts())

    def gill_attachment_pivot_table(self, nominal_features):
        """
          generate pivot table for feature gill_attachment.
          """
        new_data = self.create_nominal_df_ord(nominal_features)
        ordinal_data = (self.ordinal())[['gill_spacing_ord', 'ring_number_ord']].copy()
        # pivot table with numerical features -
        print('pivot table with numerical features -')
        data = pd.DataFrame()
        data['global_address'] = self.data['global_address'].copy()
        data['odor'] = self.data['odor'].copy()
        data['population'] = self.data['population'].copy()
        data['gill_attachment'] = self.data['gill_attachment'].copy()
        platform_numerical = pd.pivot_table(data, columns="gill_attachment", aggfunc=np.mean)
        print(platform_numerical.head(), '\n')

        # pivot table with nominal features -
        new_data['gill_attachment'] = self.data['gill_attachment'].copy()
        # e=1, p=0
        new_data['classes_bin'] = nominal_features['classes_bin']
        print('pivot table with nominal features -')
        platform_nominal = pd.pivot_table(new_data, columns="gill_attachment", aggfunc=lambda x: x.value_counts()[:1].sort_values(ascending=False).index)
        print(platform_nominal.head(6), '\n')

        # pivot table with ordinal features -
        print('pivot table with ordinal features -')
        ordinal_data['gill_attachment'] = self.data['gill_attachment'].copy()
        platform_ordinal = pd.pivot_table(ordinal_data, columns="gill_attachment", aggfunc=lambda x: x.value_counts()[:1].sort_values(ascending=False).index)
        print(platform_ordinal.head(), '\n')

    def exploratory_data_analysis(self):
        """
            explore features graphs and features relationship.
            """
        # create new feature based on Latitude and Longitude -
        self.data['global_address'] = self.data['latitude'] / self.data['longitude']

        self.feature_graphs()
        nominal_features = self.nominal()

        # explore features relations -
        self.features_relation(nominal_features)

        # feature classes -
        # explore classes (edible, poisonous) relations with other features we saw high correlation with -
        self.classes_relations(nominal_features)

        # pivot table for classes feature -
        self.classes_pivot_table(nominal_features)

        # feature population -
        self.population_relations(nominal_features)

        # feature gill attachment -
        self.gill_attachment_relations(nominal_features)

        # pivot table for gill attachment feature -
        self.gill_attachment_pivot_table(nominal_features)

########################################################################################################################
# exploratory_data_analysis -


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
