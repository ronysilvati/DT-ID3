import math

'''
    Implementation of algorithm ID3 from Ross Quinlan.
    This implementation is based on example of website sefiks.com
    Developer: Rony Silva
    Email: rcs@ic.ufal.edu.br
    Github: github.com/ronysilvati
'''

'''Dataset (Column details): Day, Outlook, Temp., Humidity, Wind, Decision (Label)'''
dataSet = [
    [1, "Sunny", "Hot", "High", "Weak", "No"],
    [2, "Sunny", "Hot", "High", "Strong", "No"],
    [3, "Overcast", "Hot", "High", "Weak", "Yes"],
    [4, "Rain", "Mild", "High", "Weak", "Yes"],
    [5, "Rain", "Cool", "Normal", "Weak", "Yes"],
    [6, "Rain", "Cool", "Normal", "Strong", "No"],
    [7, "Overcast", "Cool", "Normal", "Strong", "Yes"],
    [8, "Sunny", "Mild", "High", "Weak", "No"],
    [9, "Sunny", "Cool", "Normal", "Weak", "Yes"],
    [10, "Rain", "Mild", "Normal", "Weak", "Yes"],
    [11, "Sunny", "Mild", "Normal", "Strong", "Yes"],
    [12, "Overcast", "Mild", "High", "Strong", "Yes"],
    [13, "Overcast", "Hot", "Normal", "Weak", "Yes"],
    [14, "Rain", "Mild", "High", "Strong", "No"],
]


'''Calculate a entropy to a column of a dataset'''
'''Ref: https://www.quora.com/What-is-meant-by-entropy-in-machine-learning-contexts'''


def entropy(dataset, col_collection):
    total_instances = len(dataset)
    entropy = 0

    for column in col_collection:
        try:
            p_value = (tt_inst_col(dataset, column['col_position'], column['value']) / total_instances)
            partial_result = -p_value * math.log(p_value, 2)
            entropy += partial_result

        except ValueError:
            entropy += 0

    return entropy


'''Return the total of instances that contains a value in a feature and that is labeled with a informed value'''


def tt_inst_col_and_lab_with(dataset, col_position, expected_value_col, col_position_label, expected_value_label):
    total = 0

    for instance in dataset:
        if (instance[col_position] == expected_value_col) and (instance[col_position_label] == expected_value_label):
            total += 1

    return total


'''Return the total of instances that contains a value in a column'''


def tt_inst_col(dataset, col_position, expected_value_col):
    total = 0

    for instance in dataset:
        if instance[col_position] == expected_value_col:
            total += 1

    return total


'''Return a list of possibles values in a column. If "add_only_this" is defined, only this value is returned'''


def get_possible_values_to_column(dataset, col_position, add_only_this=None):
    possible_values_list = []

    for instance in dataset:
        obj = {"col_position": col_position, "value": instance[col_position]}

        if(obj not in possible_values_list):
            if((add_only_this == None) or (add_only_this == instance[col_position])):
                possible_values_list.append(obj)

    return possible_values_list


'''Return the gain to one feature'''

def gain(dataset, col_collection, label_collection):
    gain = 0

    for column in col_collection:
        total_instances = tt_inst_col(dataset, column['col_position'], column['value'])

        for label in label_collection:
            total_instances_with_value_and_label = \
                tt_inst_col_and_lab_with(dataset, column['col_position'],
                                         column['value'], label['col_position'], label['value'])

            value_after_division = total_instances_with_value_and_label / total_instances

            try:
                gain += -value_after_division * math.log(value_after_division, 2)
            except ValueError:
                gain += 0

    return gain


'''This function calc the general gain to one feature'''


def calc_feature_gain(dataset, possible_values_feature, possible_values_labeled):
    total_instances = len(dataset)
    general_entropy = entropy(dataset, possible_values_labeled)
    col_gain = general_entropy

    for value in possible_values_feature:
        partial_gain = gain(dataset,
                            get_possible_values_to_column(dataSet, value['col_position'], value['value']),
                            possible_values_labeled)

        col_gain -= ((tt_inst_col(dataset, value['col_position'], value['value']) / total_instances) * partial_gain)

    return col_gain


print('--------------------------------------------')
print("Gain to Outlook feature:")
print(calc_feature_gain(dataSet, get_possible_values_to_column(dataSet, 1), get_possible_values_to_column(dataSet, 5)))
print('--------------------------------------------')
print("Gain to Temp. feature:")
print(calc_feature_gain(dataSet, get_possible_values_to_column(dataSet, 2), get_possible_values_to_column(dataSet, 5)))
print('--------------------------------------------')
print("Gain to Humidity. feature:")
print(calc_feature_gain(dataSet, get_possible_values_to_column(dataSet, 3), get_possible_values_to_column(dataSet, 5)))
print('--------------------------------------------')
print("Gain to Wind feature:")
print(calc_feature_gain(dataSet, get_possible_values_to_column(dataSet, 4), get_possible_values_to_column(dataSet, 5)))
print('--------------------------------------------')
