# MODULE
import predict_salary as ps
import pandas as pd
import joblib
import otomkdir
import warnings
warnings.filterwarnings("ignore")

# will create a folder named 'Nadzmil' in user desktop
# change accordingly as required
mypath = otomkdir.otomkdir.auto_create_folder(folder_extend='Nadzmil')

'''
info for predictSalary()

    test_data : dataset to test the model. Must in dataframe/csv with the same data structure of raw data.
    model : model name (.sav)

    this function return the results of the prediction in a dataframe format.

'''
test_data_name = '\submit.csv'
model_name = '\predictor_model.sav'
test_data = pd.read_csv(str(mypath) + test_data_name) 
model = joblib.load(str(mypath) + model_name)

# function
result = ps.predictSalary(model,test_data)

# export result to csv
result.to_csv(str(mypath) + '\prediction.csv',index=False)

print(result)

print('hi')