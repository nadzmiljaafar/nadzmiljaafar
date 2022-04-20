# MODULE
import pandas as pd
import numpy as np
import random as rnd
import joblib
import otomkdir


def predictSalary(model,test_data):

    # setup test columns to make sure matching with trained one-hot-encoded data
    cols = ['house_rental_fee', 'house_loan_pmt', 'transport_spending', 'public_transport_spending', 'house_utility', 'food_spending', 'kids_spending', 'personal_loan', 'education_loan', 'other_loan', 'investment', 'race_Kree', 'race_Others', 'race_Sapiens', 'race_Skrull', 'gender_F', 'gender_M', 'employment_Employed', 'employment_Government retiree', 'employment_Others', 'employment_Private sector retiree', 'employment_Self-employed', 'employment_Unemployed', "education_Bachelor's Degree", 'education_Certificates', 'education_Diploma', 'education_High School', 'married_Divorcee', 'married_No', 'married_Yes', 'house_type_Own house - bungalows, private lots', 'house_type_Own house - condominiums', 'house_type_Own house - double storey terrace', 'house_type_Own house - flat', 'house_type_Own house - kampung / wooden house', 'house_type_Own house - one storey terrace', "house_type_Parent's House", 'house_type_Rental House  - bungalows, private lots', 'house_type_Rental house - double storey terrace', 'house_type_Rental house - flat', 'house_type_Rental house - kampung / wooden house', 'house_type_Rental house - single storey', 'house_type_Rental house- condominiums', 'vehicle_Asia brand car', 'vehicle_Did not own any vehicle', 'vehicle_Europe brand car', 'vehicle_Local brand car', 'vehicle_Motorcycle']
    df = pd.DataFrame(columns=cols)

    # clean test dataset
    cols_drop = ['house_value','age','person_living_in_house','transport_use','salary']
    for cols in cols_drop:
        test_data.drop([cols],axis=1, inplace = True)

    # perform one-hot-encoded on test data  
    cat_columns = ['race','gender','employment', 'education', 'married', 'house_type','vehicle']
    test_data_mod = pd.get_dummies(test_data, columns = cat_columns)
    
    df = df.drop( df.index.to_list()[0:] ,axis = 0 )
    test_data_mod = pd.concat([df, test_data_mod], axis=0)
    test_data_mod = test_data_mod.fillna(0)
    
    # read dataset rows by rows and predict the income group
    prediction = {}
    prediction_dict = []

    for i in range(len(test_data_mod)) :
        dataset = test_data_mod.iloc[i,:]
        dataset = dataset.tolist()
        prediction_dict.append(model.predict([dataset]))
    
    # export result as dataframe
    prediction['Prediction'] = prediction_dict
    maindf= pd.DataFrame.from_dict( prediction)
    maindf = pd.concat([maindf, test_data], axis=1)
    return maindf

print('hi')
    

