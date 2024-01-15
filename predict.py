from fastapi import FastAPI, HTTPException, Request
import pickle
import json
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd

app = FastAPI()

# how to run the server:
# uvicorn predict:app --reload

# Load the pre-trained model, DictVectorizer and StandardScaler
with open('models_binary/catboost_model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)
with open('models_binary/label_encoder.pkl', 'rb') as f_bin:
    label_encoder = pickle.load(f_bin)
with open('models_binary/one_hot_encoder.pkl', 'rb') as f_bin:
    one_hot_encoder = pickle.load(f_bin)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    print("data", data)

    # Ensure the required fields are present
    required_fields = {"work_year",
                        "job_category",
                        "employee_residence",
                        "experience_level",
                        "work_setting",
                        "company_location",
                        "company_size"}

    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    df = pd.DataFrame([json.loads(data)])

    print("label encoder")
    df['work_year'] = label_encoder.transform(df['work_year'])

    print("one hot encoder")
    categorical_cols = ['job_category', 'employee_residence', 'experience_level', 
                     'work_setting', 'company_location', 'company_size']
    encoded_df_columns = one_hot_encoder.transform(df[categorical_cols]).toarray()
    encoded_df_col_names = one_hot_encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_df_columns, columns=encoded_df_col_names, index=df.index)
    df = pd.concat([df, encoded_df], axis=1)

    df.drop(columns=["job_category", "employee_residence", "experience_level", "work_setting", "company_location", "company_size"], axis=1, inplace=True )

    # Make a prediction
    print("prediction")
    prediction = np.expm1(model.predict(df))
  
    return jsonable_encoder({"prediction": prediction[0]})
    