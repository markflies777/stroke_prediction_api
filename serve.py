import pickle
import pandas as pd


def prediction():
    loaded_model = pickle.load(open("stroke_prediction_model.pkl", "rb"))

    def stroke_predictor(df, model=loaded_model):
        df.gender.replace({'Male': 0,
                           'Female': 1,
                           'Other': 2}, inplace=True)

        df.Residence_type.replace({'Rural': 0,
                                   'Urban': 1}, inplace=True)

        df.work_type.replace({'Private': 0,
                              'SelfEmployed': 1,
                              'GovtJob': 2,
                              'Children': 3,
                              'NeverWorked': 4},
                             inplace=True)

        df.smoking_status.replace({'NeverSmoked': 0,
                                   'Unknown': 1,
                                   'FormerlySmoked': 2,
                                   'Smokes': 3}, inplace=True)

        df.ever_married.replace({'Yes': 1, 'No': 0}, inplace=True)

        X = df.to_numpy()[0].reshape(1, -1)
        result = loaded_model.predict_proba(X)[0][1] * 100

        return result

    return stroke_predictor

