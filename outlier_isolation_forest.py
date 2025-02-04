from sklearn.ensemble import IsolationForest

def train_isolation_forest(data, intent):
    # train the detection model
    iso = IsolationForest(contamination=0.02)
    yhat = iso.fit_predict(data)
    data_new = data.copy()
    data_new['Predicted Outlier'] = yhat == -1
    data_new.intent = intent
    return data_new