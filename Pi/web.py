# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
lag = 8

def load_models():
    lstm = load_model('../model/lstm8.h5')
    gru = load_model('../model/gru8.h5')
    saes = load_model('../model/saes.h5')
    
def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'COUNT'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
    for i in range (lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range (lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

@app.route("/predict", methods=["POST"])
def predict():
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
        # Filter csv files accordingly
        train = train[(train['SCATS'] == my_scats) & (train['DAY'] == my_day) 
        & (train['TIME'] >= st) & (train['TIME'] <= et) & (train[d] > 0)]
        # Test set:
        test = test[(test['SCATS'] == my_scats) & (test['DAY'] == my_day) 
        & (test['TIME'] >= st) & (test['TIME'] <= et) & (test[d] > 0)]
        
        # Save files to csv
        test.to_csv('./data/my_test.csv', encoding='utf-8', index=False)
        train.to_csv('./data/my_train.csv', encoding='utf-8', index=False)
        my_test = './data/my_test.csv'
        my_train = './data/my_train.csv'
        
        # Process data
        _, _, X_test, y_test, scaler = process_data(my_train, my_test, lag)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
        
        # Reshape data
        y_preds = []
        for name, model in zip(names, models):
            if name == 'SAEs':
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        
        # Get predictions
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
       
        
        # Get average traffic count per hour
        time_range = int((et - st) / 100)
        count = 0
        for i in range(time_range):
            count += predicted[i]
        
        count_phour = count / time_range
        
        
        
        

	# return the data dictionary as a JSON response
	return flask.jsonify(count_phour)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras models and Flask starting server..."
		"please wait until server has fully started"))
	load_models()
	app.run()
    #app.run(host='0.0.0.0', port=8080)