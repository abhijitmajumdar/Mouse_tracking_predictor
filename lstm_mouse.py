import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint

# Algorithm parameters
dataset_filename = 'mouse_data.txt'
weights_save_name = "weights_lstm_mouse.hdf5"
CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF=True
timesteps = 16
n_predictions = 64
MAX_X,MAX_Y = 1024,1024

# Global variables
img = []
dataset = []
testing_dataset = []
model = None
model_generated = False
state = 'idle'

# Save and Load dataset from file
def save_dataset():
    with open(dataset_filename, "wb") as fp:
        pickle.dump(dataset, fp)
#
def load_dataset():
    global dataset
    with open(dataset_filename, "rb") as fp:
        dataset = pickle.load(fp)

# Normalize and de-normalize
def normalize(data_points,scale_down=True):
    if(scale_down==True):
        data_points[:] = [[float(data[0])/MAX_X, float(data[1])/MAX_Y] for data in data_points]
    else:
        data_points[:] = [[int(data[0]*MAX_X), int(data[1]*MAX_Y)] for data in data_points]

# Mouse callback function
# The predictive model is in here, which is activated only when the corresponding state is enabled.
def trace_mouse_movements(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        if(state == 'record_data'):
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            dataset.append([x,y])
        elif(state == 'predict'):
            global model
            global img
            img.fill(0)
            testing_dataset.append([x,y])
            for [_x,_y] in testing_dataset:
                cv2.circle(img,(_x,_y),2,(0,127,255),-1)
            if(len(testing_dataset)>timesteps):
                testing_dataset.pop(0)
                pred_dataset = [i for i in testing_dataset]
                normalize(pred_dataset,scale_down=True)
                for i in range(n_predictions):
                    input_sequence = np.array([j for j in pred_dataset[i:i+timesteps]])
                    input_sequence = np.reshape(input_sequence,(1,input_sequence.shape[0],input_sequence.shape[1]))
                    _temp = model.predict(input_sequence)
                    pred_dataset.append([_temp[0][0],_temp[0][1]])
                normalize(pred_dataset,scale_down=False)
                for [_x,_y] in pred_dataset[timesteps:]:
                    cv2.circle(img,(_x,_y),2,(255,0,0),-1)
        elif(state == 'idle'):
            cv2.circle(img,(x,y),2,(int(np.random.randint(0,255)),int(np.random.randint(0,255)),int(np.random.randint(0,255))),-1)

# Load weights from file if file exists
def load_pretrained_weights():
    global model
    try:
        model.load_weights(weights_save_name)
    except:
        print('Pre-trained weights do not exist. Please train model to obtain weights')

# Graph the model. Edit the model here if desired
def generate_model(load_weights=False):
    global model
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(timesteps,2), implementation=2))
    model.add(Dense(2, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    if(load_weights==True): load_pretrained_weights()
    model_generated = True

# Generator for data
def initialize_data_xy():
    load_dataset()
    normalize(dataset,scale_down=True)
    xy_coord = [i for i in dataset]
    n = len(xy_coord)-1
    while(True):
        i = np.random.randint(0,n)
        try:
            x_data = xy_coord[i:i+timesteps]
            y_data = xy_coord[i+timesteps]
            yield x_data, y_data
        except:
            pass

# Function to fetch a dataset
def get_data_to_train(how_many):
    data = initialize_data_xy()
    input_data = []
    output_data = []
    for i in range(how_many):
        _temp = data.next()
        input_data.append(_temp[0])
        output_data.append(_temp[1])
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    return input_data,output_data

# Training...
def train(load_weights=False):
    global model
    x_train, y_train = get_data_to_train(6000)
    if(model_generated==False):
        generate_model(load_weights=load_weights)
    elif(load_weights==True):
        global model
        model.load_weights(weights_save_name)
    callbacks_list = [ModelCheckpoint(weights_save_name, monitor='loss', verbose=1, save_best_only=True, mode='auto', save_weights_only='True')]
    model.fit(x_train, y_train, batch_size=6000, epochs=5000, verbose=2, callbacks=callbacks_list)

# Helper function
def print_help():
    print('''Predict mouse movements using Long Short Term Memory(LSTM) Network
    Usage:
    Press \'i\' for idle mode (mouse points are plotted in different colours)
    Press \'r\' to start recording data, press \'r\' again to save dataset (mouse points are plotted red)
    Press \'t\' to start training using the dataset (mouse are not plotted)
    Press \'p\' to predict mouse movements (actual movements are plotted orange, predictions are plotted blue)
    Press \'c\' to clear screen
    Press \'h\' to show this help script
    Press \'ESC\' to quit
    Notes:
    - Predictions are not real. The algorithm cannot figure out what you are going to do next. It is just a
      demo to show how LSTM predictions can be used
    - An example dataset file is already provided, along with trained model weights. So you can directly start
      training and/or prediction.''')

def main():
    global img
    global state
    img = np.zeros((MAX_X,MAX_Y,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',trace_mouse_movements)
    print_help()
    while(1):
        cv2.imshow('image',img)
        ip = cv2.waitKey(20) & 0xFF
        if ip == 27:
            break
        elif(ip==ord('r')):
            if(state != 'record_data'):
                state = 'record_data'
                print('Started recording data')
            else:
                save_dataset()
                print('Saved recorded data')
        elif(ip==ord('p')):
            global testing_dataset
            if(model_generated==True):
                load_pretrained_weights()
            else:
                generate_model(load_weights=True)
            testing_dataset = []
            state = 'predict'
        elif(ip==ord('t')):
            state = 'train'
            print('Training started')
            train(load_weights=CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF)
            print('Training done')
            state = 'idle'
        elif(ip==ord('i')):
            state = 'idle'
        elif(ip==ord('c')):
            img.fill(0)
        elif(ip==ord('h')):
            print_help()
    cv2.destroyAllWindows()

main()
