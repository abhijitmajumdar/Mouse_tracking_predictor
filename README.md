
# Mouse_tracking_predictor
LSTM network to predict mouse movements - training, prediction and interactive dataset generation. This is just a demo to show how LSM can be used and by no means can accurately predict how the user is going to move the mouse. However it is interesting to note how the predictions tend to follow the direction in which the mouse is moving initially, and also diverge from the boundaries of the drawing area.

![Alt text](/Images/Mouse_tracker.gif?raw=true "Demo")

## How it works
A dataset is initially created by recording mouse movements within the drawing area. These data points with (x,y) coordinates are normalized before training LSTM network. The network expects 16(defined by 'timesteps') consecutive samples to predict the next (x,y) coordinate of a point. To train such a network, random index from the dataset is selected and the following 16 data-points are fed to the network as input. The 17th data-point is used as the ground truth (labeled) output, which is compared to the predicted output of the network to compute the error which is then used in back-propagation to update the weights in the network.

The prediction mode works iteratively. The last 16 user inputs are used to predict a new point. In the next iteration this new point along with the last 15 points are used to predict another point. This process continues 128(defined by 'n_predictions') times. Hence, initially the predictions tend to follow the direction of the mouse, but then diverge into some crazy shapes!

## Pre-requisites
- OpenCV
- Numpy
- Pickle
- Tensorflow
- Keras

## How to run
Tested on a Intel i7-7500U. Probably not optimized for GPU. Use following command:
    python lstm_mouse.py

## Dataset
Once the python script is running, you can press 'r' to start recording mouse positions into a dataset. During recording, the points are plotted in red. Pressing 'r' again will save this dataset into a file(using pickle) defined in the script (default: mouse_data.txt)
A sample dataset is provided(~6500 sample points). So you can train using that file.

## Training
To train the model using the dataset(either generated or using the provided dataset), press 't'. This will start the training process.
A pre-trained weight file is provided(weights_lstm_mouse.hdf5). The variable 'CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF' in the script can be configured to determine if one wants to continue to train on top of pre-trained weights.

## Prediction
To start prediction, press 'p'. The actual trail of the mouse position for 'timesteps' points are plotted in amber, while the predictions are plotted in blue. Since pre-trained weights are already provided, you can test out the algorithm without training.

## Misc
Press 'c' to clear the screen.
Press 'h' to print out a help script.
Press 'i' to go back to idle mode.
Press 'ESC' to quit the program.

## Changes
- New branch changes:
    - Changed the input shape from (batch,features,timesteps) to (batch,timesteps,features); complaint with keras LSTM layer inputs
    - This change uses less network parameters(~20k down to ~17k)
    - This algorithm however runs slower than before (both training and prediction). Since the shape is changed, the feed forward is different. Hence even though the parameters are less, more gates are now used(due to an actual timestep), which makes the feed-forward slower.
