# Visualising Layers
By visualising weights and outputs for a given layer we can gain insight about how each layer contribute to the model.
## Convolution layers
The model has convolution layers with variable kernel size, mainly used for downsampling and upsampling at both ends of the model.
### Filters
We can visualise the filters of the CNN layers. Each layer has a specific kernel size(think it as a 1-D window for our case), these kernels slide through the data (overlapping some points) looking for features in these windows and training through back-propagation. We can plot these filters and see what each "convolution layer window" look like.

    from keras import backend as K
    from keras.models import load_model, Model
    from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import h5py

    model = load_model('./test_trainer_EQT_outputs/final_model.h5',
                                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                                       'FeedForward': FeedForward,
                                                       'LayerNormalization': LayerNormalization,
                                                       'f1': f1})

    conv = []
    for i in range(len(model.layers)):
        if 'conv' not in model.layers[i].name:
            continue
        filters, biases = model.layers[i].get_weights()
        conv.append(i)


    for i in conv:
        filters, biases = model.layers[i].get_weights()
        for i in range(len(filters)):
            for j in range(len(filters[i])):
                plt.plot(filters[i][j])
                plt.show()



### Feature Maps
Feature maps are a way to understand what each layers does to an input when fed into a trained model.

We do this by creating a new model for each convolution layer, making that layer the output layer. We use weights from the trained model.

    import numpy as np
    import h5py
    import os


    hf = h5py.File('micro_merged.hdf5', 'r')
    ev1 = np.array(hf['data']['AAM3.ZQ_20080515154616_EV'])
    ev2 = np.array(hf['data']['112A.TA_20080211100400_EV'])
    ev3 = np.array(hf['data']['B045.PB_20180212205945_EV'])
    ev4 = np.array(hf['data']['109C.TA_20070905204756_EV'])
    n1 = np.array(hf['data']['B039.PB_201404291736_NO'])
    n2 = np.array(hf['data']['ANON.AV_20180116200442_NO'])
    n3 = np.array(hf['data']['ACTO.PO_201205081247_NO'])
    
    pred_dict = {'ev1':ev1,'ev2':ev2,'ev3':ev3,"ev4":ev4,'n1':n1,'n2':n2,'n3':n3}

    for i in conv:
        model_vis = Model(inputs=model.inputs, outputs=model.layers[i].output)
        for data in pred_dict.keys():
            data_reshaped= pred_dict[data].reshape(1,6000,3)
            prediction = model_vis.predict(data_reshaped)
            np.save('./predictions/{}_conv{}'.format(data,i),prediction) 
   
    pred_list = sorted(os.listdir("./predictions"))
    
    for i in pred_list:
        if "conv" in i:
            if i.endswith(".npy"):
                plt.plot()
                predicted = np.load("./predictions/{}".format(i))
                predicted_reshaped = predicted.reshape((predicted.shape[1],predicted.shape[2]))
                plt.plot(predicted_reshaped)
                plt.title(i[:-4])
                plt.show()

## Attention Layers
There are 4 attention layers in the model. 2 global attentions in the main branch and two local attentions afterwards, for P and S picking(one each).
### Weights
Attention weights can be interpreted as how related a time step is to the label. These help us understand where the models' "attention" is at.

    attention = []
    for i in range(len(model.layers)):
        if 'attention' not in model.layers[i].name:
            continue
        attention.append(i)

    weights = model.get_weights()
    
    d0 = weights[attention[0]]
    d = weights[attention[1]]
    p = weights[attention[2]]
    s = weights[attention[3]]
    
    attention_dict = {"d0":d0,"d":d,"p":p,"s":s}
    
    for i in attention_dict.keys():
        for j in range(len(attention_dict[i])):
            plt.plot(attention_dict[i][j])
            plt.title(i)
            plt.show()

### Feature Maps
We can also create a feature map using the same method we used for the CNN blocks. Using the weights from the trained model, feeding an input and taking outputs from the layers we are interested in.

    for i in attention:
        model_vis = Model(inputs=model.inputs, outputs=model.layers[i].output)
        for data in pred_dict.keys():
            data_reshaped= pred_dict[data].reshape(1,6000,3)
            prediction = model_vis.predict(data_reshaped)
            np.save('./predictions/{}_attention{}'.format(data,i),np.dstack(prediction)) 
    
    pred_list = sorted(os.listdir("./predictions"))
    
    for i in pred_list:
        if "attention" in i:
            if i.endswith(".npy"):
                plt.plot()
                predicted = np.load("./predictions/{}".format(i))
                predicted_reshaped = predicted.reshape((predicted.shape[1],predicted.shape[2]))
                plt.plot(predicted_reshaped)
                plt.title(i[:-4])
                plt.show()

