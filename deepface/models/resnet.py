import keras

def Resnet50(weights_path, output_layers=[]):
    m = keras.models.load_model(weights_path)
    if len(output_layers) == 0:
        return m
    m = keras.models.Model(inputs=m.inputs,
                           outputs=[m.get_layer(lname).output for lname in output_layers])
    return m
