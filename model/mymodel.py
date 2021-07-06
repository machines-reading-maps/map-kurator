import keras                              
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Flatten , Activation
from keras.layers import Conv2D, MaxPooling2D    
from keras import backend as K                   
from keras.callbacks import Callback             
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose 
from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape 
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy                           
from keras.models import Model        
from keras.layers import Lambda,TimeDistributed
from keras import layers

def UNET(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1-1')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1-2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2-1')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2-2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3-1')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3-2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4-1')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4-2')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5-1')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5-2')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6-1')(pool5)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6-2')(conv6)
    drop6 = Dropout(0.5)(conv6)
    
    #conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7-1')(pool6)
    #conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7-2')(conv7)
    #drop7 = Dropout(0.5)(conv7)
    
    #//////////////////////////////////////////////////////////

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6u-0')(UpSampling2D(size = (2,2))(conv6))
    #merge6 = concatenate([drop4,up6], axis = 3)
    conv6u = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6u-1')(up6)
    conv6u = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6u-2')(conv6u)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7u-0')(UpSampling2D(size = (2,2))(conv6u))
    #merge7 = concatenate([conv3,up7], axis = 3)
    conv7u = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7u-1')(up7)
    conv7u = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7u-2')(conv7u)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8u-0')(UpSampling2D(size = (2,2))(conv7u))
    #merge8 = concatenate([conv2,up8], axis = 3)
    conv8u = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8u-1')(up8)
    conv8u = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8u-2')(conv8u)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9u-0')(UpSampling2D(size = (2,2))(conv8u))
    #merge9 = concatenate([conv1,up9], axis = 3)
    conv9u = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9u-1')(up9)
    conv9u = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9u-2')(conv9u)
    
    
    up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10u-0')(UpSampling2D(size = (2,2))(conv9u))
    #merge9 = concatenate([conv1,up9], axis = 3)
    conv10u = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10u-1')(up10)
    conv10u = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10u-2')(conv10u)    
    conv10u = Conv2D(3, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10u-3')(conv10u)

    model = Model(inputs, conv10u)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def model_U_VGG():
    #input_shape = (720, 1280, 3)
    #input_shape = (512,512,3)
    input_shape = (None,None,3)
    inputs = Input(shape=input_shape, name='input') 


    # Block 1
    x0 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x0 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x0)
    x0 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x0)

    # Block 2
    x1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x0)
    x1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x1)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x1)

    # Block 3
    x2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x1)
    x2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x2)
    x2_take = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x2)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x2_take)

    # Block 4
    x3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x2)
    x3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x3)
    x3_take = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x3)
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x3_take)

    # Block 5
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x3)
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x4)
    x4_take = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x4)
    x4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x4_take)

    #f1 = UpSampling2D((2,2))(x4) 
    #if TASK_4:
    #    f1 = ZeroPadding2D(padding=((1,0), (0,0)), name = 'f1')(f1)
    f1 = x4_take
    f2 = x3
    h1 = Concatenate()([f2, f1])
    h1 = layers.Conv2D(128, (1, 1),
                      activation='relu',
                      padding='same',
                      name='up1_1')(h1)

    h1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up1_2')(h1)


    h2 = Concatenate()([x2, UpSampling2D((2,2))(h1)])
    h2 = layers.Conv2D(64, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up2_1')(h2)
    h2 = layers.Conv2D(64, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up2_2')(h2)

    h3 = Concatenate()([x1, UpSampling2D((2,2))(h2)])
    h3 = layers.Conv2D(32, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up3_1')(h3)
    h3 = layers.Conv2D(32, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up3_2')(h3)

    h4 = Concatenate()([x0, UpSampling2D((2,2))(h3)])
    h4 = layers.Conv2D(32, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up4_1')(h4)
    h4 = layers.Conv2D(32, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up4_2')(h4)

    h5 =  Concatenate()([inputs, UpSampling2D((2,2))(h4)])
    h5 = layers.Conv2D(16, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up5_1')(h5)
    ################## output for TEXT/NON-TEXT ############
    
    o1 = layers.Conv2D(3, (3,3),
                    activation = 'softmax',
                    padding = 'same',
                    name = 'up5_2')(h5)
    
    ################ Regression ###########################
    b1 = Concatenate(name = 'agg_feat-1')([x4_take, h1]) # block_conv3, up1_2 # 32,32,630
    b1 = layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', 
                                activation = 'relu', name = 'agg_feat-2')(b1) # 64,64,128
   
    #------ xy regression -------
    o2 = layers.Conv2DTranspose(64, (3,3), strides = (2,2),padding = 'same',
                                activation = 'relu', name = 'regress-1-1')(b1) # 128,128, 32
    o2 = layers.Conv2DTranspose(32, (3,3), strides = (1,1),padding = 'same',
                                activation = 'relu', name = 'regress-1-2')(o2) # 128,128, 32
    o2 = layers.Conv2DTranspose(16, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu',name = 'regress-1-3')(o2) # 256,256, 8
    o2 = layers.Conv2DTranspose(8, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'relu',name = 'regress-1-4')(o2) # 256,256, 8
    o2 = layers.Conv2DTranspose(4, (3,3),strides = (2,2), padding = 'same',
                                activation = 'relu', name = 'regress-1-5')(o2) # 512,512, 2
    o2 = layers.Conv2DTranspose(2, (3,3),strides = (1,1), padding = 'same',
                                activation = 'tanh', name = 'regress-1-6')(o2) # 512,512, 2
    
    #------ wh regression -------
    o4 = layers.Conv2DTranspose(64, (3,3), strides = (2,2),padding = 'same', 
                                activation = 'relu',name = 'regress-3-1')(b1) # 128,128, 32
    o4 = layers.Conv2DTranspose(32, (3,3), strides = (1,1),padding = 'same', 
                                activation = 'relu',name = 'regress-3-2')(o4) # 128,128, 32
    o4 = layers.Conv2DTranspose(16, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu', name = 'regress-3-3')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(8, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'relu', name = 'regress-3-4')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(4, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu', name = 'regress-3-5')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(2, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'sigmoid',name = 'regress-3-6')(o4) # 512,512, 2
    
    # ------ sin/cos regression -------
    b2 = Concatenate()([x3_take, b1]) # block4_conv3, agg_feat-2 # 64,64,630
    b2 = layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', 
                                activation = 'relu', name = 'regress-2-1')(b2) # 128, 128, 128
    o3 = Concatenate()([x2_take, b2 ]) # block3_conv3, agg_feat-3 # 128, 128, (256+128)
    o3 = layers.Conv2DTranspose(32, (3,3),strides = (2,2),padding = 'same',
                                activation = 'relu', name = 'regress-2-2')(o3) # 256,256, 32
    o3 = layers.Conv2DTranspose(2, (3,3),strides = (2,2),padding = 'same',
                                activation = 'tanh', name = 'regress-2-3')(o3) # 512,512,2
    

    model =  Model(inputs, [o1,o2,o3,o4], name = 'U-VGG-model')
    
    return model



def model_U_VGG_Centerline():
    #input_shape = (720, 1280, 3)
    #input_shape = (512,512,3)
    input_shape = (None,None,3)
    inputs = Input(shape=input_shape, name='input') 


    # Block 1
    x0 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x0 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x0)
    x0 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x0)

    # Block 2
    x1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x0)
    x1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x1)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x1)

    # Block 3
    x2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x1)
    x2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x2)
    x2_take = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x2)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x2_take)

    # Block 4
    x3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x2)
    x3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x3)
    x3_take = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x3)
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x3_take)

    # Block 5
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x3)
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x4)
    x4_take = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x4)
    x4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x4_take)

    #f1 = UpSampling2D((2,2))(x4) 
    #if TASK_4:
    #    f1 = ZeroPadding2D(padding=((1,0), (0,0)), name = 'f1')(f1)
    f1 = x4_take
    f2 = x3
    h1 = Concatenate()([f2, f1])
    h1 = layers.Conv2D(128, (1, 1),
                      activation='relu',
                      padding='same',
                      name='up1_1')(h1)

    h1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up1_2')(h1)


    h2 = Concatenate()([x2, UpSampling2D((2,2))(h1)])
    h2 = layers.Conv2D(64, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up2_1')(h2)
    h2 = layers.Conv2D(64, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up2_2')(h2)

    h3 = Concatenate()([x1, UpSampling2D((2,2))(h2)])
    h3 = layers.Conv2D(32, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up3_1')(h3)
    h3 = layers.Conv2D(32, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up3_2')(h3)

    h4_take = Concatenate()([x0, UpSampling2D((2,2))(h3)])

    h4 = layers.Conv2D(32, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up4_1')(h4_take)
    h4 = layers.Conv2D(32, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up4_2')(h4)

    h5 =  Concatenate()([inputs, UpSampling2D((2,2))(h4)])
    h5 = layers.Conv2D(16, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up5_1')(h5)
    ################## output for TEXT/NON-TEXT ############
    
    o1 = layers.Conv2D(3, (3,3),
                    activation = 'softmax',
                    padding = 'same',
                    name = 'up5_2')(h5)
    ################## output for centerline /other ###########
    h41 = layers.Conv2D(32, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up41_1')(h4_take)
    h41 = layers.Conv2D(32, (3,3),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up41_2')(h41)

    h51 =  Concatenate()([inputs, UpSampling2D((2,2))(h41)])
    h51 = layers.Conv2D(16, (1,1),
                    activation = 'relu',
                    padding = 'same',
                    name = 'up51_1')(h51)
    
    o11 = layers.Conv2D(2, (3,3),
                    activation = 'softmax',
                    padding = 'same',
                    name = 'up51_2')(h51)
    
    ################ Regression ###########################
    b1 = Concatenate(name = 'agg_feat-1')([x4_take, h1]) # block_conv3, up1_2 # 32,32,630
    b1 = layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', 
                                activation = 'relu', name = 'agg_feat-2')(b1) # 64,64,128
   
    #------ xy regression -------
    o2 = layers.Conv2DTranspose(64, (3,3), strides = (2,2),padding = 'same',
                                activation = 'relu', name = 'regress-1-1')(b1) # 128,128, 32
    o2 = layers.Conv2DTranspose(32, (3,3), strides = (1,1),padding = 'same',
                                activation = 'relu', name = 'regress-1-2')(o2) # 128,128, 32
    o2 = layers.Conv2DTranspose(16, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu',name = 'regress-1-3')(o2) # 256,256, 8
    o2 = layers.Conv2DTranspose(8, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'relu',name = 'regress-1-4')(o2) # 256,256, 8
    o2 = layers.Conv2DTranspose(4, (3,3),strides = (2,2), padding = 'same',
                                activation = 'relu', name = 'regress-1-5')(o2) # 512,512, 2
    o2 = layers.Conv2DTranspose(2, (3,3),strides = (1,1), padding = 'same',
                                activation = 'tanh', name = 'regress-1-6')(o2) # 512,512, 2
    
    #------ wh regression -------
    o4 = layers.Conv2DTranspose(64, (3,3), strides = (2,2),padding = 'same', 
                                activation = 'relu',name = 'regress-3-1')(b1) # 128,128, 32
    o4 = layers.Conv2DTranspose(32, (3,3), strides = (1,1),padding = 'same', 
                                activation = 'relu',name = 'regress-3-2')(o4) # 128,128, 32
    o4 = layers.Conv2DTranspose(16, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu', name = 'regress-3-3')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(8, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'relu', name = 'regress-3-4')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(4, (3,3),strides = (2,2), padding = 'same', 
                                activation = 'relu', name = 'regress-3-5')(o4) # 256,256, 8
    o4 = layers.Conv2DTranspose(2, (3,3),strides = (1,1), padding = 'same', 
                                activation = 'sigmoid',name = 'regress-3-6')(o4) # 512,512, 2
    
    # ------ sin/cos regression -------
    b2 = Concatenate()([x3_take, b1]) # block4_conv3, agg_feat-2 # 64,64,630
    b2 = layers.Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', 
                                activation = 'relu', name = 'regress-2-1')(b2) # 128, 128, 128
    o3 = Concatenate()([x2_take, b2 ]) # block3_conv3, agg_feat-3 # 128, 128, (256+128)
    o3 = layers.Conv2DTranspose(32, (3,3),strides = (2,2),padding = 'same',
                                activation = 'relu', name = 'regress-2-2')(o3) # 256,256, 32
    o3 = layers.Conv2DTranspose(2, (3,3),strides = (2,2),padding = 'same',
                                activation = 'tanh', name = 'regress-2-3')(o3) # 512,512,2



    #o1: t/nt, o11:centerline, o2:x,y, o3:sin,cos, o4:bounding box width,height
    model =  Model(inputs, [o1,o11, o2,o3,o4], name = 'U-VGG-model')

    
    return model


def model_U_VGG_Centerline_Localheight():
    # input_shape = (720, 1280, 3)
    # input_shape = (512,512,3)
    input_shape = (None, None, 3)
    inputs = Input(shape=input_shape, name='input')

    # Block 1
    x0 = layers.Conv2D(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv1')(inputs)
    x0 = layers.Conv2D(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block1_conv2')(x0)
    x0 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x0)

    # Block 2
    x1 = layers.Conv2D(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv1')(x0)
    x1 = layers.Conv2D(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block2_conv2')(x1)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x1)

    # Block 3
    x2 = layers.Conv2D(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv1')(x1)
    x2 = layers.Conv2D(256, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block3_conv2')(x2)
    x2_take = layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3')(x2)
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x2_take)

    # Block 4
    x3 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x2)
    x3 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(x3)
    x3_take = layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block4_conv3')(x3)
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x3_take)

    # Block 5
    x4 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x3)
    x4 = layers.Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(x4)
    x4_take = layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block5_conv3')(x4)
    x4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x4_take)

    # f1 = UpSampling2D((2,2))(x4)
    # if TASK_4:
    #    f1 = ZeroPadding2D(padding=((1,0), (0,0)), name = 'f1')(f1)
    f1 = x4_take
    f2 = x3
    h1 = Concatenate()([f2, f1])
    h1 = layers.Conv2D(128, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up1_1')(h1)

    h1 = layers.Conv2D(128, (3, 3),
                       activation='relu',
                       padding='same',
                       name='up1_2')(h1)

    h2 = Concatenate()([x2, UpSampling2D((2, 2))(h1)])
    h2 = layers.Conv2D(64, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up2_1')(h2)
    h2 = layers.Conv2D(64, (3, 3),
                       activation='relu',
                       padding='same',
                       name='up2_2')(h2)

    h3 = Concatenate()([x1, UpSampling2D((2, 2))(h2)])
    h3 = layers.Conv2D(32, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up3_1')(h3)
    h3 = layers.Conv2D(32, (3, 3),
                       activation='relu',
                       padding='same',
                       name='up3_2')(h3)

    h4_take = Concatenate()([x0, UpSampling2D((2, 2))(h3)])

    h4 = layers.Conv2D(32, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up4_1')(h4_take)
    h4 = layers.Conv2D(32, (3, 3),
                       activation='relu',
                       padding='same',
                       name='up4_2')(h4)

    h5 = Concatenate()([inputs, UpSampling2D((2, 2))(h4)])
    h5 = layers.Conv2D(16, (1, 1),
                       activation='relu',
                       padding='same',
                       name='up5_1')(h5)
    ################## output for TEXT/NON-TEXT ############

    o1 = layers.Conv2D(3, (3, 3),
                       activation='softmax',
                       padding='same',
                       name='up5_2')(h5)
    ################## output for centerline /other ###########
    h41 = layers.Conv2D(32, (1, 1),
                        activation='relu',
                        padding='same',
                        name='up41_1')(h4_take)
    h41 = layers.Conv2D(32, (3, 3),
                        activation='relu',
                        padding='same',
                        name='up41_2')(h41)

    h51 = Concatenate()([inputs, UpSampling2D((2, 2))(h41)])
    h51 = layers.Conv2D(16, (1, 1),
                        activation='relu',
                        padding='same',
                        name='up51_1')(h51)

    o11 = layers.Conv2D(2, (3, 3),
                        activation='softmax',
                        padding='same',
                        name='up51_2')(h51)

    ################ Regression ###########################
    b1 = Concatenate(name='agg_feat-1')([x4_take, h1])  # block_conv3, up1_2 # 32,32,630
    b1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='agg_feat-2')(b1)  # 64,64,128

    # ------ xy regression -------
    o2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-1-1')(b1)  # 128,128, 32
    o2 = layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-1-2')(o2)  # 128,128, 32
    o2 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-1-3')(o2)  # 256,256, 8
    o2 = layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-1-4')(o2)  # 256,256, 8
    o2 = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-1-5')(o2)  # 512,512, 2
    o2 = layers.Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same',
                                activation='tanh', name='regress-1-6')(o2)  # 512,512, 2

    # ------ wh regression -------
    o4 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-3-1')(b1)  # 128,128, 32
    o4 = layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-3-2')(o4)  # 128,128, 32
    o4 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-3-3')(o4)  # 256,256, 8
    o4 = layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-3-4')(o4)  # 256,256, 8
    o4 = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-3-5')(o4)  # 256,256, 8
    o4 = layers.Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same',
                                activation='sigmoid', name='regress-3-6')(o4)  # 512,512, 2

    # ------ sin/cos regression -------
    b2 = Concatenate()([x3_take, b1])  # block4_conv3, agg_feat-2 # 64,64,630
    b2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-2-1')(b2)  # 128, 128, 128
    o3 = Concatenate()([x2_take, b2])  # block3_conv3, agg_feat-3 # 128, 128, (256+128)
    o3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-2-2')(o3)  # 256,256, 32
    o3 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2), padding='same',
                                activation='tanh', name='regress-2-3')(o3)  # 512,512,2

    # ------ local height regression ------
    o5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-4-1')(b1)  # 128,128, 32
    o5 = layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-4-2')(o5)  # 128,128, 32
    o5 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-4-3')(o5)  # 256,256, 8
    o5 = layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-4-4')(o5)  # 256,256, 8
    o5 = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same',
                                activation='relu', name='regress-4-5')(o5)  # 256,256, 8
    o5 = layers.Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-4-6')(o5)  # 512,512, 2
    o5 = layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='regress-4-7')(o5)  # 512,512, 1

    # o1: t/nt, o11:centerline, o2:x,y, o3:sin,cos, o4:bounding box width,height, o5:localheight
    # model =  Model(inputs, [o1,o11, o2,o3,o4], name = 'U-VGG-model')
    model = Model(inputs, [o1, o11, o5], name='U-VGG-model-Localheight')

    return model

