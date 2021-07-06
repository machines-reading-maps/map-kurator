from keras import backend as K    

# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def mean_squared_error_mask(y_true, y_pred):
    y_mask = y_true[:,:,:,0] #bsize, h, w, 5(m,x,y,sin,cos)
    y_mask = K.expand_dims(y_mask, axis = -1)
    #print y_true.shape, y_pred.shape
    y_true = y_true[:,:,:,1:]
    #y_pred = y_pred[:,:,:,1:]
    return K.sum(K.square((y_pred - y_true)*y_mask),  axis= -1) # mse at each pixel location

def mean_absolute_error_mask(y_true, y_pred):                                                                                                                                                               
    y_mask = y_true[:,:,:,0] #bsize, h, w, 5(m,x,y,sin,cos)                                                                                                                                                 
    y_mask = K.expand_dims(y_mask, axis = -1)                                                                                                                                                               
    #print y_true.shape, y_pred.shape                                                                                                                                                                       
    y_true = y_true[:,:,:,1:]                                                                                                                                                                               
                                                                                                                                                                                                            
    return K.sum(K.abs((y_pred - y_true)*y_mask), axis=-1)                                                                                                                                                  
                                                                                                                                                                                                            
                                                                                                                                                                                                            
def mean_absolute_percentage_error_mask(y_true, y_pred):                                                                                                                                                    
    y_mask = y_true[:,:,:,0] #bsize, h, w, 5(m,x,y,sin,cos)                                                                                                                                                 
    y_mask = K.expand_dims(y_mask, axis = -1)                                                                                                                                                               
    #print y_true.shape, y_pred.shape                                                                                                                                                                       
    y_true = y_true[:,:,:,1:]                                                                                                                                                                               
                                                                                                                                                                                                            
    diff = K.abs(((y_true - y_pred))*y_mask / K.clip(K.abs(y_true * y_mask),                                                                                                                                
                                            K.epsilon(),                                                                                                                                                    
                                            None))                                                                                                                                                          
    return 100. * K.sum(diff, axis=-1)             
    