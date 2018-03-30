import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense,Reshape, Input,merge
from keras.layers.merge import concatenate
from keras.layers.core import Activation, Dropout, Flatten,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D,Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

# convolution batchnormalization relu
def CBR(ch,shape,bn=True,sample='down',activation=LeakyReLU, dropout=False):
    model = Sequential()
    if sample=='down':
        model.add(Conv2D(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    else:
        model.add(Conv2DTranspose(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    if bn:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.5))
    if activation == LeakyReLU:
        model.add(LeakyReLU(alpha=0.2))
    else:
        model.add(Activation('relu'))
    return model



def discriminator_sonar():
    h = 512
    w = 256
    label_input = Input(shape=(h,w,1))
    gen_output = Input(shape=(h,w,3))
    x1 = CBR(32,(512,256,1), bn=False)(label_input)
    x2 = CBR(32,(512,256,3),bn=False)(gen_output)
    x = concatenate([x1,x2])
    x = CBR(128,(256,128,64))(x)
    x = CBR(256,(128,64,128))(x)
    x = CBR(512,(64,32,256))(x)
    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[label_input,gen_output], outputs = [output])
    return model

def discriminator_sonar_3cond():
    h = 512
    w = 256
    input_s = Input(shape=(h,w,1))
    input_c = Input(shape=(h,w,3))
    input_x = Input(shape=(h,w,3))
    x1 = CBR(32,(512,256,1), bn=False)(input_s)
    x2 = CBR(32,(512,256,3),bn=False)(input_c)
    x3 = CBR(32,(512,256,3),bn=False)(input_x)
    x = concatenate([x1,x2,x3])
    x = CBR(128,(256,128,96))(x)
    x = CBR(256,(128,64,128))(x)
    x = CBR(512,(64,32,256))(x)
    x = CBR(1024,(32,16,512))(x)
    x = CBR(1024,(16,8,1024))(x)
    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[label_input,gen_output], outputs = [output])
    return model



def discriminator_sonar2():
    h = 512
    w = 256
    label_input = Input(shape=(h,w,1))
    gen_output = Input(shape=(h,w,3))
    x1 = CBR(32,(512,256,1), bn=False)(label_input)
    x2 = CBR(32,(512,256,3),bn=False)(gen_output)
    x = concatenate([x1,x2])
    x = CBR(128,(256,128,64))(x)
    x = CBR(256,(128,64,128))(x)
    x = CBR(512,(64,32,256))(x)
    x = CBR(1024,(32,16,512))(x)
    x = CBR(1024,(16,8,1024))(x)
    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[label_input,gen_output], outputs = [output])
    return model


def discriminator():
    h = 512
    w = 256
    label_input = Input(shape=(h,w,3))
    gen_output = Input(shape=(h,w,3))
    x1 = CBR(32,(512,256,3), bn=False)(label_input)
    x2 = CBR(32,(512,256,3),bn=False)(gen_output)
    x = concatenate([x1,x2])
    x = CBR(128,(256,128,64))(x)
    x = CBR(256,(128,64,128))(x)
    x = CBR(512,(64,32,256))(x)
    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[label_input,gen_output], outputs = [output])
    return model


def discriminator_nocondition():
    h = 512
    w = 256
    input_img = Input(shape=(h,w,3))
    x = CBR(32,(512,256,3),bn=False)(input_img)
    x = CBR(64,(256,128,32))(x)
    x = CBR(128,(128,64,64))(x)
    x = CBR(256,(64,32,128))(x)
    x = Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
    x = Activation('sigmoid')(x)
    output = Lambda(lambda x: K.mean(x, axis=[1,2]),output_shape=(1,))(x)
    model = Model(inputs =[input_img], outputs = [output])
    return model


def discriminator_sonar_nobatch():
    h = 512
    w = 256
    label_input = Input(shape=(h,w,1))
    gen_output = Input(shape=(h,w,3))
    x1 = CBR(32,(512,256,1), bn=False)(label_input)
    x2 = CBR(32,(512,256,3),bn=False)(gen_output)
    x = concatenate([x1,x2])
    x = CBR(128,(256,128,64))(x)
    x = CBR(256,(128,64,128))(x)
    x = CBR(512,(64,32,256))(x)
    x = CBR(512,(16,8,512))(x)
    x = CBR(512,(8,4,512))(x)
    x = Flatten()(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    model = Model(inputs =[label_input,gen_output], outputs = [output])
    return model

def generator_sonar():

    # encoder
    input1 = Input(shape=(512,256,1))
    enc_1 = Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,1))(input1)

    input2 = Input(shape=(512,256,3))
    enc_2 = Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,3))(input2)

    enc_3 = concatenate([enc_1,enc_2])

    enc_4 = CBR(64,(512,256,32))(enc_3)
    enc_5 = CBR(128,(256,128,64))(enc_4)
    enc_6 = CBR(256,(128,64,128))(enc_5)
    enc_7 = CBR(512,(64,32,256))(enc_6)
    enc_8 = CBR(512,(32,16,512))(enc_7)
    enc_9 = CBR(512,(16,8,512))(enc_8)
    enc_10 = CBR(512,(8,4,512))(enc_9)

    # decoder
    x = CBR(512,(4,2,512),sample='up',activation='relu',dropout=True)(enc_10)
    x = CBR(512,(8,4,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_9]))
    x = CBR(512,(16,8,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_8]))
    x = CBR(256,(32,16,1024),sample='up',activation='relu',dropout=False)(concatenate([x,enc_7]))
    x = CBR(128,(64,32,512),sample='up',activation='relu',dropout=False)(concatenate([x,enc_6]))
    x = CBR(64,(128,64,256),sample='up',activation='relu',dropout=False)(concatenate([x,enc_5]))
    x = CBR(32,(256,128,128),sample='up',activation='relu',dropout=False)(concatenate([x,enc_4]))
    output = Conv2D(filters=3, kernel_size=(3,3),strides=1,padding="same")(concatenate([x,enc_3]))
    model = Model(inputs=[input1,input2], outputs=output)
    return(model)


def generator():

    # encoder
    input = Input(shape=(512,256,3))
    enc_1 = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,3))(input)

    enc_2 = CBR(64,(512,256,32))(enc_1)
    enc_3 = CBR(128,(256,128,64))(enc_2)
    enc_4 = CBR(256,(128,64,128))(enc_3)
    enc_5 = CBR(512,(64,32,256))(enc_4)
    enc_6 = CBR(512,(32,16,512))(enc_5)
    enc_7 = CBR(512,(16,8,512))(enc_6)
    enc_8 = CBR(512,(8,4,512))(enc_7)

    # decoder
    x = CBR(512,(4,2,512),sample='up',activation='relu',dropout=True)(enc_8)
    x = CBR(512,(8,4,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_7]))
    x = CBR(512,(16,8,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_6]))
    x = CBR(256,(32,16,1024),sample='up',activation='relu',dropout=False)(concatenate([x,enc_5]))
    x = CBR(128,(64,32,512),sample='up',activation='relu',dropout=False)(concatenate([x,enc_4]))
    x = CBR(64,(128,64,256),sample='up',activation='relu',dropout=False)(concatenate([x,enc_3]))
    x = CBR(32,(256,128,128),sample='up',activation='relu',dropout=False)(concatenate([x,enc_2]))
    output = Conv2D(filters=3, kernel_size=(3,3),strides=1,padding="same")(concatenate([x,enc_1]))
    model = Model(inputs=input, outputs=output)
    return(model)


def GAN_sonar(generator, discriminator):

    s_input = Input(shape=(512,256,1))
    c_input = Input(shape=(512,256,3))
    generated_image = generator([s_input,c_input])
    DCGAN_output = discriminator([s_input,generated_image])
    DCGAN = Model(inputs=[s_input,c_input],outputs=[generated_image, DCGAN_output],name="DCGAN")
    return DCGAN


def GAN(generator, discriminator):
    input = Input(shape=(512,256,3))
    generated_image = generator(input)
    DCGAN_output = discriminator([input,generated_image])
    DCGAN = Model(inputs=input,outputs=[generated_image, DCGAN_output],name="DCGAN")
    return DCGAN

def GAN_nocond_dis(generator,discriminator):
    s_input = Input(shape=(512,256,1))
    c_input = Input(shape=(512,256,3))
    generated_image = generator([s_input,c_input])
    DCGAN_output = discriminator([generated_image])
    DCGAN = Model(input=[s_input,c_input],outputs=[generated_image,DCGAN_output],name="DCGAN")
    return DCGAN

def GAN_sonar_3cond(generator, discriminator):
    input_s = Input(shape=(512,256,1))
    input_c = Input(shape=(512,256,3))
    generated_image = generator([input_s,input_c])
    DCGAN_output = discriminator([s_input,generated_image,input_c])
    DCGAN = Model(inputs=[s_input,c_input],outputs=[generated_image, DCGAN_output],name="DCGAN")
    return DCGAN

# 改良版U-net
def generator2_sonar():

    # encoder
    input1 = Input(shape=(512,256,1))
    enc_1 = Conv2D(filters=16,kernel_size=(3,3), strides=1, padding='same', input_shape=(512,256,1))(input1)

    input2 = Input(shape=(512,256,3))
    enc_2 = Conv2D(filters=16, kernel_size=(3,3), strides = 1, padding = 'same', input_shape=(512,256,3))(input2)
    enc_3 = concatenate([enc_1,enc_2])

    enc_4 = CBR(64,(512,256,32))(enc_3)
    enc_5 = CBR(128,(256,128,64))(enc_4)
    enc_6 = CBR(256,(128,64,128))(enc_5)
    enc_7 = CBR(512,(64,32,256))(enc_6)
    enc_8 = CBR(512,(32,16,512))(enc_7)
    enc_9 = CBR(512,(16,8,512))(enc_8)
    enc_10 = CBR(512,(8,4,512))(enc_9)

    # decoder
    x = CBR(512,(4,2,512),sample='up',activation='relu',dropout=True)(enc_10)
    x = CBR(512,(8,4,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_9]))
    x = CBR(512,(16,8,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_8]))
    x = CBR(256,(32,16,1024),sample='up',activation='relu',dropout=False)(concatenate([x,enc_7]))
    x = CBR(128,(64,32,512),sample='up',activation='relu',dropout=False)(concatenate([x,enc_6]))
    x = CBR(64,(128,64,128),sample='up',activation='relu',dropout=False)(x)
    x = CBR(32,(256,128,64),sample='up',activation='relu',dropout=False)(x)
    output = Conv2D(filters=3, kernel_size=(3,3),strides=1,padding="same")(x)
    model = Model(inputs=[input1,input2], outputs=output)
    return(model)


def generator2():

    # encoder
    input = Input(shape=(512,256,3))
    enc_1 = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same',input_shape=(512,256,3))(input)

    enc_2 = CBR(64,(512,256,32))(enc_1)
    enc_3 = CBR(128,(256,128,64))(enc_2)
    enc_4 = CBR(256,(128,64,128))(enc_3)
    enc_5 = CBR(512,(64,32,256))(enc_4)
    enc_6 = CBR(512,(32,16,512))(enc_5)
    enc_7 = CBR(512,(16,8,512))(enc_6)
    enc_8 = CBR(512,(8,4,512))(enc_7)

    # decoder
    x = CBR(512,(4,2,512),sample='up',activation='relu',dropout=True)(enc_8)
    x = CBR(512,(8,4,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_7]))
    x = CBR(512,(16,8,1024),sample='up',activation='relu',dropout=True)(concatenate([x,enc_6]))
    x = CBR(256,(32,16,1024),sample='up',activation='relu',dropout=False)(concatenate([x,enc_5]))
    x = CBR(128,(64,32,512),sample='up',activation='relu',dropout=False)(concatenate([x,enc_4]))
    x = CBR(64,(128,64,128),sample='up',activation='relu',dropout=False)(x)
    x = CBR(32,(256,128,64),sample='up',activation='relu',dropout=False)(x)
    output = Conv2D(filters=3, kernel_size=(3,3),strides=1,padding="same")(x)
    model = Model(inputs=input, outputs=output)
    return(model)
