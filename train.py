import keras
import keras.backend as K
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from models import discriminator, generator,generator2,GAN
from fish_dataset import load_dataset, load_dataset_data_augument
from PIL import Image
import math
import os
import tensorflow as tf
import argparse
from progressbar import ProgressBar


def train():
    parser = argparse.ArgumentParser(description = "keras pix2pix")
    parser.add_argument('--batchsize', '-b', type=int, default = 1)
    parser.add_argument('--patchsize', '-p', type=int, default = 64)
    parser.add_argument('--epoch', '-e', type=int, default = 500)
    parser.add_argument('--out', '-o',default = 'result')
    parser.add_argument('--lmd', '-l',type=int, default = 100)
    parser.add_argument('--dark', '-d',type=float, default = 0.01)
    parser.add_argument('--gpu', '-g', type = int, default = 2)
    args = parser.parse_args()
    args = parser.parse_args()
    PATCH_SIZE = args.patchsize
    BATCH_SIZE = args.batchsize
    epoch      = args.epoch
    lmd        = args.lmd

    # set gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    K.set_session(sess)




    # make directory to save results
    if not os.path.exists("./result"):
        os.mkdir("./result")
    resultDir = "./result/" + args.out
    modelDir  = resultDir + "/model/"
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)

    # make a logfile and add colnames
    o = open(resultDir + "/log.txt","w")
    o.write("batch:" + str(BATCH_SIZE) + "  lambda:" + str(lmd) + "\n")
    o.write("epoch,dis_loss,gan_mae,gan_entropy,vdis_loss,vgan_mae,vgan_entropy" + "\n")
    o.close()

    # load data
    ds1_first, ds1_last, num_ds1 = 1,    1145, 1145
    ds2_first, ds2_last, num_ds2 = 2000, 6749, 4750
    # ds1_first, ds1_last, num_ds1 = 1,    100, 100
    # ds2_first, ds2_last, num_ds2 = 101, 200, 100
    train_data_i = np.concatenate([np.arange(ds1_first,ds1_last+1)[:int(num_ds1 * 0.7)],
                                 np.arange(ds2_first,ds2_last+1)[:int(num_ds2*0.7)]])
    test_data_i  = np.concatenate([np.arange(ds1_first,ds1_last+1)[int(num_ds1 * 0.7):],
                                 np.arange(ds2_first,ds2_last+1)[int(num_ds2*0.7):]])
    train_gt, _, train_night = load_dataset(data_range=train_data_i, dark = args.dark)
    test_gt,  _, test_night  = load_dataset(data_range=test_data_i,  dark = args.dark)

    # Create optimizers
    opt_Gan           = Adam(lr=1E-3)
    opt_Discriminator = Adam(lr=1E-3)
    opt_Generator     = Adam(lr=1E-3)

    # set the loss of gan
    def dis_entropy(y_true, y_pred):
        return -K.log(K.abs((y_pred - y_true)) + 1e-07)
    gan_loss = ['mae', dis_entropy]
    gan_loss_weights = [lmd,1]


    # make models
    Generator     = generator()
    Generator.compile(loss = 'mae', optimizer=opt_Generator)
    Discriminator = discriminator()
    Discriminator.trainable = False
    Gan = GAN(Generator,Discriminator)
    Gan.compile(loss = gan_loss, loss_weights = gan_loss_weights,optimizer = opt_Gan)
    Discriminator.trainable = True
    Discriminator.compile(loss=dis_entropy, optimizer=opt_Discriminator)

    # start training
    n_train = train_gt.shape[0]
    n_test = test_gt.shape[0]
    print(n_train, n_test)
    p = ProgressBar()
    for epoch in p(range(epoch)):
        p.update(epoch+1)
        out_file = open(resultDir + "/log.txt","a")
        train_ind = np.random.permutation(n_train)
        test_ind  = np.random.permutation(n_test)
        dis_losses = []
        gan_losses = []
        test_dis_losses = []
        test_gan_losses = []
        y_real = np.array([1] * BATCH_SIZE)
        y_fake = np.array([0] * BATCH_SIZE)
        y_gan  = np.array([1] * BATCH_SIZE)

        # training
        for batch_i in range(int(n_train/BATCH_SIZE)):
            gt_batch        = train_gt[train_ind[(batch_i*BATCH_SIZE) : ((batch_i+1)*BATCH_SIZE)],:,:,:]
            night_batch     = train_night[train_ind[(batch_i*BATCH_SIZE) : ((batch_i+1)*BATCH_SIZE)],:,:,:]
            generated_batch = Generator.predict(night_batch)
            # train Discriminator
            dis_real_loss = np.array(Discriminator.train_on_batch([night_batch,gt_batch],y_real))
            dis_fake_loss = np.array(Discriminator.train_on_batch([night_batch,generated_batch],y_fake))
            dis_loss_batch = (dis_real_loss + dis_fake_loss) / 2
            dis_losses.append(dis_loss_batch)
            gan_loss_batch = np.array(Gan.train_on_batch(night_batch, [gt_batch, y_gan]))
            gan_losses.append(gan_loss_batch)
        dis_loss = np.mean(np.array(dis_losses))
        gan_loss = np.mean(np.array(gan_losses), axis=0)

        # validation
        for batch_i in range(int(n_test/BATCH_SIZE)):
            gt_batch        = test_gt[test_ind[(batch_i*BATCH_SIZE) : ((batch_i+1)*BATCH_SIZE)],:,:,:]
            night_batch     = test_night[test_ind[(batch_i*BATCH_SIZE) : ((batch_i+1)*BATCH_SIZE)],:,:,:]
            generated_batch = Generator.predict(night_batch)
            # train Discriminator
            dis_real_loss = np.array(Discriminator.test_on_batch([night_batch,gt_batch],y_real))
            dis_fake_loss = np.array(Discriminator.test_on_batch([night_batch,generated_batch],y_fake))
            test_dis_loss_batch = (dis_real_loss + dis_fake_loss) / 2
            test_dis_losses.append(test_dis_loss_batch)
            test_gan_loss_batch = np.array(Gan.test_on_batch(night_batch, [gt_batch, y_gan]))
            test_gan_losses.append(test_gan_loss_batch)
        test_dis_loss = np.mean(np.array(test_dis_losses))
        test_gan_loss = np.mean(np.array(gan_losses), axis=0)
        # write log of leaning
        out_file.write(str(epoch) + "," + str(dis_loss) + "," + str(gan_loss[1]) + "," + str(gan_loss[2]) + "," + str(test_dis_loss) + ","+ str(test_gan_loss[1]) +"," + str(test_gan_loss[2]) + "\n")

        # visualize
        if epoch % 50 == 0 :
            # for training data
            gt_batch        = train_gt[train_ind[0:9],:,:,:]
            night_batch     = train_night[train_ind[0:9],:,:,:]
            generated_batch = Generator.predict(night_batch)
            save_images(night_batch,     resultDir + "/label_"     + str(epoch)+"epoch.png")
            save_images(gt_batch,        resultDir + "/gt_"        + str(epoch)+"epoch.png")
            save_images(generated_batch, resultDir + "/generated_" + str(epoch)+"epoch.png")
            # for validation data
            gt_batch        = test_gt[test_ind[0:9],:,:,:]
            night_batch     = test_night[test_ind[0:9],:,:,:]
            generated_batch = Generator.predict(night_batch)
            save_images(night_batch,     resultDir + "/vlabel_"     + str(epoch)+"epoch.png")
            save_images(gt_batch,        resultDir + "/vgt_"        + str(epoch)+"epoch.png")
            save_images(generated_batch, resultDir + "/vgenerated_" + str(epoch)+"epoch.png")

            Gan.save_weights(modelDir + 'gan_weights' + "_lambda" + str(lmd) + "_epoch"+ str(epoch) + '.h5')

        out_file.close()
    out_file.close()
    # gan.save("gan_" + "patch" + str(patch_size) + ".h5")

def save_images(imgs, out_file_name):
    combined_img = combine_images(imgs)
    if combined_img.shape[2] == 1:
        combined_img = combined_img.reshape(combined_img.shape[0:2])
    combined_rgb_img = combined_img*128.0+128.0
    Image.fromarray(combined_rgb_img.astype(np.uint8)).save(out_file_name)

# バッチ画像を並べて1つにする
# shapeが(9,10,10,3)から(1,30,30,3)になる。
def combine_images(imgs):
    num    = imgs.shape[0]
    width  = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    ch     = imgs.shape[3]
    shape  = imgs.shape[1:3]
    image  = np.zeros((height*shape[0], width*shape[1],ch),dtype=imgs.dtype)
    for index, img in enumerate(imgs):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

if __name__ == '__main__':
    train()
