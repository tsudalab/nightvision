import numpy as np
from PIL import Image,ImageOps
from io import BytesIO


def load_dataset(dataDir='/data1/train_data/', data_range=range(0,300),test=False, dark=10,exclude = False):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset    = []
        nightDataset = []
        sonarDataset  = []
        if exclude:
            # trainingに使えないデータ(エイがカメラの前を通った場面など)を除去
            excludes = np.concatenate([np.arange(226,253), np.arange(445,455), np.arange(796, 803), np.arange(2100,2117),
                                   np.arange(2267, 2317), np.arange(2764, 2835), np.arange(3009, 3029), np.arange(3176, 3230),
                                   np.arange(3467, 3490), np.arange(3665, 3735), np.arange(3927, 4001), np.arange(4306,4308),
                                   np.arange(4416, 4476), np.arange(4737, 4741), np.arange(4846, 4906), np.arange(5406, 5464),
                                   np.arange(5807, 5841), np.arange(6101, 6145)]) # training対象外
            mask = [d not in excludes for d in data_range]
            data_range = data_range[mask]

        imgStart   = 0
        sonarStart = 0
        nightStart = 0
        for i in data_range:
            if test:
                if i%3 != 1:
                    continue
            if not test:
                if i%3 != 0:
                    continue
            imgNum   = imgStart + i
            sonarNum = sonarStart + i
            nightNum = nightStart + i
            img   = Image.open(dataDir + "up/up%05d.png"%imgNum)
            night = Image.open(dataDir + "night_100/" +"up_" + str(dark).replace('.','') + "night/night_up%05d.png"%nightNum)
            sonar = Image.open(dataDir + "sonar/sonar%05d.png"%sonarNum)
            sonar = sonar.convert("L")

            # 短い辺が300pixになるようにresizeし、rgbを(-1,1)に正規化
            w,h = img.size
            r = 300/min(w,h)
            img   = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            night = night.resize((int(r*w), int(r*h)),Image.BILINEAR)
            sonar = sonar.resize((int(r*w), int(r*h)),Image.BILINEAR)
            img   = np.asarray(img)/128.0-1.0
            sonar = (np.asarray(sonar)/128.0-1.0)[:,:,np.newaxis]
            night = np.asarray(night)/128.0-1.0
            #512 * 256にランダムクリップ → ランダムを廃止
            h,w,_ = img.shape
            xl = int((w-256)/2)
            yl = int(h-512)
            # xl = np.random.randint(0,w-256)
            # yl = np.random.randint(0,h-512)
            img = img[yl:yl+512, xl:xl+256, :]
            sonar = sonar[yl:yl+512, xl:xl+256,:]
            night = night[yl:yl+512, xl:xl+256,:]

            imgDataset.append(img)
            sonarDataset.append(sonar)
            nightDataset.append(night)


        print("load dataset done")
        return np.array(imgDataset),np.array(sonarDataset),np.array(nightDataset)

def load_dataset_box(dataDir='/data1/train_data/', data_range=range(0,300),test=False, dark=10):
    print("load dataset start")
    print("     from: %s"%dataDir)
    imgDataset    = []
    nightDataset = []
    sonarDataset  = []

    # trainingに使えないデータ(エイがカメラの前を通った場面など)を除去
    excludes = np.concatenate([np.arange(226,253), np.arange(445,455), np.arange(796, 803), np.arange(2100,2117),
                               np.arange(2267, 2317), np.arange(2764, 2835), np.arange(3009, 3029), np.arange(3176, 3230),
                               np.arange(3467, 3490), np.arange(3665, 3735), np.arange(3927, 4001), np.arange(4306,4308),
                               np.arange(4416, 4476), np.arange(4737, 4741), np.arange(4846, 4906), np.arange(5406, 5464),
                               np.arange(5807, 5841), np.arange(6101, 6145)]) # training対象外
    mask = [d not in excludes for d in data_range]
    data_range = data_range[mask]

    imgStart   = 0
    sonarStart = 0
    nightStart = 0
    for i in data_range:
        if test:
            if i%3 != 1:
                continue
        if not test:
            if i%3 != 0:
                continue
        imgNum   = imgStart + i
        sonarNum = sonarStart + i
        nightNum = nightStart + i
        img   = Image.open(dataDir + "up/up%05d.png"%imgNum)
        sonar = Image.open(dataDir + "sonar/sonar%05d.png"%sonarNum)
        sonar = sonar.convert("L")

        # 短い辺が300pixになるようにresizeし、rgbを(-1,1)に正規化
        w,h = img.size
        r = 300/min(w,h)
        img   = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
        sonar = sonar.resize((int(r*w), int(r*h)),Image.BILINEAR)
        img   = np.asarray(img)/128.0-1.0
        sonar = (np.asarray(sonar)/128.0-1.0)[:,:,np.newaxis]
        # 512 * 256にランダムクリップ
        h,w,_ = img.shape
        if test:
            xl = int(w-256)
            yl = int(h-512)
        else:
            xl = np.random.randint(0,w-256)
            yl = np.random.randint(0,h-512)
        img = img[yl:yl+512, xl:xl+256, :]
        sonar = sonar[yl:yl+512, xl:xl+256,:]

        imgDataset.append(img)
        sonarDataset.append(sonar)
        box_img = np.copy(img)
        box_size = 150
        xl = np.random.randint(0,256 - box_size)
        yl = np.random.randint(0,512 - box_size)
        box_img[yl:yl+box_size,xl:xl+box_size] = -1
        nightDataset.append(box_img)


    print("load dataset done")
    return np.array(imgDataset),np.array(sonarDataset),np.array(nightDataset)



def load_dataset_data_augument(dataDir='/data1/train_data/', data_range=range(0,300),test=False, dark=10):
        print("load dataset start")
        print("     from: %s"%dataDir)
        imgDataset    = []
        nightDataset = []
        sonarDataset  = []

        # trainingに使えないデータ(エイがカメラの前を通った場面など)を除去
        excludes = np.concatenate([np.arange(226,253), np.arange(445,455), np.arange(796, 803), np.arange(2100,2117),
                                   np.arange(2267, 2317), np.arange(2764, 2835), np.arange(3009, 3029), np.arange(3176, 3230),
                                   np.arange(3467, 3490), np.arange(3665, 3735), np.arange(3927, 4001), np.arange(4306,4308),
                                   np.arange(4416, 4476), np.arange(4737, 4741), np.arange(4846, 4906), np.arange(5406, 5464),
                                   np.arange(5807, 5841), np.arange(6101, 6145)]) # training対象外
        mask = [d not in excludes for d in data_range]
        data_range = data_range[mask]

        imgStart   = 0
        sonarStart = 0
        nightStart = 0
        for i in data_range:
            if test:
                if i%3 != 1:
                    continue
            if not test:
                if i%3 != 0:
                    continue

            imgNum   = imgStart + i
            sonarNum = sonarStart + i
            nightNum = nightStart + i
            img   = Image.open(dataDir + "up/up%05d.png"%imgNum)
            night = Image.open(dataDir + "night_100/" +"up_" + str(dark).replace('.','') + "night/night_up%05d.png"%nightNum)
            sonar = Image.open(dataDir + "sonar/sonar%05d.png"%sonarNum)
            sonar = sonar.convert("L")



            # 短い辺が300pixになるようにresizeし、rgbを(-1,1)に正規化
            # データを対称変換したaugmentation dataを追加
            w,h = img.size
            r = 300/min(w,h)
            img   = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
            night = night.resize((int(r*w), int(r*h)),Image.BILINEAR)
            sonar = sonar.resize((int(r*w), int(r*h)),Image.BILINEAR)

            aug_img   = augument(img)
            aug_night = augument(night)
            aug_sonar = augument(sonar)
            img   = np.asarray(img)/128.0-1.0
            sonar = (np.asarray(sonar)/128.0-1.0)[:,:,np.newaxis]
            night = np.asarray(night)/128.0-1.0
            aug_img   = np.asarray(aug_img)/128.0-1.0
            aug_sonar = (np.asarray(aug_sonar)/128.0-1.0)[:,:,np.newaxis]
            aug_night = np.asarray(aug_night)/128.0-1.0

            # 512 * 256にランダムクリップ
            h,w,_ = img.shape
            if test:
                xl = int(w-256)
                yl = int(h-512)
            else:
                xl = np.random.randint(0,w-256)
                yl = np.random.randint(0,h-512)
            # img = img[yl:yl+512, xl:xl+256, :]
            # sonar = sonar[yl:yl+512, xl:xl+256,:]
            # night = night[yl:yl+512, xl:xl+256,:]
            aug_img = aug_img[yl:yl+512, xl:xl+256, :]
            aug_sonar = aug_sonar[yl:yl+512, xl:xl+256,:]
            aug_night = aug_night[yl:yl+512, xl:xl+256,:]

            # imgDataset.append(img)
            imgDataset.append(aug_img)
            # sonarDataset.append(sonar)
            sonarDataset.append(aug_sonar)
            # nightDataset.append(night)
            nightDataset.append(aug_night)

        print("load dataset done")
        return np.array(imgDataset),np.array(sonarDataset),np.array(nightDataset)

def augument(img):
    flag = np.random.choice([0,1,2,3])
    if flag == 0:
        return ImageOps.mirror(img)
    if flag == 1:
        return ImageOps.flip(img)
    if flag == 2:
        return ImageOps.flip(ImageOps.mirror(img))
    if flag == 3:
        return img
