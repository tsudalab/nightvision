## sonar to real by pix2pix

### USAGE
- ソナーを使わないモデル  
`python train.py -g [gpuの番号] -b [バッチサイズ] -e [エポック数] -o [出力ディレクトリ] -l [lambda] -d [暗さレベル]`
- ソナーを使うモデル  
`python train_sonar.py -g [gpuの番号] -b [バッチサイズ] -e [エポック数] -o [出力ディレクトリ] -l [lambda] -d [暗さレベル]`


### METHOD
![network](https://user-images.githubusercontent.com/17425130/35025664-155e67ec-fb8a-11e7-9e98-697c07d5b163.png)
