代码需要在tensorflow2.0上运行或者1.14上运行

输入支持letter_box操作

keras模式可以正常训练,多gpu测试暂未通过，报段错误，预计和tf.data和strategy有一定关系, tf.estimator支持多卡和分布式训练

result文件夹中是我训练的一些马赛克检测的模型检测结果,是在bacth=4,step=30000,三卡模式下训练的，训练数据大概3000-4000张图片的结果

## 数据格式:

/home/admin-seu/sss/yolo-V3/data/train/0.jpg 0,0.6059782608695652,0.5742971887550201,0.9402173913043478,0.9116465863453815

/home/admin-seu/sss/yolo-V3/data/train/1.jpg 0,0.10852713178294573,0.07625272331154684,0.46511627906976744,0.3159041394335512 0,0.6124031007751938,0.1111111111111111,0.8837209302325582,0.2657952069716    776 0,0.3953488372093023,0.032679738562091505,0.5387596899224806,0.29193899782135074

数据已支持数据翻转，对比度变换等数据增强操作

## Steps
config/yolov3.yaml  训练测试配置文件

yolov3_model.py 模型训练, 运行python yolov3_model.py进行训练和评估

目前支持用BN和GN进行训练

## 评估

根据训练结果自己进行评估0.0

## 靠

1. 靠縄CCV2019 oral CARAFE靠靠靠靠靠靠靠縝ilinear靠靠靠
