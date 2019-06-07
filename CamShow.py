from OboardCamDisp import Ui_MainWindow
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer,QCoreApplication
from PyQt5.QtGui import QPixmap,QImage
import cv2
import predict
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time



def Init_net():
    class_names = ['angry', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised']
    vgg16 = models.vgg16_bn()
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    was_training = vgg16.training
    vgg16.train(False)
    vgg16.eval()
    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 6 outputs
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

    vgg16.load_state_dict(torch.load("../2VGG16_v2-OCT_Retina_half_dataset.pth", map_location='cpu'))  #载入训练所得参数集
    return vgg16         #返回网络



class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CamShow, self).__init__(parent)
        self.setupUi(self)

        self.bt1 =QtWidgets.QPushButton('选择图片', self)  # 选择图片按钮
        self.bt1.setGeometry(10, 350, 80, 30)
        self.bt2 = QPushButton('识别', self)  # 识别按钮
        self.bt2.setGeometry(10, 500, 80, 30)
        self.bt1.clicked.connect(self.choiceimage)  # 选择图片信号
        self.bt2.clicked.connect(self.detect)       #识别信号
        self.choicelable = QLabel(self)
        self.detectlabele=QLabel(self)
        self.textlable = QLabel(self)
        self.pic=''                        #存储图片路径
        self.image_path=''                 #存储图片路径
        self.flag=0                        #flag=1时为图片识别模式；flsg=2时为摄像头模式
        self.vgg=Init_net()

        self.PrepSliders()
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

#选择图片函数，响应选择图片按钮，将选择的图片放至GUI界面，并返回文件路径
    def choiceimage(self):
        image_file, imgtype= QFileDialog.getOpenFileName(self, 'Open file', "C:\\", '*.jpg  *.png *.jpeg')
        self.pic = image_file
        self.flag=1
        imagein = QImage()
        imagein.load(image_file)
        self.choicelable.setGeometry(100, 350, 200, 150)
        self.choicelable.setPixmap(QPixmap.fromImage(imagein))
        self.choicelable.setScaledContents(True)
#识别函数，识别表情
    def detect(self):
        if self.flag==1:   #图片识别模式
            imageout = QImage()
            imageout.load(self.pic)   #载入选择图片
            self.detectlabele.setGeometry(100, 515, 200, 150)
            self.detectlabele.setPixmap(QPixmap.fromImage(imageout))
            self.detectlabele.setScaledContents(True)
            out=predict.Predict(self.pic,self.vgg)  #识别，返回结果
            self.textlable.setText(out)
            self.textlable.setFont(QFont("Microsoft YaHei",13))  #展示识别结果
            self.textlable.setGeometry(310,570,80,30)
        if self.flag==2:    #摄像头模式
            imageout = QImage()
            imageout.load(self.image_path)   #载入摄像头传回的图片
            self.detectlabele.setGeometry(100, 515, 200, 150)
            self.detectlabele.setPixmap(QPixmap.fromImage(imageout))
            self.detectlabele.setScaledContents(True)
            out=predict.Predict(self.image_path,self.vgg)  #识别，返回结果
            self.textlable.setText(out)
            self.textlable.setFont(QFont("Microsoft YaHei", 13))  #展示识别结果
            self.textlable.setGeometry(310, 570, 80, 30)


    def PrepSliders(self):
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)

    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)

    def PrepCamera(self):
        try:
            self.camera=cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))

    def PrepParameters(self):
        self.RecordFlag = 0
        self.RecordPath = 'C:\\Users\\ustclzt\\PycharmProjects\\camera\\'
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num = 0
        self.R = 1
        self.G = 1
        self.B = 1

        self.ExpTimeSld.setValue(self.camera.get(15))   # 曝光
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))  # 图像增益
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))  # 亮度
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))  # 对比度
        self.SetContrast()
        self.MsgTE.clear()

    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)
        self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)

    #以下三个函数分别可以调整图像中R、G、B三种颜色的占比
    def SetR(self):
        R=self.RedColorSld.value()
        self.R=R/255

    def SetG(self):
        G=self.GreenColorSld.value()
        self.G=G/255

    def SetB(self):
        B=self.BlueColorSld.value()
        self.B=B/255

    #调整对比度
    def SetContrast(self):
        contrast_toset=self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    #调整亮度
    def SetBrightness(self):
        brightness_toset=self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

     #调整增益
    def SetGain(self):
        gain_toset=self.GainSld.value()
        try:
            self.camera.set(14,gain_toset)
            self.MsgTE.setPlainText('The gain is set to '+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

     #调整曝光
    def SetExposure(self):
        try:
            exposure_time_toset=self.ExpTimeSld.value()
            self.camera.set(15,exposure_time_toset)
            self.MsgTE.setPlainText('The exposure time is set to '+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    #对三个颜色通道的slider和spinbox启用与否的控制，若为真，则将图像转为灰度图
    def SetGray(self):
        if self.GrayImgCkB.isChecked():
            self.RedColorSld.setEnabled(False)
            self.RedColorSpB.setEnabled(False)
            self.GreenColorSld.setEnabled(False)
            self.GreenColorSpB.setEnabled(False)
            self.BlueColorSld.setEnabled(False)
            self.BlueColorSpB.setEnabled(False)
        else:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)

    #配置摄像头参数，实现图像在GUI界面的实时显示
    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.GrayImgCkB.setEnabled(True)
        if self.GrayImgCkB.isChecked()==0:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('录像')

        self.Timer.start(1)
        self.timelb=time.clock()

    #调用QFileDialog.getExistingDirectory函数弹出对话框，让用户自己选择路径，并将选择好的路径显示在FilePathLE中。
    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'

    #从摄像头读取图像，调用ColorAdjust函数来调整图片的颜色，调用DispImg函数来显示图像，保存视频，获取并显示摄像头帧频和图像尺寸。
    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            #self.Image=img
            self.DispImg()
            self.Image_num+=1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
                #size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')

    #调整颜色函数
    def ColorAdjust(self,img):
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R
            #B.astype(cv2.PARAM_UNSIGNED_INT)
            #G.astype(cv2.PARAM_UNSIGNED_INT)
            #R.astype(cv2.PARAM_UNSIGNED_INT)

            img1 = img
            img1[:, :, 0] = B
            img1[:, :, 1] = G
            img1[:, :, 2] = R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

     #显示图像
    def DispImg(self):
        height, width, bytesPerComponent = self.Image.shape
        bytesPerLine = 3 * width
        if self.GrayImgCkB.isChecked():
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY,self.Image)  #若选择了Gray，则将图片转成灰度图显示
        else:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB,self.Image)   #否则以RGB的形式显示
        qimg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.DispLb.setPixmap(QPixmap.fromImage(qimg))
        self.DispLb.show()

     #暂停’和‘继续’功能是复用的同一个按钮，若点击时按钮显示的是‘暂停’，那么就停止计时器，同时将本按钮的显示改成‘继续’
    def StopCamera(self):
        if self.StopBt.text() == '暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text() == '继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)

    def RecordCamera(self):
        tag = self.RecordBt.text()
        if tag == '保存':  #保存暂停时的图片
            try:
                image_name = self.RecordPath+'image'+'display'+'.jpg'
                self.Image= cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB, self.Image) #将图片转成RGB模式存储
                cv2.imwrite(image_name, self.Image)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
                self.image_path=image_name
                self.flag=2
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag == '录像':  #保存录下的视频
            self.RecordBt.setText('停止')
            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)

    def ExitApp(self):  #退出GUI界面
        #self.Timer.Stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CamShow()
    ui.show()
    sys.exit(app.exec_())
