# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os

from utils import *


import sys
from PyQt5 import QtGui, QtCore,QtWidgets
import cv2
from PIL import Image
from numpy import array
global FLAGS #= argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
global framenum
global load_op
global FLAGS 

global framenum


#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#def _main():
def detect():
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface%05d.jpg'%framenum
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    if not args.image:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        if args.image:
            cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')




def my_func(a):
    return (a[0] + a[1]+ a[2])/3

class QtCapture(QtWidgets.QWidget):
    def __init__(self, *args):
        super(QtWidgets.QWidget, self).__init__()

        self.fps = 10
        self.cap = cv2.VideoCapture(*args)

        self.video_frame = QtWidgets.QLabel()
        lay = QtWidgets.QVBoxLayout()
        lay.setContentsMargins(0,0,0,0)
#        lay.setContentsMargin(0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)

        # ------ Modification ------ #
        self.isCapturing = False
        self.ith_frame = 1
        # ------ Modification ------ #

    def setFPS(self, fps):
        self.fps = fps
        


    def nextFrameSlot(self):
        global framenum
        framenum=self.ith_frame
        ret, frame = self.cap.read()

        # ------ Modification ------ #
        # Save images if isCapturing
        if self.isCapturing:
            cv2.imwrite('input\\img_%05d.jpg'%self.ith_frame, frame)
            img = Image.open('input\\img_%05d.jpg'%self.ith_frame)
            #img.show()
#            arr = array(img)/255  
#            fiesta=np.apply_along_axis(my_func,2,arr) 
            args.image='input\\img_%05d.jpg'%self.ith_frame
            detect( ) 
            self.ith_frame += 1
            framenum=self.ith_frame
            #self.timer.stop()

        # ------ Modification ------ #

        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)
        #self.timer.stop()
        

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()


class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.capture = None

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QtWidgets.QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QtWidgets.QPushButton('Stop')
#        self.capture = QtCapture(0)
#        self.end_button.clicked.connect(self.capture.stop)
#        # self.capture.setFPS(1)
#        self.capture.setParent(self)
#        self.capture.setWindowFlags(QtCore.Qt.Tool)
#        self.capture.start()

        # ------ Modification ------ #
        self.capture_button = QtWidgets.QPushButton('Capture')
        self.capture_button.clicked.connect(self.saveCapture)
        # ------ Modification ------ #

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.capture)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        # ------ Modification ------ #
        vbox.addWidget(self.capture_button)
        # ------ Modification ------ #

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(300,300,200,200)

        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    # ------ Modification ------ #
    def saveCapture(self):
        if self.capture:
            self.capture.capture()
    # ------ Modification ------ #


if __name__ == '__main__':
    global FLAGS
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
#    _main()
