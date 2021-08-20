import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import numpy as np
import time
import cv2
import dlib
import torch
import torchvision
import json
import socket
from scipy import spatial
from scipy.spatial.transform import Rotation
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

class Detect(object):
    def __init__(self, opt):
        self.img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.out = opt.output
        self.source = opt.source
        self.weights = opt.weights
        self.fourcc = opt.fourcc
        self.half = opt.half
        self.view_img = opt.view_img
        self.device = opt.device
        self.cfg = opt.cfg
        self.data = opt.data
        self.conf_thres = opt.conf_thres
        self.nms_thres = opt.conf_thres
        self.undistort = opt.undistort
        self.use_socket = opt.disable_socket
        self.save_img = opt.save
        self.debug = opt.silent
        self.vid_path = None
        self.vid_writer = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fps = 0
        self.load_camera_parameters(opt.camera_params)
        if self.use_socket:
            self.socket_connection(opt.test_socket)

        self.robot = 'KUKA'
        self.cartesian_xcoor = 0.017 
        self.cartesian_ycoor = 0.0096 

    def load_camera_parameters(self, cam_params_path):
        if self.debug:
            print('LOADING CAMERA PARAMETERS.......')
        with open(cam_params_path, 'rb') as f:
            self.rms = np.load(f)
            self.camera_matrix = np.load(f)
            self.dist_coefs = np.load(f)
            print('RMS: \n', self.rms)
            print('CAMERA MATRIX: \n', self.camera_matrix)
            print('DISTORTION COEFFICIENTS: \n', self.dist_coefs)
        if self.debug:
            print('CAMERA PARAMETERS LOADED')

    def socket_connection(self, test):
        if test:
            PORT = 8080
            IP = '127.0.0.1'
        else:
            PORT = 54600
            IP = '192.168.1.10'
        print('ADDRESS: TCP://{}:{}'.format(IP, PORT))
        print("CONNECTING......")
        while True:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            try:
                self.sock.connect((IP, PORT))
                print("CONNECTED.")
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        self.sock.settimeout(None)
        self.sock.setblocking(0)
        if not test:
            self.socket_recieve()

    def socket_recieve(self):
        while True:
            try:
                robot_data = self.sock.recv(1024)
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        if len(robot_data) == 0:
            print('RECIEVED EMPTY DATA, RETRYING CONNECTION.....')
            return self.socket_connection(False)
        else:
            return self.read_robot_data(robot_data)

    def read_robot_data(self, robot_data):
        if self.robot == 'ABB':
            if self.debug:
                print('ABB ROBOT')
            robot_data = self.read_json(robot_data)
        elif self.robot == 'KUKA':
            if self.debug:
                print('KUKA ROBOT')
            robot_data = self.read_xml(robot_data)
        else:
            assert True, 'ROBOT TYPE NOT SUPPORTED'
        if debug:
            print('STATE: ', robot_data)

    def read_json(self, robot_data):
        rd = json.loads(robot_data)
        state = rd[0]['mode']
        print('RECIEVED: ', state)
        if 0 < float(state) < 10 :
            return 1 
        elif 10 < float(state) < 20:
            return 2 
        elif 20 < float(state) < 30 :
            return 3
        elif 30 < float(state) < 40:
            return 4
        else:
            return 0

    def read_xml(self, robot_data):
        data = ElementTree.fromstring(robot_data)
        print('RECIEVED: ', data)
        data = float(data[0].attrib['Trig'])
        print(data)
        if 0 < data < 10:
            return 1
        elif 10 < data < 20:
            return 2
        elif 20 < data < 30:
            return 3
        elif 30 < data < 40:
            return 4
        else:
            return 0 

    def get_sources(self):
        if self.debug:
            print('CONFIGURING SOURCES....')
        try:
            int(self.source)
            self.webcam = True
        except:
            self.webcam = self.source.startswith('rtsp') or self.source.startswith('http') or self.source.endswith('.txt') or self.source.startswith('v4l2src') or self.source.startswith('basler')
        if self.webcam:
            if self.debug:
                print('USING STREAM SOURCES...')
            torch.backends.cudnn.benchmark = True  
            dataset = LiveFeed(self.source, half=self.half)
        else:
            if self.debug:
                print('USING MEDIA FILES SOURCES....')
            dataset = MediaFiles(self.source, half=self.half)
        if self.debug:
            print('SOURCES CONFIGURED')
        return dataset

    def load_model(self, device):
        if self.debug:
            print('LOADING MODEL....')
        model = Darknet(self.cfg, self.img_size)
        if self.weights.endswith('.pt'): 
            model.load_state_dict(torch.load(self.weights, map_location=device)['model'])
        else:  
            _ = load_darknet_weights(model, self.weights)
        model.to(device).eval()
        self.half = self.half and device.type != 'cpu'  
        if self.half:
            model.half()
        if self.debug:
            print('LOAD MODEL COMPLETED')
        return model

    def initialize_output(self):
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  
        os.makedirs(self.out)  

    def get_colors(self):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        return colors

    def yolov3_transform(self, im0, device):
        if self.webcam:
            img = [letterbox(x, new_shape=self.img_size, interp=cv2.INTER_LINEAR)[0] for x in im0]
            img = np.stack(img, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32) 
            img /= 255.0  
        else:
            img = letterbox(im0, new_shape=self.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
            img /= 255.0 
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def scale_detection_box(self, data):
        x, y, xmax, ymax = data
        x = x/self.width * 1920
        xmax = xmax/self.width * 1920
        y = y/self.height * 1080
        ymax = ymax/self.height * 1080
        return (x,y,xmax,ymax)

    def detect(self, img, model, colors):
        if self.debug:
            print('DETECTING IMAGE....')
        g_rects = []
        pred = model(img)[0]
        if self.half:
            pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.im0.shape).round()
                for *xyxy, conf, _, cls in det:
                    if self.save_img or self.view_img: 
                        label = '%s %.2f' % (self.classes[int(cls)], conf)
                        plot_one_box(xyxy, self.im0, label=label, color=colors[int(cls)])
                    x, y, xmax, ymax = self.scale_detection_box(xyxy)
                    g_rects.append([int(x), int(y), int(xmax), int(ymax), float(conf), int(cls)])
        g_rects = np.array(g_rects)
        if self.debug:
            print('RESULTS: ')
            print(g_rects)
        return g_rects

    def put_status_bar(self):
        cv2.rectangle(self.im0, (0, 0), (self.width, int(self.height/20.0)), (255, 255, 255), -1)
        cv2.putText(self.im0, 'FPS: %.1f' % (self.fps), (self.width - 112, 20), self.font, 0.4, (255, 0, 0), 1)

    def save_video(self, vid_cap):
        self.save_path = os.path.join(self.out,'sample.mp4')
        if self.vid_path != self.save_path:  
            self.vid_path = self.save_path
            if isinstance(self.vid_writer, cv2.VideoWriter):
                self.vid_writer.release()  
            h,w = self.im0.shape[:2]
            self.vid_writer = cv2.VideoWriter(self.save_path,
                                              cv2.VideoWriter_fourcc(*self.fourcc),
                                              30, (w, h))
        self.vid_writer.write(self.im0)

    def save_to_out(self, mode, vid_cap):
        #cv2.imwrite(self.save_path, self.im0)
        self.save_video(vid_cap)

    def get_xyedge_pos(self, g_rects):
        if len(g_rects) == 1:
            return np.squeeze(g_rects, axis=0)
        else:
            x_idx = np.argmax(g_rects[:,2])
            x_cluster = g_rects[abs(g_rects[x_idx][2] - g_rects[:,2]) < 0.2]
            xy_idx = np.argmin(x_cluster[:,3])
        ret = x_cluster[xy_idx]
        print('SELECTED COORDINATES: ', ret)
        return ret

    def transform_pnp(self, rect):
        if self.debug:
            print('CALCULATING COORDINATES.....')
        planar_coordinate = self.get_planar_coordinate()
        camera_coordinate = self.get_camera_coordinate(rect)
        ret, rvec, tvec = cv2.solvePnP(planar_coordinate, camera_coordinate, self.camera_matrix, self.dist_coefs)
        if self.debug:
            print('ROTATION VECTOR: \n', rvec)
            print('TRANSLATION VECTOR: \n', tvec)
        return rvec, tvec

    def get_planar_coordinate(self):
        planar_coordinate = np.array([[-self.cartesian_xcoor / 2.0, self.cartesian_ycoor / 2.0, 0],
                                      [self.cartesian_xcoor / 2.0, self.cartesian_ycoor / 2.0, 0],
                                      [self.cartesian_xcoor / 2.0, -self.cartesian_ycoor / 2.0, 0],
                                      [-self.cartesian_xcoor / 2.0, -self.cartesian_ycoor / 2.0, 0]])
        return planar_coordinate.astype('float32')

    def get_camera_coordinate(self, camera_rect):
        camera_coordinate = np.array([[[camera_rect[0], camera_rect[3]]],
                                      [[camera_rect[2], camera_rect[3]]],
                                      [[camera_rect[2], camera_rect[1]]],
                                      [[camera_rect[0], camera_rect[1]]]])
        return camera_coordinate.astype('float32')

    def transform_euler_angles(self, rvec):
        rot_matrix, _ = cv2.Rodrigues(rvec)
        robject = Rotation.from_matrix(rot_matrix)
        euler_angles = robject.as_euler('xyz', degrees=True)
        euler_angles = [0, 0, 0] 
        if self.debug:
            print('TRANSFORMED TO EULER Z\"Y\'X')
        return euler_angles

    def generate_json(self, rect, rot):
        rot = [round(rot[0], 6),
               round(rot[1], 6),
               round(rot[2], 6),
               round(rot[3], 6)]
        rect = np.multiply(1000,rect)
        rect = np.around(rect).astype(int)
        jl = [
            [
                [rect[0], rect[1], rect[2]],
                [rot[0], rot[1], rot[2]],
                rot[3]
            ]
        ]
        if self.debug:
            print('GENERATED JSON: ', jl)
        return json.dumps(jl).encode('ascii')

    def generate_xml(self, rect, rot):
        if len(rot) == 4:
            rot = Rotation.from_quat(rot).as_euler('XYZ', degrees=True)
            print('EULER ANGLE: ', rot)
        rect = np.multiply(1000,rect)
        rect = np.around(rect).astype(int)
        rot = np.around(rot, 6)
        robot_data = Element('Kinect')
        pattern = SubElement(robot_data,
                             'Pattern',
                             {'P':'0'})
        pattern.text = ' '
        coor_xml = SubElement(robot_data,
                               'Origin1',
                               {'X':str(rect[0]),
                                'Y':str(rect[1]),
                                'Z':str(rect[2]),
                                'A':str(rot[2]),
                                'B':str(rot[1]),
                                'C':str(rot[0]),
                                'BAG':'0'})
        coor_xml.text = ' '
        return ElementTree.tostring(robot_data, encoding='ascii')

    def transform_robot_base(self, rect):
        if rect.ndim != 1:
            return self.transform_robot_base(np.squeeze(rect))
#        coor = np.asarray([-rect[1], -rect[0], -rect[2]])
#        coor = np.asarray([rect[2], -rect[0], -rect[1]])
        coor = np.asarray([0, -rect[0], rect[1]])
        if self.debug:
            print('TRANSLATION VECTOR RELATIVE TO ROBOT: \n', coor)
#        refp = [-0.139, 0.02046, -0.5125]
        refp = [0, 0, 0]
        coor = np.asarray([refp[0] * np.sign(coor[0]) + 1.0 * np.sign(coor[0]) * (abs(coor[0]) - refp[0]),
                           refp[1] * np.sign(coor[1]) + 1.0 * np.sign(coor[1]) * (abs(coor[1]) - refp[1]),
                           refp[2] * np.sign(coor[2]) + 1.0 * np.sign(coor[2]) * (abs(coor[2]) - refp[2])])
        if self.debug:
            print('SCALED TRANSLATION VECTOR RELATIVE TO ROBOT: \n', coor)
        return coor

    def send2robot(self, a_tvec, euler_angles):
        if self.robot == 'KUKA':
            robot_data = self.generate_xml(a_tvec, euler_angles)
        elif self.robot == 'ABB':
            robot_data = self.generate_json(a_tvec, euler_angles)
        print('SENDING DATA TO ROBOT.........')
        self.sock.send(robot_data)
        print('DATA SENT TO ROBOT: \n', robot_data)

    def main(self):
        self.initialize_output()
        self.classes = load_classes(parse_data_cfg(self.data)['names'])
        device = torch_utils.select_device(self.device)
        model = self.load_model(device)
        dataset = self.get_sources()
        colors = self.get_colors()
        t0 = time.time() 
        for path, im0s, vid_cap in dataset:
            t = time.time()
            if self.webcam:
                p, self.im0 = path[0], im0s[0]
            else:
                p, self.im0 = path, im0s
            self.save_path = str(Path(self.out) / Path(p).name)
            self.height, self.width = self.im0.shape[:2]

            if self.undistort:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (self.width, self.height), 1, (self.width, self.height))
                for i in range(len(im0s)):
                    im0s[i] = cv2.undistort(im0s[i], self.camera_matrix, self.dist_coefs, None, newcameramtx)

            img = self.yolov3_transform(im0s, device)
            g_rects = self.detect(img, model, colors)

            if self.view_img:
                self.put_status_bar()
                cv2.imshow(p, self.im0)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            if self.save_img:
                if dataset.mode != 'images':
                    self.put_status_bar()
                self.save_to_out(vid_cap)
            if len(g_rects) != 0:
                max_rect = self.get_xyedge_pos(g_rects)
                rvec, tvec = self.transform_pnp(max_rect)
                euler_angles = self.transform_euler_angles(rvec)
                a_tvec = self.transform_robot_base(tvec)
                if self.use_socket:
                    self.send2robot(a_tvec, euler_angles)
                    self.socket_recieve()
            if self.fps == 0:
                self.fps = 1/(time.time() - t)
            else:
                self.fps += (1 * 10 ** -1)*(1/(time.time() - t) - self.fps)
            print('COMPLETED IN {0} seconds'.format(time.time()-t))
        if self.save_img:
            print('SAVED VIDEO TO: {0}/sample.mp4'.format(self.out))
        print("ELAPSED TIME: {:.2f}".format(time.time() - t0))
        print("APPROXIMATE FPS: {:.2F}".format(self.fps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/egg_tray.data', help='data file path')
    parser.add_argument('--weights', type=str, default='weights/egg_tray.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--camera-params', type=str, default='camera_data.npy', help='intrinsic camera parameters path')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_false', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_false', help='display results')
    parser.add_argument('--disable-socket', action='store_false', help='disable socket')
    parser.add_argument('--test-socket', action='store_true', help='test socket on dummy server')
    parser.add_argument('--undistort', action='store_true', help='undistort images')
    parser.add_argument('--save', action='store_true', help='save video')
    parser.add_argument('--silent', action='store_false', help='no debug')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        Detect(opt).main()
