from pypylon import pylon
from pypylon import genicam

class BaslerCameraThread(Dataset):
    def __init__(self, sources='basler', img_size=None, half=False, load=True):
#        self.img_size = img_size
        self.half = half
        self.mode = 'images'
        self.sources = [sources]
        self.grabstrat = 'latest'
#        self.grabstrat = 'buffer'
        buffer_type = 'custom_value'
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            self.imgs = [None] * len(self.devices)
            self.cap = pylon.InstantCameraArray(len(self.devices))
            for i, cam in enumerate(self.cap):
                cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(self.devices[i]))
                print("Using device ", cam.GetDeviceInfo().GetModelName())
                cam.Open()
                if load:
                    nodeFile = "NodeMap" + str(i) + ".pfs"
                    if os.path.exists(nodeFile):
                        pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)
                    else:
                        pylon.FeaturePersistence.Save(nodeFile, cam.GetNodeMap())
                if img_size is not None:
                    height, width = img_size
                    cam.Width.SetValue(width)
                    cam.Height.SetValue(height)
                if self.grabstrat == 'latest':
                    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
                elif self.grabstrat == 'buffer':
                    if buffer_type == 'latest':
                        cam.OutputQueueSize = 1
                    elif buffer_type == 'onebyone':
                        cam.OutputQueueSize = cam.MaxNumBuffer.Value
                    elif buffer_type == 'custom_value':
                        cam.OutputQueueSize = 2
                    cam.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
                elif self.grabstrat == 'onebyone':
                    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
                grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
                while not grabResult.IsValid():
                    grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
                if grabResult.GetNumberOfSkippedImages():
                    print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    img0 = image.GetArray()
                    cameraContextValue = grabResult.GetCameraContext()
                    self.imgs[cameraContextValue] = img0
                    print("Camera ", cameraContextValue, ": ", self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        #        grabResult.Release()
                thread = Thread(target=self.update, args=([i, cam]),daemon=True)
                thread.start()
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())
    def update(self, index, cam):
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
#        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
        while not grabResult.IsValid():
            grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
        if grabResult.GetNumberOfSkippedImages():
            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            img0 = image.GetArray()
            cameraContextValue = grabResult.GetCameraContext()
#            self.imgs[cameraContextValue] = img0
            self.imgs[index] = img0
            print("Camera ", cameraContextValue, ": ", self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
#        grabResult.Release()
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
        return self.sources, img0, None
    def __len__(self):
        return 0

class BaslerCamera(Dataset):
    def __init__(self, sources='basler', img_size=None, half=False, load=True):
#        self.img_size = img_size
        self.half = half
        self.mode = 'images'
        self.sources = [sources]
        self.grabstrat = 'latest'
#        self.grabstrat = 'buffer'
        buffer_type = 'custom_value'
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            self.imgs = [None] * len(self.devices)
            self.cap = pylon.InstantCameraArray(len(self.devices))
            for i, cam in enumerate(self.cap):
                cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(self.devices[i]))
                print("Using device ", cam.GetDeviceInfo().GetModelName())
                cam.Open()
                if load:
                    nodeFile = "NodeMap" + str(i) + ".pfs"
                    if os.path.exists(nodeFile):
                        pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)
                    else:
                        pylon.FeaturePersistence.Save(nodeFile, cam.GetNodeMap())
                if img_size is not None:
                    height, width = img_size
                    cam.Width.SetValue(width)
                    cam.Height.SetValue(height)
            if self.grabstrat == 'latest':
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
            elif self.grabstrat == 'buffer':
                if buffer_type == 'latest':
                    self.cap.OutputQueueSize = 1
                elif buffer_type == 'onebyone':
                    self.cap.OutputQueueSize = self.cap.MaxNumBuffer.Value
                elif buffer_type == 'custom_value':
                    self.cap.OutputQueueSize = 2
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
            elif self.grabstrat == 'onebyone':
                self.cap.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
#        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        while not grabResult.IsValid():
            grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        if grabResult.GetNumberOfSkippedImages():
            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            img0 = image.GetArray()
            cameraContextValue = grabResult.GetCameraContext()
            self.imgs[cameraContextValue] = img0
            if len(self.devices) == 1:
                print("Camera ", cameraContextValue, ": ", self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
#        grabResult.Release()

        return self.sources, self.imgs, None

    def __len__(self):
        return 0

class SingleBaslerCamera(Dataset):
    def __init__(self, pipe=0, img_size=None, half=False, load=True):
#        self.img_size = img_size
        self.half = half
        self.mode = 'images'
        self.grabstrat = 'latest'
#        self.grabstrat = 'buffer'
        buffer_type = 'custom_value'
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            if pipe == 0:
                pipe = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.cap = pylon.InstantCamera(pipe)  
            self.cap.Open()
            print("Using device: ", self.cap.GetDeviceInfo().GetModelName())
            if load:
                nodeFile = "NodeMap.pfs"
                if os.path.exists(nodeFile):
                    pylon.FeaturePersistence.Load(nodeFile, self.cap.GetNodeMap(), True)
                else:
                    pylon.FeaturePersistence.Save(nodeFile, self.cap.GetNodeMap())
            if img_size is not None:
                height, width = img_size
                self.cap.Width.SetValue(width)
                self.cap.Height.SetValue(height)
#                # The parameter MaxNumBuffer can be used to control the count of buffers
#                # allocated for grabbing. The default value of this parameter is 10.
#                self.cap.MaxNumBuffer = 15
#                countOfImagesToGrab = 100
#                self.cap.StartGrabbingMax(countOfImagesToGrab)
            if self.grabstrat == 'latest':
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
            elif self.grabstrat == 'buffer':
                if buffer_type == 'latest':
                    self.cap.OutputQueueSize = 1
                elif buffer_type == 'onebyone':
                    self.cap.OutputQueueSize = self.cap.MaxNumBuffer.Value
                elif buffer_type == 'custom_value':
                    self.cap.OutputQueueSize = 2
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
            elif self.grabstrat == 'onebyone':
                self.cap.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
#        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        while not grabResult.IsValid():
            grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        if grabResult.GetNumberOfSkippedImages():
            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            img0 = image.GetArray()
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
#        grabResult.Release()
        path = 'webcam.jpg'

        return path, img0, None

    def __len__(self):
        return 0
