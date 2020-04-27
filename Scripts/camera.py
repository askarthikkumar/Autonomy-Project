import numpy as np
import matplotlib.pyplot as plt
import pyrep

class Camera(object):
    def __init__(self, machine):
        self.cameras={'front':machine.env._scene._cam_front,\
                    'shoulder_left':machine.env._scene._cam_over_shoulder_left,\
                    'wrist':machine.env._scene._cam_wrist,\
                    'shoulder_right':machine.env._scene._cam_over_shoulder_right,\
                    'wrist_mask':machine.env._scene._cam_wrist_mask}
        self.set_cameras_params(self.cameras,(128,128))

    #Resolution
    def set_cameras_params(self,cameras,resolution=(1024,1024)):
        for camera in cameras:
            cameras[camera].set_resolution(resolution)
        cameras["wrist"].set_render_mode(pyrep.const.RenderMode(0))
        cameras["front"].set_render_mode(pyrep.const.RenderMode(0))
    #requires a reset after changing camera parameters TODO

    def plot_img(self,cam):
        img_rgb=cam.capture_rgb()
        img_d=cam.capture_depth()
        sleep(0.5)
        if(not(np.any(img_rgb[:,:,1]) or np.any(img_rgb[:,:,2]))): #mask image
            img_rgb=img_rgb[:,:,0]
        plt.imshow(img_rgb);plt.title('Color image');plt.show()
        plt.imshow(img_d,cmap='Greys');plt.title('Depth image');plt.show()
        return img_rgb

    def segment_img(self,image,window=200):
        img_roi=image[:,window:-window]
        return img_roi

    def get_desired_label(self,img_roi,label_id=0):
        labels=np.unique(img_roi)
        labels_size=[ (len(np.argwhere(img_roi==label)),label) for label in labels] #tuple of labels and their size
        min_size=20000;max_size=200000
        labels_size.sort()
        labels_size=np.array(labels_size)
        desired_label=labels[np.where((labels_size[:,0]>min_size) & (labels_size[:,0]<max_size))[0]]
        if(desired_label is not None):
            desired_label=labels_size[label_id,1] #Change 0 to other number TODO:fix this
        return desired_label

    def binarise_image(self,image,desired_label):
        img_bin=(image==desired_label) * image
        img_bin[img_bin==0]=1
        img_bin=img_bin.astype(np.uint8)
        return img_bin
        
    def get_processed_image(self,label_id=0):
        cam_fm=self.cameras['wrist_mask']
        img_mask_rgb=cam_fm.capture_rgb()[:,:,0]
        img_roi=segment_img(img_mask_rgb)
        desired_label=get_desired_label(img_roi,label_id)
        img_bin=binarise_image(img_mask_rgb,desired_label)
        return img_bin
