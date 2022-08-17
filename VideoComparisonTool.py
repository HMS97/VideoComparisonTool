from moviepy.editor import *
import cv2
import copy
from path import Path
import os
import argparse
import numpy as np


class VCT():
    """
    Args:
        which_side: 'left' or 'right'
        zoom_point: the point to zoom in on
        source_path: the path to the source video
        target_path: the path to the target video
        source_text: the text to display on the zoomed source figure
        target_text: the text to display on the zoomed target figure
        rectangle_size: the size of the rectangle to display on the  figure
        transformed_re_size: the size of the transformed rectangle to display on the  oomed figure
        fps: the fps of the output video
        FIF: if False, there would not be figure in fgure
    """
    def __init__(self, image_size = (512,512), zoom_point = (200,200), output_name = 'output.mp4', source_text = 'input', target_text = 'proposed', text_thickness = 1, FIF = True, rectangle_size = None, transformed_re_size = None, fps = 20):
        self.source_frames = []
        self.target_frames = []
        self.zoom_point = zoom_point
        self.rectangle_size = rectangle_size if rectangle_size is not None else image_size[0]//10
        self.transformed_re_size = self.rectangle_size * 2
        self.fps = fps
        self.FIF = FIF
        self.output_name = output_name
        self.source_text = source_text
        self.target_text = target_text
        self.text_thickness = text_thickness

    def cal_frames_metirc(self,  source_folder = None, target_folder = None, metric_function = None, source_frames = None, target_frames = None):
        """
        Args:
            source_folder: the path to the source folder
            target_folder: the path to the target folder
            metric_function: the metric function to use
            source_frames: the source frames
            target_frames: the target frames
        """
        assert metric_function is not None, "The metric function is not defined"
        if source_folder is not None and target_folder is not None:
       
            source_path = Path(source_folder)
            target_path = Path(target_folder)
            assert source_path.isdir(), "The source path is not a directory"
            assert target_path.isdir(), "The target path is not a directory"
            source_path = sorted(source_path.files(), key = lambda x: x.stem)
            target_path = sorted(target_path.files(), key = lambda x: x.stem)
            assert len(source_path) == len(target_path), "The source and target video are not the same length"
            results = []
            for source_item, target_item in zip(source_path, target_path):
                source_item = cv2.imread(source_item)
                target_item = cv2.imread(target_item)
                results.append(metric_function(source_item, target_item))

        elif source_frames is not None and target_frames is not None:
            assert len(source_frames) == len(target_frames), "The source and target video are not the same length"
            results = []
            for source_item, target_item in zip(source_frames, target_frames):
                results.append(metric_function(source_item, target_item))
        print('the metric for this video is: ', np.mean(results))
        return np.mean(results)

    def cal_video_metirc(self,  source_path, target_path, metric_function = None):
        """
        Args:
            source_path: the path to the source video
            target_path: the path to the target video
            metric_function: the metric function to use
        """
        self.video2clip(source_path, target_path, needs_output = False)
        return self.cal_frames_metirc(metric_function = metric_function, source_frames = self.source_frames, target_frames = self.target_frames)


    def frames2video(self, images_folder, fps = 20, output_path = None):
        """
        Args:
            images_folder: the path to the folder containing the images
            fps: the fps of the output video
            output_path: the path to the output video
        """
        images_path = sorted(Path(images_folder).files(), key = lambda x: x.stem)
        middle_results = []
        for image in images_path:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            middle_results.append(image)

        self.clip2video(middle_results, fps = fps, output_path = output_path)



    def video2clip(self, source_path, target_path, flash_type = 'topdown', zoom_point = None, needs_output = True, resize = None):
        """
        Args:
            source_path: the path to the source video
            target_path: the path to the target video
            flash_type: the type of flash to use
            zoom_point: the point to zoom in on
            needs_output: if True, the output video will be saved
        """
        self.flash_type = flash_type
        source_path = Path(source_path)
        target_path = Path(target_path)
        assert  source_path.isfile(), "The source path is not a file"
        assert  target_path.isfile(), "The target path is not a file"
        source_clip = VideoFileClip(source_path) 
        target_clip = VideoFileClip(target_path) 
        # #alert if the video is not the same length
        self.source_frames = []
        self.target_frames = []
        for item in source_clip.iter_frames():
            self.source_frames.append(item)
        for item in target_clip.iter_frames():
            self.target_frames.append(item)    
        self.__align_videos__()

        if needs_output:
            results = []
            assert len(self.source_frames) == len(self.target_frames), "The source and target video are not the same length"
            results = self.__switch_flash__(results, zoom_point = zoom_point)
            if resize != None:
                results = [cv2.resize(i,resize) for i in results]
            self.clip2video(results)
        # return results
    def videos2clip(self, source_path, target_path, flash_type = 'topdown', zoom_dict = None):
        """
        Args:
            source_path: the path to the source video
            target_path: the path to the target video
            flash_type: the type of flash to use
            zoom_dict: the zoom point to zoom in on
        """
        self.flash_type = flash_type

        source_path = Path(source_path)
        target_path = Path(target_path)
        #assert the path is a directory
        assert source_path.isdir(), "The source path is not a directory"
        assert target_path.isdir(), "The target path is not a directory"
        results = []
        video_format = ['.mp4','.mov','.avi']
        # print([item.ext for item in source_path.files()])
        # print([item for item in source_path.files() if item.ext in video_format])
        for file_name in [item for item in source_path.files() if item.ext in video_format]:
            print('dealing with video', os.path.join(source_path,file_name.name))
            source_clip = VideoFileClip(os.path.join(source_path, file_name.name)) 
            target_clip = VideoFileClip(os.path.join(target_path, file_name.name)) 
            self.source_frames = []
            self.target_frames = []
            for item in source_clip.iter_frames():
                self.source_frames.append(item)
            for item in target_clip.iter_frames():
                self.target_frames.append(item) 
            self.__align_videos__()
            # if zoom_dict is None:
        
                # assert len(self.source_frames) == len(self.target_frames), f"The source and target video {file_name} are not the same length"
            try:
                zoom_point = zoom_dict[file_name.name] if zoom_dict is not None else None
            except:
                zoom_point = None

        results = self.__switch_flash__(results, zoom_point = zoom_point)
        self.clip2video(results)

    def __align_videos__(self):
        if len(self.source_frames) != len(self.target_frames):
            length_source = len(self.source_frames)
            length_target = len(self.target_frames)
            minimum_length = min(length_source,length_target)
            print('using the minimum length')
            self.source_frames =  self.source_frames[: minimum_length]
            self.target_frames =  self.target_frames[: minimum_length]
        if self.source_frames[0].shape !=  self.target_frames[0].shape:
            print('using the minimum size h w')
            source_size = self.source_frames[0].shape
            target_size = self.target_frames[0].shape
            newsize_h = min(source_size[0],target_size[0])
            newsize_w = min(source_size[1],target_size[1])
            self.source_frames = [cv2.resize(i,(newsize_h,newsize_w)) for i in self.source_frames]
            self.target_frames = [cv2.resize(i,(newsize_h,newsize_w)) for i in self.target_frames]

    def __switch_flash__(self,results:list, zoom_point = None):
        """
        Args:
            results: the list of frames to be switched
            zoom_point: the point to zoom in on

        """
        
        if self.flash_type == 'topdown':
            results.extend(self.__flash_topdown__(zoom_point = zoom_point))
        elif self.flash_type == 'vertical_left' or self.flash_type == 'vertical':
            results.extend(self.__flash_vertical__(which_side = 'left', zoom_point = zoom_point))
        elif self.flash_type == 'vertical_right':
            results.extend(self.__flash_vertical__(which_side = 'right', zoom_point = zoom_point))
        elif self.flash_type =='leftright':
            results.extend(self.__flash_leftright__(zoom_point = zoom_point))
        return results
    
    def clip2video(self,results:list, fps = None,  output_path = None):
        """
        Args:
            results: the list of frames to be switched
            fps: the fps of the output video
            output_path: the path to the output video
        """
        if output_path is None:
            output_path = 'output.mp4'
        if fps is None:
            fps = self.fps

        output_clip = ImageSequenceClip(results, fps=fps)
        print(f'writting video to {output_path}')
        output_clip.write_videofile(output_path)
        
        
    def __add_figure__(self, image, from_frame, text = 'proposed', index = 1, zoom_point = None, split = False ):
        h,w = image.shape[0],image.shape[1]
        if zoom_point is None:
            zoom_point_x,zoom_point_y = self.zoom_point
        else:
            zoom_point_x,zoom_point_y = zoom_point
      
        part = from_frame[int(zoom_point_y):int(zoom_point_y+self.rectangle_size),int(zoom_point_x):int(zoom_point_x+self.rectangle_size)]
        mask = cv2.resize(part, (int(self.transformed_re_size), int(self.transformed_re_size)), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        if index == 1:
            image[20:self.transformed_re_size+20,-self.transformed_re_size:]=mask
            image = cv2.rectangle(image,(w-self.transformed_re_size,20),(w,20+self.transformed_re_size),(0,255,255),3)
            image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            location = (w-self.transformed_re_size,20 + 70)
            image = cv2.putText(image, text,location,  cv2.FONT_HERSHEY_SIMPLEX,self.text_thickness, (255,255,255), 1,cv2.LINE_AA)

        else:
            image[20 + self.transformed_re_size*(index -1):+ self.transformed_re_size*(index -1) + self.transformed_re_size+20,-self.transformed_re_size:]=mask
            image = cv2.rectangle(image,(w-self.transformed_re_size,20 +  self.transformed_re_size*(index -1)),(w,20+self.transformed_re_size +  self.transformed_re_size*(index -1)),(0,255,255),3)
            if split:
                image = cv2.rectangle(image,(int(w/2)+zoom_point_x,zoom_point_y),(int(w/2)+zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            else:
                image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            # image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            location = (w-self.transformed_re_size,20 +  self.transformed_re_size*(index -1)+ 70)
            image = cv2.putText(image, text, location,  cv2.FONT_HERSHEY_SIMPLEX,self.text_thickness,  (255,255,255), 1,cv2.LINE_AA)

        return image




    def __flash_topdown__(self,  zoom_point = None):

        step = 0
        results = []
        for index,frame in enumerate(self.source_frames):
            h,w = frame.shape[0],frame.shape[1]
            step += int(h/len(self.source_frames))
            frame = self.source_frames[index].copy()
            frame[:step,:,:] = self.target_frames[index][:step,:,:]
            image = cv2.line(frame, (0,step), (w, step), (36, 255, 12), thickness=2, lineType=8)
            if self.FIF:
                image = self.__add_figure__(image, self.source_frames[index], text = self.source_text, zoom_point = zoom_point, index = 1)
                image = self.__add_figure__(image, self.target_frames[index], text = self.target_text, zoom_point = zoom_point, index = 2)
            results.append(image)
        return results


    
    def __flash_leftright__(self,  zoom_point = None):

        step = 0
        results = []
        for index,frame in enumerate(self.source_frames):
            h,w = frame.shape[0],frame.shape[1]
            step += int(w/len(self.source_frames))
            frame = self.source_frames[index].copy()
            frame[:,:step,:] = self.target_frames[index][:,:step,:]
            image = cv2.line(frame, (step,0), (step,w ), (36, 255, 12), thickness=2, lineType=8)
            if self.FIF:
                image = self.__add_figure__(image, self.source_frames[index], text = self.source_text, zoom_point = zoom_point, index = 1)
                image = self.__add_figure__(image, self.target_frames[index], text = self.target_text, zoom_point = zoom_point, index = 2)
            results.append(image)
        # print(len(results))
        return results
    
    

    def __flash_vertical__(self, which_side = 'left', zoom_point = None):
        results = []
        step = 0
        for index,frame in enumerate(self.source_frames):
            h,w = frame.shape[0],frame.shape[1]
            step += int(h//len(self.source_frames))
            # print(self.source_frames[index].shape)
            frame = self.source_frames[index].copy()
            # print(len(source_frames[index]), len(target_frames[index]), source_frames[index].shape)
            if which_side == 'left':
                frame[:,:int(w/2),:] = self.source_frames[index][:,:int(w/2),:]
                frame[:,int(w/2):,:] = self.target_frames[index][:,:int(w/2),:]
            else:
                frame[:,:int(w/2),:] = self.target_frames[index][:,int(w/2):,:]
                frame[:,int(w/2):,:] = self.source_frames[index][:,int(w/2):,:]
            if self.FIF:
                frame = self.__add_figure__(frame, self.source_frames[index], text = self.source_text, zoom_point = zoom_point, index = 1)
                frame = self.__add_figure__(frame, self.target_frames[index], text = self.target_text, zoom_point = zoom_point, index = 2, split = True)
            frame = cv2.line(frame, (int(w/2),0), (int(w/2), h), (36, 255, 12), thickness=2, lineType=8)
            results.append(frame)
        return results

# # args = parser.parse_args()
# if __name__ == '__main__':
#     def  CalPSNR(x,y,range = 255):
#         return 10*np.log10(range**2/np.mean((x-y)**2))

#     vct = VCT(source_text = 'input', target_text = 'output', zoom_point = (400,500), FIF = False)
#     # # set different zoom point for different videos
#     # zoom_dict = {}
#     # zoom_dict['indoor1.mp4'] = (800,800)
#     # # generate the videos in different folder
#     # vct.videos2clip('input_video', 'output_video', flash_type= 'topdown', zoom_dict = zoom_dict)
#     # vct.cal_video_metirc('/Users/huimingsun/Downloads/DAVIS/compare_videos/indoor1.mp4', '/Users/huimingsun/Downloads/DAVIS/compare_videos/indoor1_noise.mp4', metric_function = CalPSNR)
