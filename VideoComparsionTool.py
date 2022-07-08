from moviepy.editor import *
import cv2
import copy
from path import Path
import os
import argparse



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
    """
    def __init__(self, zoom_point = (800,500), output_name = 'output.mp4', source_text = 'input', target_text = 'proposed', FIF = True, rectangle_size = 300, transformed_re_size = 500, fps = 20):
        self.source_frames = []
        self.target_frames = []
        self.zoom_point = zoom_point
        self.rectangle_size = rectangle_size
        self.transformed_re_size = transformed_re_size
        self.fps = fps
        self.FIF = FIF
        self.output_name = output_name
        self.source_text = source_text
        self.target_text = target_text


    def frames2video(self, images_folder, fps = 20, output_path = None):
        images_path = sorted(Path(images_folder).files(), key = lambda x: x.stem)
        middle_results = []
        for image in images_path:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            middle_results.append(image)

        self.clip2video(middle_results, fps = fps, output_path = output_path)



    def video2clip(self, source_path, target_path, flash_type = 'topdown', zoom_point = None):
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
        results = []
        assert len(self.source_frames) == len(self.target_frames), "The source and target video are not the same length"

        results = self.__switch_flash__(results, zoom_point = zoom_point)
        self.clip2video(results)

    def videos2clip(self, source_path, target_path, flash_type = 'topdown', zoom_dict = None):
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
            # if zoom_dict is None:
            assert len(self.source_frames) == len(self.target_frames), f"The source and target video {file_name} are not the same length"
            try:
                zoom_point = zoom_dict[file_name.name] if zoom_dict is not None else None
            except:
                zoom_point = None

        results = self.__switch_flash__(results, zoom_point = zoom_point)
        self.clip2video(results)

    def __switch_flash__(self,results:list, zoom_point = None):
        
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
        part = from_frame[zoom_point_y:zoom_point_y+self.rectangle_size,zoom_point_x:zoom_point_x+self.rectangle_size]
        mask = cv2.resize(part, (self.transformed_re_size, self.transformed_re_size), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        if index == 1:
            image[20:self.transformed_re_size+20,-self.transformed_re_size:]=mask
            image = cv2.rectangle(image,(w-self.transformed_re_size,20),(w,20+self.transformed_re_size),(0,255,255),3)
            image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            location = (w-self.transformed_re_size,20 + 70)
            image = cv2.putText(image, text,location,  cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255),2,cv2.LINE_AA)

        else:
            image[20 + self.transformed_re_size*(index -1):+ self.transformed_re_size*(index -1) + self.transformed_re_size+20,-self.transformed_re_size:]=mask
            image = cv2.rectangle(image,(w-self.transformed_re_size,20 +  self.transformed_re_size*(index -1)),(w,20+self.transformed_re_size +  self.transformed_re_size*(index -1)),(0,255,255),3)
            if split:
                image = cv2.rectangle(image,(int(w/2)+zoom_point_x,zoom_point_y),(int(w/2)+zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            else:
                image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            # image = cv2.rectangle(image,(zoom_point_x,zoom_point_y),(zoom_point_x+self.rectangle_size,zoom_point_y+self.rectangle_size),(0,255,255),3)
            location = (w-self.transformed_re_size,20 +  self.transformed_re_size*(index -1)+ 70)
            image = cv2.putText(image, text, location,  cv2.FONT_HERSHEY_SIMPLEX,2,  (255,255,255),2,cv2.LINE_AA)

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
        print(len(results))
        return results
    
    

    def __flash_vertical__(self, which_side = 'left', zoom_point = None):
        results = []
        step = 0
        for index,frame in enumerate(self.source_frames):
            h,w = frame.shape[0],frame.shape[1]
            step += int(h//len(self.source_frames))
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

# args = parser.parse_args()
if __name__ == '__main__':


    vct = VCT(source_text = 'input', target_text = 'output', zoom_point = (400,500), FIF = False)
    # set different zoom point for different videos
    zoom_dict = {}
    zoom_dict['indoor1.mp4'] = (800,800)
    # generate the videos in different folder
    vct.videos2clip('input_video', 'output_video', flash_type= 'topdown', zoom_dict = zoom_dict)
