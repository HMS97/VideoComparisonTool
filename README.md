
<p align="center">
  <img src="images/topdown.gif" width="60%" title="topdown">
</p>

<p align="center">
  <img src="images/vertical.gif" width="60%" title="vertial">
</p>



## What is it?

It is a video comparison tool to compare the raw video and processed video. At the top right of the image, there are  two figures from the raw video and the processed video which could show the detail and difference. Two mainly comparison methods are implemented, top down flash and vertial flash. The example is shown above. This tool can be used to compare two videos and the videos with same name in the different folders\.
  
## Support flash types
- topdown flash
- leftright flash
- vertical flash


## Installation
```
git clone https://github.com/wuchangsheng951/VideoComparisonTool.git
cd VideoComparisonTool
pip install -r requirements.txt
```

## Usage: Command line
```python
from VideoComparisonTool import VCT 

# compare two video
vct = VCT(source_text = 'input', target_text = 'output', image_size = (1080,1920), zoom_point = (400,500))
#output size
vct.video2clip('Findx3.mp4','Findx3.mp4', flash_type= 'vertical_left', resize = (1580,1920))


#vct = VCT(source_text = 'input', target_text = 'output', zoom_point = (400,500))
#set different zoom point for different videos
#zoom_dict = {}
#zoom_dict['indoor1.mp4'] = (800,800)
#compare the videos in different folder
#vct.videos2clip('input_video', 'output_video', flash_type= 'topdown', zoom_dict = zoom_dict)
#compare the two videos
#vct.video2clip('input.mp4','output.mp4', flash_type= 'vertical')

```

## License
> MIT License
