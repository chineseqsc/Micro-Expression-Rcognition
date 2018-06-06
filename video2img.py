import cv2
import os

#import cv

file_path="/home/qsc/DEMO/VGGFace_TF/video_path.txt"
videos_save_path="/home/qsc/DEMO/CASME2_video2img"
videos_src_path="/home/qsc/DEMO/CASME2_videos/test"

for i in os.listdir(videos_src_path):
    sub_dir=os.path.join(videos_src_path,i)
    imgs_save_path=os.path.join(videos_save_path,i)
    #print(imgs_save_path)
    videos = os.listdir(sub_dir)
    #print(videos)
    videos = filter(lambda x: x.endswith('avi'), videos)
    for each_video in videos:
        #print(each_video)

        # get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        os.makedirs(imgs_save_path + '/' + each_video_name,exist_ok=True)

        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

        # get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(sub_dir, each_video)
        #print(each_video_full_path)
        cap  = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while(success):
            success, frame = cap.read()
            print('Read a new frame: ', success)
            if(frame_count % 40 == 0):
                cv2.imwrite(imgs_save_path +'/'+ each_video_name +'/'+ "%d.jpg" %(frame_count), frame, [int(cv2.IMWRITE_JPEG_QUALITY) , 100])
            frame_count = frame_count + 1

        cap.release()