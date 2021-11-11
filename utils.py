import cv2
import moviepy.video.io.ImageSequenceClip

import os
import glob

def frames_to_video(out_folder, filename, fps):
    image_files = []

    for file in glob.glob(f'{out_folder}/*.png'):
        image_files.append(file)

    image_files.sort()

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(filename)

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
            print(f'{count}.png')
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

def video_framerate(video):
    video = cv2.VideoCapture(video);

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)

    video.release()

    return fps
