import cv2

import shutil
import glob
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from utils import video_to_frames, video_framerate, frames_to_video

class ScaleVideo:
    '''
    Scale video with Real-ESRGAN.
    input accepts a file
    '''

    def __init__(self, settings={}) -> None:
        self.settings = {
            'input': './test.mp4',
            'output': './test_upscaled.mp4',
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'netscale': 4,
            'outscale': 4,
            'suffix': 'scaled',
            'tile': 0,
            'tile_pad': 10,
            'pre_pad': 0,
            'face_enhance': True,
            'half': True,
            'block': 23,
            'alpha_upsampler': 'realesrgan',
            'ext': 'auto'
        }
        for key, value in settings.items():
            self.settings[key] = value

        self.settings['frames_output'] = './out'
        os.makedirs(self.settings['frames_output'], exist_ok=True)
        os.makedirs(self.settings['frames_output'] + '_upscaled', exist_ok=True)
        video_to_frames(self.settings['input'], self.settings['frames_output'])
        fps = video_framerate(self.settings['input'])

        print(f'Video framerate {fps}')

        if 'RealESRGAN_x4plus_anime_6B.pth' in self.settings['model_path']:
            self.settings['block'] = 6
        elif 'RealESRGAN_x2plus.pth' in self.settings['model_path']:
            self.settings['netscale'] = 2

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=self.settings['block'], num_grow_ch=32, scale=self.settings['netscale'])

        upsampler = RealESRGANer(
            scale=self.settings['netscale'],
            model_path=self.settings['model_path'],
            model=model,
            tile=self.settings['tile'],
            tile_pad=self.settings['tile_pad'],
            pre_pad=self.settings['pre_pad'],
            half=self.settings['half'])

        if self.settings['face_enhance']:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
                upscale=self.settings['outscale'],
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)

        if os.path.isfile(self.settings['frames_output']):
            paths = [self.settings['frames_output']]
        else:
            paths = sorted(glob.glob(os.path.join(self.settings['frames_output'], '*')))

        for idx, path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(path))
            print('Testing', idx, imgname)

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if max(h, w) > 1000 and self.settings['netscale'] == 4:
                import warnings
                warnings.warn('The input image is large, try X2 model for better performance.')
            if max(h, w) < 500 and self.settings['netscale'] == 2:
                import warnings
                warnings.warn('The input image is small, try X4 model for better performance.')

            try:
                if self.settings['face_enhance']:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=self.settings['outscale'])
            except Exception as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set tile with a smaller number.')
            else:
                if self.settings['ext'] == 'auto':
                    extension = extension[1:]
                else:
                    extension = self.settings['ext']
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                save_path = os.path.join(self.settings['frames_output'] + '_upscaled', f'{imgname}.{extension}')
                cv2.imwrite(save_path, output)

        # video output
        frames_to_video(self.settings['frames_output'] + '_upscaled', self.settings['output'], fps)
        shutil.rmtree(self.settings['frames_output'])
        shutil.rmtree(self.settings['frames_output'] + '_upscaled')

if __name__ == '__main__' :
    settings = {
        'input': './test.mp4',
        'output': './test_upscaled.mp4'
    }
    ScaleVideo(settings)
