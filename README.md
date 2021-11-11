# esrgan-upscale-video
upscale video with esrgan

# usage
`main.py`
`python main.py`
```
from main import ScaleVideo

settings = {
    'input': './test.mp4',
    'output': './test_upscaled.mp4',
    'netscale': 4,
    'outscale': 4
}
ScaleVideo(settings)
```

# reference
https://github.com/xinntao/Real-ESRGAN
