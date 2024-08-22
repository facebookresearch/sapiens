from mmpretrain.utils.dependency import WITH_MULTIMODAL

if WITH_MULTIMODAL:
    from .blip import *  # noqa: F401,F403
    from .blip2 import *  # noqa: F401,F403
    from .chinese_clip import *  # noqa: F401, F403
    from .clip import *  # noqa: F401, F403
    from .flamingo import *  # noqa: F401, F403
    from .llava import *  # noqa: F401, F403
    from .minigpt4 import *  # noqa: F401, F403
    from .ofa import *  # noqa: F401, F403
    from .otter import *  # noqa: F401, F403
else:
    from mmpretrain.registry import MODELS
    from mmpretrain.utils.dependency import register_multimodal_placeholder

    register_multimodal_placeholder([
        'Blip2Caption', 'Blip2Retrieval', 'Blip2VQA', 'BlipCaption',
        'BlipNLVR', 'BlipRetrieval', 'BlipGrounding', 'BlipVQA', 'Flamingo',
        'OFA', 'ChineseCLIP', 'MiniGPT4', 'Llava', 'Otter', 'CLIP',
        'CLIPZeroShot'
    ], MODELS)
