import os
import torch
import torchvision
from diffusers import StableDiffusionGLIGENPipeline

from PIL import Image

pipe = StableDiffusionGLIGENPipeline.from_pretrained("gligen/diffusers-inpainting-text-box", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

os.makedirs("images", exist_ok=True)

prompt = "a dog and a birthday cake"

images = pipe(
    prompt,
    num_images_per_prompt=2,
    gligen_phrases = ['a dog', 'a birthday cake'],
    gligen_inpaint_image=Image.open("resources/arg_corgis.jpeg").convert('RGB'),
    gligen_boxes = [
        [0.1871, 0.3048, 0.4419, 0.5562],
        [0.2152, 0.6792, 0.7671, 0.9482]
    ],
    gligen_scheduled_sampling_beta=1,
    output_type="numpy",
    num_inference_steps=50
).images

images = torch.stack([torch.from_numpy(image) for image in images]).permute(0, 3, 1, 2)

torchvision.utils.save_image(images, "images/inpaint_text_box.png", nrow=2, normalize=False)
