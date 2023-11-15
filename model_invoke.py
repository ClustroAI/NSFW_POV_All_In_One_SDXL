from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights(
    "Remilistrasza/NSFW_LoRAs", 
    weight_name="NsfwPovAllInOneLoraSdxl-000009.safetensors", 
    revision="NSFW_POV_All_In_One_SDXL"
)

def invoke(input_text):
    prompt = input_text
    negative_prompt = '''worst quality, low quality, normal quality, 
    bad quality, poor quality, lowres, extra fingers, missing fingers, 
    poorly rendered hands, mutation, deformed iris, deformed pupils, 
    deformed limbs, missing limbs, amputee, amputated limbs, watermark
    '''
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=7,
        num_inference_steps=20,
        #generator=torch.manual_seed(3885156286),
        cross_attention_kwargs={"scale": 0.65}
    ).images[0]
    image.save("generated_image.png")
    return "generated_image.png"