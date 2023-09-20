import os
# os.system('pip install "modelscope" --upgrade -f https://pypi.org/project/modelscope/')
os.system('pip install gradio_client==0.2.7')
os.system('pip install compel --upgrade')
os.system('pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git')
os.system('pip install -U git+https://github.com/huggingface/diffusers.git@main')
import time
import re
import pathlib

import requests
import gradio as gr

import torch

from PIL import Image

from requests import get
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
from PIL import Image


from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
   
)
#from modelscope import snapshot_download
from RealESRGAN import RealESRGAN
from compel import Compel


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}
Model_path_MAP={#your model path
    #"GhostMix": "wyj123456/GhostMix",
    "chilloutmix":"wyj123456/chilloutmix",
    #"Realistic": "wyj123456/Realistic_Vision_V5.1_noVAE",
    "RevAnimated_v11": "wyj123456/RevAnimated_v11"   
}
Lora_MAP={#your lora path
    "护肤美妆": "lora/hufu.safetensors",
    "国风茶饮": "lora/chayin.safetensors"

}

def inference(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    seed: int = -1,
    sampler="DPM++ Karras",
    num_inference_steps=30,
    model_path="chilloutmix",
    lora_path="护肤美妆",
    lora_w=0.8,
    ):
    #model_dir_sd = snapshot_download(Model_path_MAP[model_path],revision='v1.0.0')

    pipe = StableDiffusionPipeline.from_pretrained(
        Model_path_MAP[model_path],
        #vae=vae,
        #safety_checker=None,
        torch_dtype=torch.float32
    ).to("cuda")
    #lora_path="lora/hufu.safetensors"
    pipe.unload_lora_weights()
    pipe.load_lora_weights(Lora_MAP[lora_path])
    #lora_w = 0.8
    pipe._lora_scale = lora_w

    state_dict, network_alphas = pipe.lora_state_dict(
        Lora_MAP[lora_path]
    )

    for key in network_alphas:
        network_alphas[key] = network_alphas[key] * lora_w

    #network_alpha = network_alpha * lora_w
    pipe.load_lora_into_unet(
        state_dict = state_dict
        , network_alphas = network_alphas
        , unet = pipe.unet
    )

    pipe.load_lora_into_text_encoder(
        state_dict = state_dict
        , network_alphas = network_alphas
        , text_encoder = pipe.text_encoder
    )
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
    #pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    out = pipe(
        prompt_embeds=compel_proc(prompt),
        negative_prompt_embeds=compel_proc(negative_prompt),
        guidance_scale=float(guidance_scale),
        height=768, 
        width=512,
        #controlnet_conditioning_scale=[float(controlnet_conditioning_scale1),float(controlnet_conditioning_scale2)],
        num_inference_steps=int(num_inference_steps)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    image=out.images[0]
    image = image.convert('RGB')

    sr_image = model.predict(image)
    return sr_image
    #return out.images[0]

import gradio as gr


with gr.Blocks() as demo:
  guide_markdown=("""
   
    在选项卡2:商品图生成，这个模块里，我为您提供了两种风格产品的商品图生成，分别是护肤美妆风格和国风茶饮风格，下面我将为您说明操作流程：
                  

    🌈 放在最前面：在选项卡2下方，我已经为您对两种风格分别提供了三个生成示例，&#x1F4A1;您可以直接点击示例，💡再点击运行即可获得我为您提供的示例商品图。
    
    
                                         
    🌈🌈 模型选择和提示词：
                  
        🍓护肤美妆风格：基模型选择chilloutmix，lora风格模型选择护肤美妆，prompt和negative prompt可以参考示例1，2，3进行填写，需要注意的是在prompt中：
                  
        触发词可以用瓶子bottle，或者化妆品Cosmetic，自然场景就打flowers,plant,water,dead wood,等等，
                  
        渐变背景Gradient background，简单背景Simple background,  俯视用from above，模糊景深效果打blurry background。
                  
        🍓国风茶饮风格：基模型选择RevAnimated_v11，lora风格模型选择国风茶饮，prompt和negative prompt可以参考示例4，5，6进行填写，需要注意的是在prompt中：
                  
        触发词：(paper cup),(still life photography)，(Surrealist dream style)，(iron chain)
                  
                  
    🌈🌈🌈 参数部分：
                  
                🍓lora scale一般在0.6～0.8比较合适，值的大小与lora对图像的影响程度成正比关系。
                   
                🍓 Guidance Scale一般在7附近，它代表了提示词对图像的控制力度。
                  
                🍓sampler在这里一般选择DPM++ Karras，当然您可以自由切换，观察不同效果。
                  
                🍓steps在这里一般在20到35之间。
                  
                🍓seed是一个可以发挥您创造性的参数，您可以尝试不同的数字，得到不同的图像结果，当它为-1时，程序将会为您随机选择一个数字。
    
    
                  
    🌈🌈🌈🌈 重要的补充说明：
                  
                🍓在提示词中，您可以这样操作来改变单词的权重，也就是单词在生成图像里的重要性：
                  
                  1.在单词后面添加+，或者-，分别对应增加和减少权重，并且：+等同于1.1，++等同于1.2，-等同于0.9，--等同于0.8，以此类推。

                  2.可以参考示例中给单词加上括号，再直接填写数字。

                🍓您可以通过调节提示词，参数，直到获得您满意的图片。

                🍓基于该程序，可以添加其他更多的lora风格，用来丰富和细节刻画商品图，此功能正在开发。
                  
                🍓如需要生成某特定品牌的商品图，一种方式是将生成的图片后期ps，换上品牌商品，
                
                  另外与本项目相符合的做法是根据特定的品牌商品，训练一个特定的lora模型，目前对lora的训练也很方便。
                 
                🍓如果您遇到任何问题，可以通过邮箱 220221113@seu.edu.cn 联系到我。
                  
    🌈🌈🌈🌈🌈 示例商品图展示：           
                
        
""")
  with gr.Tab('选项卡1:操作说明书'):
      with gr.Row():
            with gr.Column():
                gr.Markdown(guide_markdown)
      with gr.Row():
            with gr.Column():#your image path
                gr.Image(value="image/image-3.jpg").style(height=768,width=512)
                gr.Image(value="image/image-4.jpg").style(height=768,width=512)
                gr.Image(value="image/image-5.jpg").style(height=768,width=512)
            with gr.Column():
                gr.Image(value="image/image-6.jpg").style(height=768,width=512)
                gr.Image(value="image/image-7.jpg").style(height=768,width=512)
                gr.Image(value="image/image-8.jpg").style(height=768,width=512)
                
  with gr.Tab('选项卡2:商品图生成'):
    with gr.Row():
        with gr.Column():
            model_path = gr.Dropdown(choices=list(
                        Model_path_MAP.keys()), value="chilloutmix", label="基模型",info="生成图像的基础模型")
            lora_path=gr.Dropdown(choices=list(
                        Lora_MAP.keys()), value="护肤美妆", label="lora风格模型",info="控制生成不同产品的商品图，不同产品对应不同的lora模型")
            prompt = gr.Textbox(
                    label="Prompt",
                    info="正向提示词(Prompt that guides the generation towards）",
                )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="low quality,",
                info="负面提示词(Prompt that guides the generation away from)",
            )

            with gr.Accordion(
                    label="参数：生成的图像很大程度上受下面详细参数的影响(Params: The generated QR Code functionality is largely influenced by the parameters detailed below)",
                    open=True,
            ):
                lora_w = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.8,
                    label="Lora Scale",
                    info="lora模型对图像生成的引导量(Controls the amount of guidance the lora model guides the image generation)"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=25.0,
                    step=0.25,
                    value=7,
                    label="Guidance Scale",
                    info="控制文本提示引导图像生成的引导量(Controls the amount of guidance the text prompt guides the image generation)"
                )
                sampler = gr.Dropdown(choices=list(
                    SAMPLER_MAP.keys()), value="Euler a", label="Sampler")
                num_inference_steps=gr.Slider(minimum=0.0,
                    maximum=60.0,
                    step=5,
                    value=30,
                    label="steps",
                    info="step for the generation")
                seed = gr.Number(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                    info="随机数生成器的种子。随机种子设置为 -1(Seed for the random number generator. Set to -1 for a random seed)"
                )
            with gr.Row():
                run_btn = gr.Button("运行")
        with gr.Column():
            result_image = gr.Image(label="Result Image", elem_id="result_image")
    run_btn.click(
        inference,
        inputs=[
            prompt,
            negative_prompt,
            guidance_scale,
            seed,
            sampler,
            num_inference_steps,
            model_path,
            lora_path,
            lora_w,
        ],
        outputs=[result_image],
    )

    gr.Examples(
        examples=[
            [
                "Bottle, close-up, flowers, Blurred background, Gray Background, Blurred Foreground, Simple Background, Plants, Skin care---",
                "worst quality+++, low quality+++, normal quality+++",
                7,
                972133707,
                "DPM++ Karras",
                30,
                "chilloutmix",
                "护肤美妆",
                0.8,
            ],
            [
                "bottle, branch, plant, nature, tree, blurry background, day, leaf,",
                "worst quality+++, low quality+++, normal quality+++",
                7,
                2103271826,
                "DPM++ Karras",
                30,
                "chilloutmix",
                "护肤美妆",
                0.8,
            ],
            [
                "cosmetic, ice, cosmetic cream, blue theme, day, blurry background,",
                "worst quality+++, low quality+++, normal quality+++",
                5.5,
                668000173,
                "DPM++ Karras",
                30,
                "chilloutmix",
                "护肤美妆",
                0.6,
            ],
            [
                "(Masterpiece, high quality, best quality, official art, beauty and aesthetics: 1.2), milk tea cup, surrounded by red rocks, splashing spray, (Chinese landscape paper carving, Chinese Song Dynasty landscape painting: 1.2), (surrealist dream style), cream organic fluid, light tracing, environmental shielding, hazy, natural light, limestone, gel resin sheet, oc rendering, (peach blossom forest background: 1.4),",
                "fog,nude,Paintings,sketches, (worst quality, low quality, normal quality)1.7, lowres, blurry, text, logo, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, strabismus, wrong finger, lowres, bad anatomy, bad hands, text,error,missing fingers,extra digit ,fewer digits,cropped,wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet, (worst quality, low quality)1.4",
                7,
                2912249350,
                "DPM++ Karras",
                20,
                "RevAnimated_v11",
                "国风茶饮",
                0.8,
            ],  
            [
                "(Masterpiece, High Quality, Best Quality, Official Art, Aesthetics and Aesthetics: 1.2), (Milk Tea Cup Surrounded by Ice: 1.2), Surrounded by Ice and Snow, Many Ice Blocks, Blue Theme, Surrealist Dream Style, Cream Organic++ Fluid, Ray Tracing, Foreground Occlusion, Natural Light, OC Rendering, Studio Lighting",
                "fog,nude, Paintings,sketches, (worst quality, low quality, normal quality)1.7, lowres, blurry, text, logo, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, strabismus, wrong finger, lowres, bad anatomy, bad hands, text,error,missing fingers,extra digit ,fewer digits,cropped,wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet, (worst quality, low quality)1.4",
                7,
                1357235544,
                "DPM++ Karras",
                30,
                "RevAnimated_v11",
                "国风茶饮",
                0.7,
            ], 
            [
                "(Masterpiece, high quality, best quality, official art, beauty and aesthetics:1.2),milk tea cup,surrounded by red rocks,splashing spray,(Chinese landscape paper carving, Chinese Song Dynasty landscape painting:1.2),(surrealist dream style),cream organic fluid,light tracing,environmental shielding,hazy,natural light,limestone,gel resin sheet,oc rendering",
                "fog,nude, Paintings,sketches, (worst quality, low quality, normal quality)1.7, lowres, blurry, text, logo, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, strabismus, wrong finger, lowres, bad anatomy, bad hands, text,error,missing fingers,extra digit ,fewer digits,cropped,wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet, (worst quality, low quality)1.4",
                7,
                836568600,
                "DPM++ Karras",
                30,
                "RevAnimated_v11",
                "国风茶饮",
                0.7,
            ], 


        ],
        fn=inference,
        inputs=[
            prompt,
            negative_prompt,
            guidance_scale,
            seed,
            sampler,
            num_inference_steps,
            model_path,
            lora_path,
            lora_w,
        ],
        outputs=[result_image],
        
    )

demo.queue().launch(debug=True)
