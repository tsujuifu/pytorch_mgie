
import os

from tqdm.auto import tqdm

from PIL import Image

import torch as T
import transformers, diffusers

from llava.conversation import conv_templates
from llava.model import *

import gradio as gr

def crop_resize(f, sz=512):
    w, h = f.size
    if w>h:
        p = (w-h)//2
        f = f.crop([p, 0, p+h, h])
    elif h>w:
        p = (h-w)//2
        f = f.crop([0, p, w, p+w])
    f = f.resize([sz, sz])
    return f
def remove_alter(s):  # hack expressive instruction
    if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:')+10:].strip()
    if '</s>' in s: s = s[:s.index('</s>')].strip()
    if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
    if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
    s = '.'.join([s.strip() for s in s.split('.')[:2]])
    if s[-1]!='.': s += '.'
    return s.strip()

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'
PATH_LLAVA = './_ckpt/LLaVA-7B-v1'

tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
model = LlavaLlamaForCausalLM.from_pretrained(PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=T.float16, use_cache=True).cuda()
image_processor = transformers.CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=T.float16)

tokenizer.padding_side = 'left'
tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
ckpt = T.load('./_ckpt/mgie_7b/mllm.pt', map_location='cpu')
model.load_state_dict(ckpt, strict=False)

mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end: tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

vision_tower = model.get_model().vision_tower[0]
vision_tower = transformers.CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=T.float16, low_cpu_mem_usage=True).cuda()
model.get_model().vision_tower[0] = vision_tower
vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
if mm_use_im_start_end: vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
image_token_len = (vision_config.image_size//vision_config.patch_size)**2

_ = model.eval()
EMB = ckpt['emb'].cuda()
with T.inference_mode(): NULL = model.edit_head(T.zeros(1, 8, 4096).half().to('cuda'), EMB)

pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=T.float16, safety_checker=None).to('cuda')
pipe.set_progress_bar_config(disable=True)
pipe.unet.load_state_dict(T.load('./_ckpt/mgie_7b/unet.pt', map_location='cpu'))

print('--init MGIE--')

def go_mgie(img, txt, seed, cfg_txt, cfg_img):
    img, seed = crop_resize(Image.fromarray(img).convert('RGB')), int(seed)
    inp = img

    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
    txt = "what will this image be like if '%s'"%(txt)
    txt = txt+'\n'+DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_PATCH_TOKEN*image_token_len+DEFAULT_IM_END_TOKEN
    conv = conv_templates['vicuna_v1_1'].copy()
    conv.append_message(conv.roles[0], txt), conv.append_message(conv.roles[1], None)
    txt = conv.get_prompt()
    txt = tokenizer(txt)
    txt, mask = T.as_tensor(txt['input_ids']), T.as_tensor(txt['attention_mask'])

    with T.inference_mode():
        out = model.generate(txt.unsqueeze(dim=0).cuda(), images=img.half().unsqueeze(dim=0).cuda(), attention_mask=mask.unsqueeze(dim=0).cuda(), 
                             do_sample=False, max_new_tokens=96, num_beams=1, no_repeat_ngram_size=3, 
                             return_dict_in_generate=True, output_hidden_states=True)
        out, hid = out['sequences'][0].tolist(), T.cat([x[-1] for x in out['hidden_states']], dim=1)[0]
        
        p = min(out.index(32003)-1 if 32003 in out else len(hid)-9, len(hid)-9)
        hid = hid[p:p+8]

        out = remove_alter(tokenizer.decode(out))
        emb = model.edit_head(hid.unsqueeze(dim=0), EMB)
        res = pipe(image=inp, prompt_embeds=emb, negative_prompt_embeds=NULL, 
                   generator=T.Generator(device='cuda').manual_seed(seed), guidance_scale=cfg_txt, image_guidance_scale=cfg_img).images[0]

    return res, out

def go_example(seed, cfg_txt, cfg_img):
    ins = ['make the frame red', 'turn the day into night', 'give him a beard', 'make cottage a mansion', 
           'remove yellow object from dogs paws', 'change the hair from red to blue', 'remove the text', 'increase the image contrast', 
           'remove the people in the background', 'please make this photo professional looking', 'darken the image, sharpen it', 'photoshop the girl out', 
           'make more brightness', 'take away the brown filter form the image', 'add more contrast to simulate more light', 'dark on rgb', 
           'make the face happy', 'change view as ocean', 'replace basketball with soccer ball', 'let the floor be made of wood']
    i = T.randint(len(ins), (1, )).item()
    
    return './_input/%d.jpg'%(i), ins[i], seed, cfg_txt, cfg_img

with gr.Blocks() as app:
    gr.Markdown('# Guiding Instruction-based Image Editing via Multimodal Large Language Models')
    with gr.Row(): inp, res = [gr.Image(height=384, width=384, label='Input Image', interactive=True), 
                               gr.Image(height=384, width=384, label='Goal Image', interactive=False)]
    with gr.Row(): txt, out = [gr.Textbox(label='Instruction', interactive=True), 
                               gr.Textbox(label='Expressive Instruction', interactive=False)]
    with gr.Row(): seed, cfg_txt, cfg_img = [gr.Number(value=13331, label='Seed', interactive=True), 
                                             gr.Number(value=7.5, label='Text CFG', interactive=True), 
                                             gr.Number(value=1.5, label='Image CFG', interactive=True)]
    with gr.Row(): btn_exp, btn_sub = [gr.Button('More Example'), gr.Button('Submit')]
    btn_exp.click(fn=go_example, inputs=[seed, cfg_txt, cfg_img], outputs=[inp, txt, seed, cfg_txt, cfg_img])
    btn_sub.click(fn=go_mgie, inputs=[inp, txt, seed, cfg_txt, cfg_img], outputs=[res, out])
    
    ins = ['make the frame red', 'turn the day into night', 'give him a beard', 'make cottage a mansion', 
           'remove yellow object from dogs paws', 'change the hair from red to blue', 'remove the text', 'increase the image contrast', 
           'remove the people in the background', 'please make this photo professional looking', 'darken the image, sharpen it', 'photoshop the girl out', 
           'make more brightness', 'take away the brown filter form the image', 'add more contrast to simulate more light', 'dark on rgb', 
           'make the face happy', 'change view as ocean', 'replace basketball with soccer ball', 'let the floor be made of wood']
    gr.Examples(examples=[['./_input/%d.jpg'%(i), ins[i]] for i in [1, 5, 8, 14, 16]], inputs=[inp, txt])
    
app.queue(concurrency_count=1), app.launch(server_port=7122)
