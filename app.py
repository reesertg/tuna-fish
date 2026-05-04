import torch,torchaudio
import soundfile
from codec import get_model
from dualAR import Generator
import gradio as gr

# Demo ----------------------------------------------------------------------------------------------------------------------------
def Demo(rank=0,weights_path="./s2-pro",max_context=1024,port=7560):
    fishcodec=get_model().eval().to(dtype=torch.float16).to(rank) # using bfloat16 in the decoder will result in significant background noise
    fishcodec.load_state_dict(torch.load(f"{weights_path}/codec.pth",map_location="cpu"),strict=False)
    # fishcodec=torch.compile(fishcodec) # compile codec maybe a litter quicker

    generator=Generator(rank,weights_path,quant="bfloat16",max_context=max_context)
    generator.prefill=torch.compile(generator.prefill,mode="default",fullgraph=True,dynamic=True) # default or max-autotune-no-cudagraphs
    generator.decode=torch.compile(generator.decode,mode="reduce-overhead",fullgraph=True) # reduce-overhead or max-autotune with fullgraph=True
    
    def process_upload(prompt_text,prompt_wave,text,temperature=1.0,top_p=0.95,top_k=50):
        # read wave
        wave,sr=soundfile.read(prompt_wave,always_2d=True); wave=torch.FloatTensor(wave).T # for torch >= 2.9.0
        if wave.size(0)>1: wave=wave.mean(dim=0,keepdim=True)
        if sr!=44100:wave=torchaudio.functional.resample(wave,orig_freq=sr,new_freq=44100)
        wave=wave.to(device=rank,dtype=torch.float16)

        # encode
        with torch.inference_mode():
            indices,feature_lengths=fishcodec.encode(wave[None])
            prompt_tokens=indices[:1,:,:feature_lengths[0]]

        # generate
        code=generator.generate(prompt_text,prompt_tokens,text,temperature,top_p,top_k)

        # decode
        with torch.inference_mode():
            z=fishcodec.quantizer.decode(code)
            audio=fishcodec.decoder(z)[0,0].float()

        return 44100,(audio.cpu().clamp(-1.0,1.0)*32767)[:,None].to(torch.int16).numpy()

    demo=gr.Interface(
        fn=process_upload,
        inputs=[
            gr.Textbox(label="Text Condition",placeholder="Input Text Condition here..."),
            gr.Audio(label="Acoustic Condition",type="filepath"),
            gr.Textbox(label="Prompt Text",placeholder="Input Prompt Text here..."),
            gr.Slider(label="temperature",minimum=0.1,maximum=2.0,step=0.01,value=1.0),
            gr.Slider(label="top-p",minimum=0.1,maximum=1.0,step=0.01,value=0.95),
            gr.Slider(label="top-k",minimum=1,maximum=100,step=1,value=50),
        ],
        outputs=gr.Audio(label="Result",type="numpy"),
        title="Fish Speech S2-Pro Inference",
        description="This is a demo of Fish Speech S2-Pro Inference. [model](https://huggingface.co/fishaudio/s2-pro), [github](https://github.com/fishaudio/fish-speech).",
        # examples="./examples",
    )
    demo.launch(server_port=port)

if __name__=="__main__":
    Demo(
        weights_path="./s2-pro",
        max_context=1024,
        port=7560
    )