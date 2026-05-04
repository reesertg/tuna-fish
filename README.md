# TUNA-FISH

### Minimal PyTorch Implementation of RQ-Transformer(Fish Speech S2-Pro...) Inference

![](./tunafish.png)

### Fish Speech S2-Pro    [Paper](https://arxiv.org/abs/2603.08823)    |    [GitHub](https://github.com/fishaudio/fish-speech)    |    [Hugging Face model](https://huggingface.co/fishaudio/s2-pro)

Complete rewrite of Fish Speech S2-pro Inference with around 150 lines of clean code. using [flex-decoding](https://github.com/meta-pytorch/gpt-fast) to achieve fullgraph in RQ-Transformer model decoding stage(typically slowAR decode + fastAR loop in fishspeech s2-pro). 

Featuring:
1. Very clean code, <150 lines of python for dualAR inference, <100 lines for pure RQ-Transformer model forward and prefill/decode
2. Achieved fullgraph in dualAR decode stage, and got considerable decoding speed on H20 GPU with 135 token/s
3. Clean dependence, only pytorch and safetensors/soundfile/tokenizers was needed
4. Support int8/fp8/int4/nvfp4 quantization. Due to the simplicity of the code, you can easily use various quantization methods in [torchao](https://github.com/pytorch/ao)


### Inference 

> [!IMPORTANT]
> **License Notice**  
> fishaudio s2-pro model weights are released under FISH AUDIO RESEARCH LICENSE. Please refer to [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) for more details.

```bash
uv pip install -U --pre torch torchao torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install safetensors tokenizers soundfile
# install mslk for torchao NVFP4 Quantized Inference  
# uv pip install -U --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu132

git clone https://github.com/reesertg/tuna-fish
cd tuna-fish

# get vocab codec and weights
# replace huggingface.co to hf-mirror.com if needed
mkdir ./s2-pro

wget https://huggingface.co/fishaudio/s2-pro/resolve/main/tokenizer.json -P ./s2-pro
wget https://huggingface.co/fishaudio/s2-pro/resolve/main/codec.pth -P ./s2-pro
wget https://huggingface.co/fishaudio/s2-pro/resolve/main/model-00001-of-00002.safetensors -P ./s2-pro
wget https://huggingface.co/fishaudio/s2-pro/resolve/main/model-00002-of-00002.safetensors -P ./s2-pro
```

1. encode prompt audio to code 
```bash
# Encode all audio files in the prompt directory
python codec.py encode
```

2. genente code 
```bash
# Generate RVQ codes based on the given prompt_text, prompt_tokens and text
# you can run fishspeech s2-pro on 8GB GPU
# quantize + compile may take several minutes at first run
# got prefill 5000~6000 token/s decode 26~27 token/s on RTX 4070 Laptop GPU with fp8 quantization and batch 1
python dualAR.py
```

3. decode code to audio
```bash
# Decode all RVQ codes in the generate directory
python codec.py
```

### web app
```bash
# You may need >16GB of GPU memory to load both codec and dualAR
python app.py
```

### Inference estimation

```bash
model:                slowAR              fastAR            dualAR
params w/o emb:       3644000256          406346240         3644000256+406346240*9
kv_cache:             2*B*L*C*T           2*B*L*C*T         2*1*36*1024*1024+2*1*4*1024*10*9
batch=1, max_length=1024, C_kv=1024, slowAR n_layers=36, fastAR n_layers=4
decode params load: 
(3644000256+406346240*9)+(2*1*36*1024*1024+2*1*4*1024*10*9)=7377351168, bfloat16 memory: 7377351168*2/1024**3 = 13.74 GB 
params w/o emb + buffer(static kv_cache only and no others, refer to https://github.com/meta-pytorch/gpt-fast)
```

### Inference speedtest

```bash
bfloat16:    3090: 48.1it/s|661.0GB/s    4090: 56.3it/s|773.6GB/s    5090: 94.8it/s|1302.7GB/s    H20: 135.6it/s|1863.3GB/s

memory use:  bfloat16: 9~10GB      int8wo: 6~7GB       fp8: 6~7GB       int4wo: 4~5GB       nvfp4: 4~5GB
4070m:       CUDA out of memory    int8wo: 24.5it/s    fp8: 26.8it/s    int4wo: 43.7it/s

# prefill 128, decode 896
4070m: Q:fp8, P:2861.5tok/s 0.045s, D:25.1tok/s 172.2GB/s, Context: 100%|██████████████████| 1024/1024 [00:35<00:00, 24.96it/s]
H20:   Q:bfloat16, P:8986.6tok/s 0.014s, D:135.0tok/s 1855.0GB/s, Context: 100%|██████████████████| 1024/1024 [00:06<00:00, 134.34it/s]
5090:  Q:bfloat16, P:9628.9tok/s 0.013s, D:91.4tok/s 1255.9GB/s, Context: 100%|██████████████████| 1024/1024 [00:09<00:00, 91.18it/s]

# dingzhen prompt, prefill 163, decode: ~280
Q:fp8, P:3416.6tok/s 0.048s, D:26.7tok/s 183.2GB/s, Context:  43%|████████▏          | 440/1024 [00:10<00:22, 26.23it/s]
Q:fp8, P:3354.9tok/s 0.049s, D:26.6tok/s 182.9GB/s, Context:  43%|████████▎          | 445/1024 [00:11<00:23, 24.41it/s]
Q:fp8, P:3399.0tok/s 0.048s, D:26.8tok/s 183.8GB/s, Context:  44%|████████▍          | 452/1024 [00:11<00:22, 25.76it/s]
```

### training estimation

```bash
slowAR tied    emb params: 4032298496
fastAR without emb params: 414210560

10 million hour data, 21hz audio codec, 4~5hz text BPE: 
semantic token: 1e7*3600*21/1e9=756 B, text token: 1e7*3600*5/1e9=180 B

32 x H100, BF16 989TFLOPS, Linear+Attention 0.6 MFU, training cost:
slowAR: toks=756e9+180e9; p=4032298496; L,C,T=32,4096,2048; FLOPs_slow=(6*p+12*L*C*T)*toks
fastAR: toks=756e9*10;    p=414210560;  L,C,T= 4,4096,10;   FLOPs_fast=(6*p+12*L*C*T)*toks
training cost: (FLOPs_slow+FLOPs_fast)/(32*989e12*0.6)/86400=27.1day
```

### Next Step
the bset TTS model you can get in ￥100 and 4hour

```bash
Data: 50000 hour audio and text, 21hz 10codebook RVQ audio codec, 4~5hz text BPE tokenizer
Temporal Transformer: params: 360M, training token: 50000*3600*(21+5)/1e9=4.68 B
Depth Transformer:    params: 40M,  training token: 50000*3600*21*10/1e9= 37.8 B
training FLOPS: 6*(360e6*4.68e9+40e6*37.8e9)=1.92e19

audodl, GPU: 1 x 5090, Transformer training speed(torch2.9.1+cu130):
model: L12 C1024 GQA1, param: 150M, fp8+Adamw,  TPS:233721tok/s | FLOPS:249TFLOPs  | MFU:118.6%
model: L16 C2048 GQA2, param: 750M, fp8+Adamw,  TPS:59982tok/s  | FLOPS:320TFLOPs  | MFU:152.8%

conservative estimate—assuming: 249T per gpu and an 8-card efficiency loss of 0.2
400M RQ-Transformer + 50000 hours data requires:
duration=6*(360e6*4.68e9+40e6*37.8e9)/(249e12*8*(1-0.2))/3600=3.34 hour
the price is 3.03/hour on autodl, which mean cost=3.34*8*3.03=81 yuan
```

## Credits

- [fish-speech](https://github.com/fishaudio/fish-speech), dualAR.py was writed refer to fish_speech/models/text2semantic, codec was modified from fish_speech/models/dac and normalized by LLM.
- [gpt-fast](https://github.com/meta-pytorch/gpt-fast), dualAR.py using flex-decoding in gpt-fast to achieve fullgraph in decode stage.


## Cite

If you find tuna-fish helpful in your research cite simply as:

```bibtex
@misc{tuna-fish,
  author = {Reese},
  title = {tuna-fish: Minimal PyTorch Implementation of RQ-Transformer Inference},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/reesertg/tuna-fish}
}
```

## License

MIT