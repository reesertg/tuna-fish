import torch
import time,tqdm
from safetensors.torch import load_file
from tokenizers import Tokenizer

from torch.nn.attention.flex_attention import create_block_mask,flex_attention
create_block_mask=torch.compile(create_block_mask)

def apply_rope(x,cache):
    x_type,x=x.dtype,x.to(torch.float32).unflatten(-1,[x.shape[-1]//2,2])
    return torch.stack([x[...,0]*cache[...,0]-x[...,1]*cache[...,1],x[...,1]*cache[...,0]+x[...,0]*cache[...,1]],-1).flatten(-2).to(x_type)

class Encoder(torch.nn.Module):
    def __init__(self,C=2560,Cattn=4096,Cffn=9728,A=32,GQA=4,qk_norm=False,cache_len=1024,dtype=torch.bfloat16):
        super().__init__()
        self.wqkv,self.wo,self.w1,self.w3,self.w2=[torch.nn.Linear(i,o,bias=False,dtype=dtype) for i,o in [[C,int((1+2/GQA)*Cattn)],[Cattn,C],[C,Cffn],[C,Cffn],[Cffn,C]]]
        self.ffn_norm,self.attention_norm,self.q_norm,self.k_norm=[torch.nn.RMSNorm(i,eps=1e-6,dtype=dtype) if (qk_norm or i==C) else torch.nn.Identity() for i in (C,C,Cattn//A,Cattn//A)]
        self.cache_k,self.cache_v=[torch.nn.Buffer(torch.zeros((1,A//GQA,cache_len,Cattn//A),dtype=dtype),persistent=False) for _ in range(2)]

    def forward(self,x,input_pos,rope_cache,mask=None,pos=0):
        q,k,v=[i.unflatten(-1,[-1,rope_cache.shape[-2]*2]).transpose(1,2) for i in self.wqkv(self.attention_norm(x)).split([self.wo.in_features]+2*[(self.wqkv.out_features-self.wo.in_features)//2],dim=-1)]
        q,k=[apply_rope(i,rope_cache) for i in (self.q_norm(q),self.k_norm(k))]
        if mask!=None:
            self.cache_k[:,:,input_pos],self.cache_v[:,:,input_pos]=k,v; k,v=self.cache_k,self.cache_v
            x=x+self.wo(flex_attention(q,k,v,block_mask=mask,enable_gqa=(q.shape[1]!=k.shape[1])).transpose(1,2).contiguous().flatten(-2))
        else: # get numerical bug when using flex_attention with fastAR
            self.cache_k[:,:,pos:pos+q.shape[2],:],self.cache_v[:,:,pos:pos+q.shape[2],:]=k,v; k,v=self.cache_k[:,:,:pos+q.shape[2],:],self.cache_v[:,:,:pos+q.shape[2],:]
            x=x+self.wo(torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=(pos==0),enable_gqa=(q.shape[1]!=k.shape[1])).transpose(1,2).contiguous().flatten(-2))
        x_norm=self.ffn_norm(x); x=x+self.w2(torch.nn.functional.silu(self.w1(x_norm))*self.w3(x_norm))
        return x

class RQTransformer(torch.nn.Module):
    def __init__(self,vocab_size=155776,codebook_size=4096,num_codebooks=10,acoustic_size=1024,n_layer=36,fast_n_layer=4,C=2560,Cattn=4096,Cffn=9728,A=32,GQA=4,base=1e6,max_len=1024,dtype=torch.bfloat16):
        super().__init__()
        self.embeddings,self.fast_embeddings,self.fast_codebook_embeddings=[torch.nn.Embedding(vocab,C,dtype=dtype) for vocab in [vocab_size,codebook_size,codebook_size*num_codebooks]]
        self.layers,self.fast_layers=torch.nn.ModuleList(Encoder(C,Cattn,Cffn,A,GQA,True,max_len,dtype) for _ in range(n_layer)),torch.nn.ModuleList(Encoder(C,Cattn,Cffn,A,GQA,False,10,dtype) for _ in range(fast_n_layer))
        self.norm,self.fast_norm=[torch.nn.RMSNorm(C,eps=1e-6,dtype=dtype) for _ in range(2)]
        self.output,self.fast_output=[torch.nn.Linear(C,vocab,bias=False,dtype=dtype) for vocab in [codebook_size+1,acoustic_size]]
        self.rope_cache=torch.nn.Buffer(torch.stack([f(torch.tensor([[j*base**(-i/(Cattn//A/2)) for i in range(Cattn//A//2)] for j in range(max_len)])) for f in (torch.cos,torch.sin)],dim=-1)[None,None,:],persistent=False)
        self.rope_cache=self.rope_cache.to(torch.bfloat16).float() # fishspeech s2-pro was trained with pure bfloat16

    def load_weights(self,weight_path,shards=2,s=151678,e=155773,eos=151645,acoustic_size=1024):
        pth=dict(); [pth.update(load_file(f"{weight_path}/model-0000{i+1}-of-00002.safetensors",device="cpu")) for i in range(shards)]
        pth.update({"text_model.model.output.weight":torch.cat([pth["text_model.model.embeddings.weight"][s:e+1],pth["text_model.model.embeddings.weight"][eos:eos+1]],dim=0)})
        pth["audio_decoder.output.weight"]=pth["audio_decoder.output.weight"][:acoustic_size] # eliminate redundant computations
        self.load_state_dict({k.replace("text_model.model.","").replace("audio_decoder.","fast_").replace("attention.","").replace("feed_forward.",""):v for k,v in pth.items()})
    
    def sample(self,logits,temperature=1.0,top_p=0.95,top_k=50):
        logit_sort,idx_sort=logits.sort(dim=-1,descending=True)
        idx_mask=((logit_sort>=logit_sort[:,top_k])&(logit_sort.softmax(-1).cumsum(-1)<=top_p))|(logit_sort==logit_sort[:,0])
        logits=torch.where(idx_mask.scatter(dim=-1,index=idx_sort,src=idx_mask),logits,float("-Inf"))
        return ((logits/max(temperature,1e-5)).softmax(-1)/torch.empty_like(logits).exponential_(1)).argmax(-1,keepdim=True)

    # Repetition Aware + Random(top-k & top-p & Temperature) Sampling for semantic token, Random Sampling for acoustic token
    def forward(self,x,input_pos,previous_tokens,mask,temperature=1.0,top_p=0.95,top_k=50,temperature_high=1.0,top_p_high=0.9,RAS_num=1): 
        for layer in self.layers: 
            x=layer(x,input_pos,self.rope_cache[:,:,input_pos],mask)
        x=self.norm(x[:,-1:,:]) # correct, prefill flops: 2*(p_slow*toks+9*p_fast*1), 4070m FP8 MFU: 0.43
        logits=self.output(x)[:,0,:]
        
        # Repetition Aware Sampling allow 0 repetitions -> Hard Masking
        logits.scatter_(1,previous_tokens,float('-inf')); idx=self.sample(logits,temperature,top_p,top_k)

        # FishSpeech Repetition Aware Sampling
        # idx=self.sample(logits,temperature,top_p,top_k)
        # idx_high=self.sample(logits,temperature_high,top_p_high,top_k)
        # idx=torch.where((previous_tokens==idx).sum(1,keepdim=True)>=RAS_num,idx_high,idx)

        rvq=torch.zeros([x.shape[0],11],dtype=torch.int32,device=x.device)
        rvq[:,0],rvq[:,1]=idx[:,0]+151678,idx[:,0]
        
        x=torch.cat([x,self.fast_embeddings(rvq[:,1:2].clamp(max=4096-1))],dim=1) # concat hidden and semantic emb to prefill: 10 step -> 9 step
        for layer in self.fast_layers:
            x=layer(x,None,self.rope_cache[:,:,:2,:,:],None,0) # block_mask got bug here
        rvq[:,2]=self.sample(self.fast_output(self.fast_norm(x[:,-1,:])),temperature,top_p,top_k)
        for fast_pos in range(2,10):
            x=self.fast_embeddings(rvq[:,fast_pos:fast_pos+1])
            for layer in self.fast_layers: 
                x=layer(x,None,self.rope_cache[:,:,fast_pos:fast_pos+1,:,:],None,fast_pos)
            rvq[:,fast_pos+1]=self.sample(self.fast_output(self.fast_norm(x[:,-1,:])),temperature,top_p,top_k)
        return rvq

class Generator:
    def __init__(self,rank=0,weight_path="./s2-pro",quant="bfloat16",batch=1,max_context=1024,RAS_window=10,num_codebooks=10,codebook_size=4096):
        quants=dict(bfloat16=2,int8wo=1,fp8=1,int4wo=0.5,nvfp4=0.5)
        assert quant in quants.keys()
        self.model=RQTransformer(n_layer=36).eval().to(rank)
        self.model.load_weights(weight_path)
        p_slowAR=sum([p.numel() for n,p in self.model.named_parameters() if "embeddings" not in n and "fast" not in n])
        p_fastAR=sum([p.numel() for n,p in self.model.named_parameters() if "embeddings" not in n and "fast"     in n])
        print(f"params without Embeddings: slowAR {p_slowAR/1e9:.1f}B, fastAR {p_fastAR/1e6:.1f}M")
        self.model_size=(p_slowAR+p_fastAR*9+2*batch*1024*(36*max_context+4*10*9))*quants[quant] # (params w/o emb + kv_cache) * p.dtype.itemsize refer gpt-fast
        
        if quant!="bfloat16":
            from torchao.quantization.quant_api import quantize_,Int8WeightOnlyConfig as INT8WO,Float8DynamicActivationFloat8WeightConfig as FP8,PerRow,Int4WeightOnlyConfig as INT4WO
            from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig as NVFP4
            aoconfigs=dict(int8wo=INT8WO(),fp8=FP8(granularity=PerRow()),int4wo=INT4WO(128,True,"tile_packed_to_4d","hqq"),nvfp4=NVFP4())
            quantize_(self.model,aoconfigs[quant],lambda m,n: isinstance(m,torch.nn.Linear) and "output" not in n)
            torch.cuda.empty_cache()

        self.tokenizer=Tokenizer.from_file(f"{weight_path}/tokenizer.json")
        self.sp_token=[self.tokenizer.encode(s,add_special_tokens=False).ids for s in ["<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n<|speaker:0|>","\n\nSpeech:\n","<|im_end|>\n<|im_start|>user\n","<|im_end|>\n<|im_start|>assistant\n<|voice|>"]]

        self.rvqs=torch.zeros([batch,11,max_context+1],dtype=torch.int32,device=rank)
        self.previous_tokens=torch.zeros([batch,RAS_window],dtype=torch.int32,device=rank)
        self.acoustic_bias=torch.tensor([0]+(num_codebooks-1)*[codebook_size],device=rank).cumsum(0)[None,:,None]
        self.Ptoken,self.PTime,self.DSpeed=0,1e-9,0
        self.max_context,self.quant=max_context,quant

    def tokenize(self,prompt_text,prompt_tokens,text,rvqs,semantic_start_token_id=151678):
        prompt_text_token,text_token=[self.tokenizer.encode(s,add_special_tokens=False).ids for s in [prompt_text,text]]
        code0=torch.tensor(self.sp_token[0]+prompt_text_token+self.sp_token[1]+(prompt_tokens[0,0,:]+semantic_start_token_id).tolist()+self.sp_token[2]+text_token+self.sp_token[3],dtype=torch.int32)
        prompt_start=sum([len(s) for s in [self.sp_token[0],prompt_text_token,self.sp_token[1]]])
        rvqs.zero_(); rvqs[0,0,:len(code0)]=code0; rvqs[0,1:,prompt_start:prompt_start+prompt_tokens.shape[-1]]=prompt_tokens[0]
        return len(code0)

    # add "padding_idx" to "embeddings" and replace "scale_codebook_embeddings" to "norm" maybe better, just in one line: x=norm(emb(x).sum(2)) 
    def prefill(self,rvqs,input_pos,previous_tokens,temperature=1.0,top_p=0.95,top_k=50):
        if not hasattr(self,"decode_mask"): self.decode_mask=create_block_mask(mask_mod=lambda b,h,q,kv: q>=kv,B=1,H=1,Q_LEN=self.max_context,KV_LEN=self.max_context,device=rvqs.device)
        mask=create_block_mask(mask_mod=lambda b,h,q,kv: q>=kv,B=1,H=1,Q_LEN=rvqs.shape[-1],KV_LEN=self.max_context,device=rvqs.device)
        is_code=(rvqs[:,0,:]>=151678)&(rvqs[:,0,:]<=155773)
        x=self.model.embeddings(rvqs[:,0,:])+self.model.fast_codebook_embeddings(rvqs[:,1:,:]+self.acoustic_bias).sum(1)*is_code.unsqueeze(-1)
        x[is_code]=x[is_code]*(10+1)**-0.5
        return self.model(x,input_pos,previous_tokens,mask,temperature,top_p,top_k)
    
    # refer to https://github.com/meta-pytorch/gpt-fast using flex-decoding instead of sdpa + static kv_cache + mask to keep fullgraph
    def decode(self,rvqs,input_pos,previous_tokens,temperature=1.0,top_p=0.95,top_k=50):
        mask=self.decode_mask[:,:,input_pos//self.decode_mask.BLOCK_SIZE[0]]
        mask.mask_mod=lambda b,h,q,kv: self.decode_mask.mask_mod(b,h,q+input_pos[0],kv)
        mask.seq_lengths=(1,self.max_context)
        x=(self.model.embeddings(rvqs[:,0,:])+self.model.fast_codebook_embeddings(rvqs[:,1:,:]+self.acoustic_bias).sum(1))*(10+1)**-0.5
        return self.model(x,input_pos,previous_tokens,mask,temperature,top_p,top_k)

    def generate(self,prompt_text,prompt_tokens,text,temperature=1.0,top_p=0.95,top_k=50):
        prompt_length=self.tokenize(prompt_text,prompt_tokens,text,self.rvqs)
        with torch.inference_mode():
            pos=prompt_length
            pbar=tqdm.tqdm(initial=pos,total=self.max_context,desc=f"Q:{self.quant}, P:{self.Ptoken/self.PTime:.1f}tok/s {self.PTime:.3f}s, D:{self.DSpeed:.1f}tok/s {self.DSpeed*self.model_size/1024**3:.1f}GB/s, Context")
            
            input_pos=torch.arange(0,pos,device=self.rvqs.device)
            torch.cuda.synchronize(); time_start=time.perf_counter()
            self.rvqs[:,:,pos]=self.prefill(self.rvqs[:,:,input_pos],input_pos,self.previous_tokens,temperature,top_p,top_k)
            self.previous_tokens=self.previous_tokens.roll(-1,dims=1); self.previous_tokens[:,-1:]=self.rvqs[:,1:2,pos]
            torch.cuda.synchronize(); time_end=time.perf_counter(); self.PTime=time_end-time_start; self.Ptoken=pos; time_start=time.perf_counter()

            input_pos=torch.tensor([pos],dtype=torch.int32,device=self.rvqs.device)
            while pos<self.max_context and self.rvqs[:,1,pos]!=4096:
                self.rvqs[:,:,pos+1]=self.decode(self.rvqs[:,:,input_pos],input_pos,self.previous_tokens,temperature,top_p,top_k)
                input_pos+=1 ;pos+=1; pbar.update(1)
                self.previous_tokens=self.previous_tokens.roll(-1,dims=1); self.previous_tokens[:,-1:]=self.rvqs[:,1:2,pos]
            torch.cuda.synchronize(); time_end=time.perf_counter(); self.DSpeed=(pos-self.Ptoken-1)/(time_end-time_start); pbar.close()
        return self.rvqs[:,1:,prompt_length:pos]

if __name__=="__main__":
    # use codec togather if memory enough

    # quantize + compile may take 5-30 minutes for JIT, use quant="bfloat16" by default as possible
    generator=Generator(rank=0,weight_path="./s2-pro",quant="fp8",max_context=1024)
    generator.prefill=torch.compile(generator.prefill,mode="default",fullgraph=True,dynamic=True) # default or max-autotune-no-cudagraphs
    generator.decode=torch.compile(generator.decode,mode="reduce-overhead",fullgraph=True) # reduce-overhead or max-autotune with fullgraph=True
    # change mode to "max-autotune-no-cudagraphs" and "max-autotune" if generate speed is slow on some old gpus like 3090
    
    prompt_tokens=torch.frombuffer(bytearray(open("./examples/prompt/paimon.code","rb").read()),dtype=torch.int16).view(1,10,-1).to(torch.int32)
    prompt_text="Uhh, Paimon doesn't need any help in that department. But if Albedo wants to pay Paimon back for helping him, a few Mora might settle the score. Tee-hee!"
    
    text_list_zh=[
        "我今年三十七岁。现在，我正坐在波音七四七的机舱里。这架硕大无比的飞机正穿过厚厚的乌云层往下俯冲，准备降落在汉堡机场。",
        "十一月冷冽的雨淹得大地一片雾蒙蒙的。穿着雨衣的整修工、整齐划一的机场大厦上竖着的旗、BMW的大型广告牌，这一切的一切看来都像是法兰德斯派画里阴郁的背景。唉！又来到德国了。",
        "这时，飞机顺利着地，禁烟灯号也跟着熄灭，天花板上的扩音器中轻轻地流出BGM音乐来。正是披头士的“挪威的森林”，倒不知是由哪个乐团演奏的。一如往昔，这旋律仍旧撩动着我的情绪。不！远比过去更激烈地撩动着我、摇撼着我。",
        "为了不叫头脑为之迸裂，我弓着身子，两手掩面，就这么一动不动。不久，一位德籍的空中小姐走了过来，用英文问我是不是不舒服，我答说不打紧，只是有点头晕而已。",
        "真的不要紧吗？不要紧，谢谢你！我说道。于是她带着微笑离开，这时，扩音器又放出比利乔的曲子。抬起头，我仰望飘浮在北海上空的乌云，一边思索着过去的大半辈子里，自己曾经失落了的。思索那些失落了的岁月，死去或离开了的人们，以及烟消云散了的思念。",
        "在飞机完全静止下来，人们纷纷解开安全带，开始从柜子里取出手提包、外套时，我始终是待在那片草原上的。我嗅着草香、聆听鸟鸣，用肌肤感受着风。那是在一九六九年秋天，我就要满二十岁的时候。",
        "就算在十八年后的今天，那片草原风光也仍旧历历在目。绵延数日的霏霏细雨冲走了山间光秃秃的地表上堆积的尘土，漾出一股深邃的湛蓝，而十月的风则撩得芒草左右摇曳，窄窄长长的云又冻僵了似的紧偎着蔚蓝的天空。",
        "天空高踞顶上，只消定睛凝视一会，你便会感到两眼发痛。风吹过草原，轻拂着她的发，然后往杂树林那头遁去。树叶沙沙作响，远处几声狗吠。那声音听来有些模糊，仿佛你正立在另一个世界的入口一般。",
        "除此以外，再没有别的声响。不管是什么声响都无法进入我们的耳里。再没有人会和我们错身而过，只看到两只鲜红的鸟怯生生地从草原上振翅飞起，飞进杂树林里。一边踱着步，直子便一边跟我聊起那口井来了。",
        "记忆这玩意儿真是不可思议。当我身历其境时，我是一点儿也不去留意那风景。当时我并不觉得它会让人留下深刻的印象，也绝没料到在十八年后，我可能将那一草一木记得这么清楚。",
    ]

    text_list_en=[
        "I am thirty-seven years old this year. Now, I am sitting in the cabin of a Boeing 747. This massive plane is diving through a thick layer of dark clouds, preparing to land at Hamburg Airport.",
        "The cold rain in November flooded the earth with mist. The repairman wearing a raincoat, the neatly arranged flags on the airport building, and the large BMW billboard all look like the gloomy background in Flemish paintings. Ah! We have arrived in Germany again.",
        "At this moment, the plane landed smoothly, the no smoking light went out, and BGM music gently flowed out from the loudspeaker on the ceiling. It is The Beatles' 'Forest of Norway, I don't know which band played it. As always, this melody still stirs my emotions. no Touching and shaking me even more fiercely than before.",
        "In order to prevent my mind from bursting, I arched my body, covered my face with both hands, and remained motionless. Shortly after, a German flight attendant walked over and asked me in English if I was feeling unwell. I replied that it didn't matter, just a little dizzy.",
        "Is it really okay? It's okay, thank you! I said. So she left with a smile, and at this moment, the loudspeaker played Billy Joel's song again. Raising my head, I looked up at the dark clouds floating over the North Sea, pondering what I had lost in the past half of my life. Reflect on those lost years, those who have died or left, and the vanished memories.",
        "When the plane came to a complete stop and people began to unfasten their seat belts and take out their handbags and jackets from the cabinets, I remained on that grassland. I smell the fragrance of grass, listen to bird songs, and feel the wind with my skin. That was in the autumn of 1969, when I was about to turn twenty years old.",
        "Even today, eighteen years later, the grassland scenery is still vividly remembered. The drizzling rain that lasted for several days washed away the dust accumulated on the bare surface of the mountains, creating a deep blue. The October wind stirred the grass to sway left and right, and the narrow and long clouds clung tightly to the blue sky as if frozen.",
        "The sky stands high above, and if you stare intently for a moment, you will feel a pain in your eyes. The wind blew through the grassland, gently brushing her hair, and then fled towards the woods. The leaves rustled and a few dogs barking in the distance. The sound sounded a bit blurry, as if you were standing at the entrance of another world.",
        "Other than that, there was no other sound. No matter what sound it is, it cannot enter our ears. No one will ever cross paths with us again, only two bright red birds timidly flapping their wings and flying into the mixed forest from the grassland. Walking along, Naoko started chatting with me about the well.",
        "Memory is truly incredible. When I experience it firsthand, I don't pay any attention to the scenery. At that time, I didn't think it would leave a deep impression on people, and I never expected that eighteen years later, I might remember every blade of grass and every tree so clearly.",
    ]

    for i,text in enumerate(text_list_zh):
        codes=generator.generate(prompt_text,prompt_tokens,text,temperature=0.7,top_p=0.7,top_k=30)
        open(f"./examples/generate/output{i}_zh.code","wb").write(codes.cpu().to(torch.int16).numpy().tobytes())

    for i,text in enumerate(text_list_en):
        codes=generator.generate(prompt_text,prompt_tokens,text,temperature=0.7,top_p=0.7,top_k=30)
        open(f"./examples/generate/output{i}_en.code","wb").write(codes.cpu().to(torch.int16).numpy().tobytes())
