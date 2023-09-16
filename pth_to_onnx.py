import onnx_opslist
def RnnRWKV(ops: onnx_opslist.RWKVOnnxOps, *args):
    class myRWKV(ops.module):

        @ops.initfunc
        def __init__(self, w, seq_length=16, converted=False):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            self.ops = ops
            self.seq_length = seq_length
            self.postprocess0 = ops.initTensor((w["ln_out.weight"]))
            self.postprocess1 = ops.initTensor((w["ln_out.bias"]))
            self.postprocess2 = ops.initTensor((w["head.weight"]))
            self.emb = ops.initTensor(w["emb.weight"])
            if not converted:
                self.emb1 = ops.initTensor(w["blocks.0.ln0.weight"])
                self.emb2 = ops.initTensor(w["blocks.0.ln0.bias"])
            self.ln1w = (ops.stack(
                [w[f"blocks.{x}.ln1.weight"] for x in range(ops.n_layers)]))
            self.ln1b = (ops.stack(
                [w[f"blocks.{x}.ln1.bias"] for x in range(ops.n_layers)]))
            self.ln2w = (ops.stack(
                [w[f"blocks.{x}.ln2.weight"] for x in range(ops.n_layers)]))
            self.ln2b = (ops.stack(
                [w[f"blocks.{x}.ln2.bias"] for x in range(ops.n_layers)]))
            if not converted:
                self.time_decay = (ops.stack([
                    w[f"blocks.{x}.att.time_decay"].double().exp().neg() for x in range(ops.n_layers)], True))
            else:
                self.time_decay = (ops.stack([
                    w[f"blocks.{x}.att.time_decay"].double() for x in range(ops.n_layers)], True))
            self.time_first = (ops.stack([
                w[f"blocks.{x}.att.time_first"] for x in range(ops.n_layers)], True))
            self.kktk = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(ops.n_layers)]))
            self.vvtv = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(ops.n_layers)]))
            self.rrtr = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(ops.n_layers)]))
            self.key = (ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(ops.n_layers)], exname="_key"))
            self.value = (ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(ops.n_layers)], exname="_value"))
            self.receptance = (ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance"))
            self.outputvv = (ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(ops.n_layers)], exname="_outputvv"))
            self.time_mix_k_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(ops.n_layers)]))
            self.time_mix_r_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(ops.n_layers)]))
            self.key_ffn = (ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(ops.n_layers)], exname="_key_ffn"))
            self.receptance_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance_ffn"))
            self.value_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(ops.n_layers)], exname="_value_ffn"))

        def wkvsafe(self, k, v, xx, statee, stateb, statec):
            ww = ops.add(k, self.time_first[xx])
            p = ops.maximum(statee, ww)

            e1 = ops.exp(ops.subtract(statee, p))
            e2 = ops.exp(ops.subtract(ww, p))
            a = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            b = ops.add(ops.multiply(e1, statec), e2)
            ww = ops.add(statee, self.time_decay[xx])

            p = ops.maximum(ww, k)

            e1 = ops.exp(ops.subtract(ww, p))
            e2 = ops.exp(ops.subtract(k, p))
            outb = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            outc = ops.add(ops.multiply(e1, statec), e2)
            eee = p
            wkv = ops.divide(a, b)

            return ops.convertToFloat16(wkv), outb, outc, eee

        def wkvunsafe(self, k, v, xx, statee, stateb, statec):
            # // const double vv = v[i + token * emb];
            #     // const double wr1 = aa + exp(float(u[i + emb * offset] + w[i + emb * offset] + k[i + token * emb])) * vv;
            #     // const double wr2 = bb + exp(float(u[i + emb * offset] + w[i + emb * offset] + k[i + token * emb]));
            #     // y[i + token * emb] = (wr1) / (wr2+0.001);
            #     // y[i + token * emb] = (1.0 / (1.0 + exp(float(-r[i + token * emb])))) * y[i + token * emb];
            #     // aa = (aa + exp(float(double(k[i + token * emb]))) * vv) * exp(float(w[i + emb * offset]));
            #     // bb = (bb + exp(float(double(k[i + token * emb])))) * exp(float(w[i + emb * offset]));

            td = ops.exp(self.time_decay[xx])
            tf = ops.exp(self.time_first[xx])

            ek = ops.exp(k)
            ekk = ops.multiply(ek, tf)
            a = ops.add(stateb, ops.multiply(ekk, v))
            b = ops.add(statec, ekk)
            wkv = ops.divide(a, ops.add(b, ops.margins))

            outb = ops.add(stateb, ops.multiply(ek, v))
            outc = ops.add(statec, ek)

            outb = ops.multiply(td, outb)
            outc = ops.multiply(td, outc)

            eee = None

            return ops.convertToFloat16(wkv), outb, outc, eee

        @ops.layerdef
        def doLayer(self, x, statea, stateb, statec, stated, statee, xx):

            xy = ops.layernorm(x, self.ln1w[xx], self.ln1b[xx])

            # Time Mixing
            k = ops.matvec(
                self.key[xx], ops.lerp(statea, xy, self.kktk[xx]), True)

            v = ops.matvec(self.value[xx], ops.lerp(
                statea, xy, self.vvtv[xx]), True)
            rr = ops.matvec(
                self.receptance[xx], ops.lerp(statea, xy, self.rrtr[xx]))
            r = ops.logistical((rr))

            wkv, outb, outc, eee = self.wkvsafe(k, v, xx, statee, stateb, statec) if ops.useSafeWKV else self.wkvunsafe(
                k, v, xx, statee, stateb, statec)

            mvv = ops.add(x, ops.matvec(
                self.outputvv[xx], ops.multiply(r, wkv)))

            # Channel Mixing
            ddd = ops.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            km = ops.relu(ops.matvec(self.key_ffn[xx], ops.lerp(
                stated, ddd, self.time_mix_k_ffn[xx])))

            rt = ops.logistical((ops.matvec(self.receptance_ffn[xx], ops.lerp(
                stated, ddd, self.time_mix_r_ffn[xx]))))

            x = ops.add(mvv, ops.multiply(
                ops.matvec(self.value_ffn[xx], ops.multiply(km, km)), rt))

            return x, ops.convertToFloat32(xy), outb, outc, ops.convertToFloat32(ddd), eee # why not eee, ops.convertToFloat32(ddd)

        @ops.layerdef
        def doSeqLayer(self, x, statea, stateb, statec, stated, statee, i):

            # Time Mixing
            xx = ops.layernorm(x, self.ln1w[i], self.ln1b[i])
            statea = ops.concate(ops.unsqueeze(statea, 0), ops.slice(xx, axes=0, starts=0, ends=-1))
            k = ops.matvec(
                ops.lerp(statea, xx, self.kktk[i]),
                self.key[i],
                True
            )
            v = ops.matvec(
                ops.lerp(statea, xx, self.vvtv[i]),
                self.value[i],
                True
            )
            rr = ops.matvec(
                ops.lerp(statea, xx, self.rrtr[i]),
                self.receptance[i],
                True
            )
            r = ops.logistical((rr))
            state_list = []
            for t in range(self.seq_length):
                kk = ops.squeeze(ops.slice(k, axes=0, starts=t, ends=t+1))
                vv = ops.squeeze(ops.slice(v, axes=0, starts=t, ends=t+1))
                temp, stateb, statec, statee = self.wkvsafe(kk, vv, i, statee, stateb, statec)\
                      if ops.useSafeWKV else self.wkvunsafe(kk, vv, i, statee, stateb, statec)
                state_list.append(ops.unsqueeze(temp))
            statea = ops.seq_concate(state_list, self.seq_length)
            mvv = ops.add(x, ops.matvec(ops.multiply(r, statea), self.outputvv[i]))
            ret1 = ops.slice(xx, axes=0, starts=-1, ends=65535)#-2????

            # Channel Mixing
            ddd = ops.layernorm(mvv, self.ln2w[i], self.ln2b[i])
            stated = ops.concate(ops.unsqueeze(stated, 0), ops.slice(ddd, axes=0, starts=0, ends=-1))
            km = ops.relu(ops.matvec(ops.lerp(stated, ddd, self.time_mix_k_ffn[i]), self.key_ffn[i]))
            rt = ops.logistical((ops.matvec(ops.lerp(stated, ddd, self.time_mix_r_ffn[i]), self.receptance_ffn[i])))
            x = ops.add(mvv, ops.multiply(
                ops.matvec(ops.multiply(km, km), self.value_ffn[i]), rt))
            ret2 = ops.slice(ddd, axes=0, starts=-1, ends=65535)#-2????
            # why not eee, ops.convertToFloat32(ret2)
            return x, ops.convertToFloat32(ret1), stateb, statec, ops.convertToFloat32(ret2), statee


        @ ops.mainfunc
        def forwardSeq(self, x, state = None):
            if (state is None):
                state = ops.emptyState
            if converted:
                x = ops.getIndex(self.emb, x)
            else:
                x = ops.layernorm(
                    ops.getIndex(self.emb, x),
                    self.emb1, self.emb2)

            statea = state[0::(4+ops.useSafeWKV)]
            stateb = state[1::(4+ops.useSafeWKV)]
            statec = state[2::(4+ops.useSafeWKV)]
            stated = state[3::(4+ops.useSafeWKV)]
            statee = state[4::5] if ops.useSafeWKV else [None]*ops.n_layers

            ot = []
            for i in range(ops.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doSeqLayer(
                    x,
                    ops.convertToFloat16(statea[i]),
                    (stateb[i]),
                    (statec[i]),
                    ops.convertToFloat16(stated[i]),
                    (statee[i]),
                    i
                )
                ot = ot + ([( aaa), (bbb), (ccc), (ddd), (eee)] if ops.useSafeWKV else [( aaa), (bbb), (ccc), (ddd)])
            x = ops.matvec(ops.layernorm(x, self.postprocess0, self.postprocess1),
                           self.postprocess2)
            return ops.convertToFloat32(x), ot


        @ ops.mainfunc
        def forward(self, x, state = None):
            if (state is None):
                state = ops.emptyState
            if converted:
                x = ops.getIndex(self.emb, x)
            else:
                x = ops.layernorm(
                    ops.getIndex(self.emb, x),
                    self.emb1, self.emb2)
            statea = state[0::(4+ops.useSafeWKV)]
            stateb = state[1::(4+ops.useSafeWKV)]
            statec = state[2::(4+ops.useSafeWKV)]
            stated = state[3::(4+ops.useSafeWKV)]
            statee = state[4::5] if ops.useSafeWKV else [None]*ops.n_layers

            ot = []
            for i in range(ops.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                    x,
                    ops.convertToFloat16(statea[i]),
                    (stateb[i]),
                    (statec[i]),
                    ops.convertToFloat16(stated[i]),
                    (statee[i]),
                    i
                )

                ot = ot + ([( aaa), (bbb), (ccc), (ddd), (eee)] if ops.useSafeWKV else [( aaa), (bbb), (ccc), (ddd)])

            x = ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                            self.postprocess1))
            return ops.convertToFloat32(x), ot


    ops.postProcessModule(myRWKV(*args))



import torch

def convert_model(path, dtype):
    w = torch.load(path, map_location="cpu")
    dims = len(w["blocks.0.att.key.weight"])
    layers = len(list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))


    ops = onnx_opslist.RWKVOnnxOps(
        layers,
        dims,
        dtype=dtype,
        opsVersion=version,
        useSafeWKV=use_safe_wkv,
        externalData=use_external_data,
        splitExternalData=splitExternalData,
        fp32inout=fp32inout,
        seq_mode=seq_mode,
        seq_length=seq_length
    )

    RnnRWKV(ops, w, seq_length, converted)


import numpy as np
def convert():
    path = input_path
    dtype = np.float16 if use_fp16 else np.float32
    convert_model(path, dtype)

# Define the variables
input_path = r"/data/user/cangshui/tianchao/pth_models/RWKV-4-PilePlus-1B5-20230520-2942-486Gtokens-ctx4096-fp32-converted.pth"
if "convert" in input_path:
    converted = True
else:
    converted = False
use_fp16 = False
use_safe_wkv = True
use_external_data = True
splitExternalData = False
fp32inout = True
# version, number either 15/17
version = 17

# set seq mode and length
seq_mode = True
seq_length = 16
convert()
