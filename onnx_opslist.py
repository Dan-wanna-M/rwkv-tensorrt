import numpy as np


class RWKVOnnxOps():

    def __init__(self, layers, embed, opsVersion = 15, useSafeWKV = True, externalData = True, splitExternalData = False,fp32inout=True, seq_mode=False, seq_length=256, quantized = False, *args, dtype=None, **kwargs):
        import onnx
        self.n_layers = layers
        self.n_embed = embed

        print("embed ", embed)

        dtype = onnx.TensorProto.FLOAT if dtype == np.float32 else onnx.TensorProto.FLOAT16 if dtype == np.float16 else onnx.TensorProto.BFLOAT16 if dtype == np.bfloat16 else onnx.TensorProto.FLOAT
        nptype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.float16 if dtype == onnx.TensorProto.FLOAT16 else np.float16 if dtype == onnx.TensorProto.BFLOAT16 else np.float32

        self.nm = 0
        exportname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{opsVersion}{'_unsafe' if not useSafeWKV else ''}{'_seq' if seq_mode else ''}.onnx"
        externalname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{opsVersion}{'_unsafe' if not useSafeWKV else ''}{'_seq' if seq_mode else ''}"

        # remove old files
        import os
        if os.path.exists(exportname):
            os.remove(exportname)
        if os.path.exists(externalname + ".bin"):
            os.remove(externalname + ".bin")

        self.TensorList = []
        self.NodeList = []

        self.useSafeWKV = useSafeWKV
        self.seq_mode = seq_mode
        self.seq_length = seq_length

        def initTensor(x, isfp32 = False, exname = ""):

            npdtype = np.float32 if (isfp32 and fp32inout) else nptype
            ddtype = onnx.TensorProto.FLOAT if (isfp32 and fp32inout) else dtype
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x).astype(npdtype)
            else:
                xx = x.squeeze().float().cpu().numpy()
                # convert to float32
                xx = xx.astype(npdtype)
            rrx = onnx.helper.make_tensor(
                name,
                ddtype,
                xx.shape,
                xx.tobytes(),
                raw=True
            )



            if externalData:
                if not splitExternalData:
                    exname = ""
                onnx.external_data_helper.set_external_data(
                    rrx,
                    location=externalname+exname+".bin",

                )

            self.TensorList.append(rrx)
            return name

        self.initTensor = initTensor

        def convertToFloat16(x):
            if x == None:
                return None
            name = f"convertToFloat16_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Cast',
                inputs=[x],
                outputs=[name],
                to=onnx.TensorProto.FLOAT16
            )
            self.NodeList.append(node)

            return name

        def convertToFloat32(x):
            if x == None:
                return None
            name = f"convertToFloat32_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Cast',
                inputs=[x],
                outputs=[name],
                to=onnx.TensorProto.FLOAT
            )
            self.NodeList.append(node)

            return name

        self.convertToFloat16 = convertToFloat16 if (dtype == onnx.TensorProto.FLOAT16 and fp32inout) else lambda x: x
        self.convertToFloat32 = convertToFloat32 if (dtype == onnx.TensorProto.FLOAT16 and fp32inout) else lambda x: x

        def sqrt(x):
            name = f"sqrt_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sqrt',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.sqrt = sqrt

        def mean(x):
            name = f"mean_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.mean = mean

        def relu(x):
            name = f"relu_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Relu',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.relu = relu

        def exp(x):
            name = f"exp_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Exp',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.exp = exp

        def stack(x, fp32 = False, exname = ""):
            return [initTensor(r, fp32, exname) for r in x]

        self.stack = stack

        def matvec(x, y, outputfp32 = False):
            name = f"matvec_{self.nm}_out"
            oname = f"matvec_g_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MatMul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)
            if outputfp32:
                return self.convertToFloat32(name)
            return name

        self.matvec = matvec

        def prod(x):
            name = f"prod_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=[x],
                outputs=[name],
                axes=[1],
                keepdims=0

            )
            self.NodeList.append(node)

            return name

        self.prod = prod

        def mul(x, y):
            name = f"mul_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Mul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.multiply = mul

        def squeeze(x):
            name = f"squeeze_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Squeeze',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.squeeze = squeeze

        def add(x, y):

            name = f"add_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Add',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.add = add

        def sub(x, y):
            name = f"sub_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sub',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.subtract = sub

        self.one = initTensor([1.0]*embed)
        self.margins = initTensor([0.00001]*embed, True)
        self.margins16 = initTensor([0.00001]*embed)
        

        def lerpx(x, y, z):
            return self.add(x, self.multiply(self.subtract(y, x), z))

        self.lerp = lerpx

        def minimum(x, y):
            name = f"minimum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Min',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.minimum = minimum
        # module def
        self.module = object

        def log(x):
            name = f"log_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Log',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.log = log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def divide(x, y):
            name = f"divide_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Div',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.divide = divide

        def layernorm17(x, w, b):
            name = f"layernorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=[x, w, b],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name 
        # ort 15 does not support layernorm

        def layernorm(x, w, b):
            xee2 = self.subtract(x,self.mean(x))
            x2 = self.add(self.sqrt(self.add(self.mean(self.multiply(xee2,xee2)), self.margins16)), self.margins16)
            return self.add(self.multiply(w, self.divide(xee2, x2)), b)


        self.layernorm = layernorm if opsVersion < 17 else layernorm17

        def getIndex(x, y):
            name = f"getIndex_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return squeeze(name)

        self.stackEmbed = False

        def neg(x):
            name = f"neg_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Neg',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.neg = neg

        def logistic(x):
            name = f"logistic_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sigmoid',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.logistical = logistic

        def maximum(x, y):
            name = f"maximum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Max',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.maximum = maximum

        self.getIndex = getIndex

        def unsqueeze(x, axes=0):
            axes_name = f"unsq_axes_init_{self.nm}"
            self.nm += 1
            axes_init_node = onnx.helper.make_tensor(
                axes_name,
                data_type=onnx.TensorProto.INT64,
                dims=(1,),
                vals=[axes]
            )
            name = f"unsqueeze_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "Unsqueeze",
                inputs=[x, axes_name],
                outputs=[name],
            )
            self.TensorList.append(axes_init_node)
            self.NodeList.append(node)
            return name
        self.unsqueeze = unsqueeze

        def slice(x, axes=0, starts=0, ends=-1):
            axes_name = f"slice_axes_init_{self.nm}"
            self.nm += 1
            axes_init_node = onnx.helper.make_tensor(
                axes_name,
                data_type=onnx.TensorProto.INT32,
                dims=(1,),
                vals=[axes]
            )
            start_name = f"slice_start_init_{self.nm}"
            self.nm += 1
            start_init_node = onnx.helper.make_tensor(
                start_name,
                data_type=onnx.TensorProto.INT32,
                dims=(1,),
                vals=[starts]
            )
            if ends is None:
                end_name = None
            else:
                end_name = f"slice_end_init_{self.nm}"
                self.nm += 1
                end_init_node = onnx.helper.make_tensor(
                    end_name,
                    data_type=onnx.TensorProto.INT32,
                    dims=(1,),
                    vals=[ends]
                )
            name = f"slice_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "Slice",
                inputs=[x, start_name, end_name, axes_name],
                outputs=[name],
            )
            self.TensorList.append(axes_init_node)
            self.TensorList.append(start_init_node)
            if end_name is not None:
                self.TensorList.append(end_init_node)
            self.NodeList.append(node)
            return name
        self.slice = slice

        def concate(x, y, axis=0):
            name = f"concate_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "Concat",
                inputs=[x, y],
                outputs=[name],
                axis=axis
            )
            self.NodeList.append(node)
            return name
        self.concate = concate

        def seq_concate(X, length, axis=0):
            name = f"seq_concate_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "Concat",
                inputs=[X[i] for i in range(length)],
                outputs=[name],
                axis=axis
            )
            self.NodeList.append(node)
            return name
        self.seq_concate = seq_concate

        # convert to float32
        self.emptyState = np.array((([[0.01]*embed, [0.01]*embed, [0.01]*embed, [
            0.01]*embed]+[[-1e30]*embed] if useSafeWKV else []))*layers)
        self.emptyState = np.array(self.emptyState)
        if dtype == onnx.TensorProto.FLOAT16 and not fp32inout:
            self.emptyState = self.emptyState.astype(np.float16)

        # self.zero = initTensor([0.0]*embed)

        def ppm(x):
            inputtensor = onnx.helper.make_tensor_value_info("input0",
                                                             onnx.TensorProto.INT32,
                                                             [1] if not self.seq_mode else [self.seq_length]), "input0"

            emptyState = list(map(lambda x: (onnx.helper.make_tensor_value_info("instate"+str(x),
                                                                                onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                                [embed]), "instate"+str(x)), range((4+useSafeWKV)*layers)))
            if self.seq_mode:
                outs = x.forwardSeq(
                    inputtensor[1], list(map(lambda x: x[1], emptyState)))
            else:
                outs = x.forward(
                    inputtensor[1], list(map(lambda x: x[1], emptyState)))
            print(self.TensorList.__len__())
            print(self.NodeList.__len__())
            print(outs)
            logits = onnx.helper.make_tensor_value_info(outs[0],
                                                        onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                        [50277] if not self.seq_mode else [self.seq_length, 50277])
            state = list(map(lambda x: onnx.helper.make_tensor_value_info(x,
                                                                          onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                          [embed]), outs[1]))

            # Create the graph (GraphProto)
            graph_def = onnx.helper.make_graph(
                nodes=self.NodeList,  # The list of nodes in the graph.
                name="RWKV",
                # Graph input

                inputs=[inputtensor[0], * \
                        list(map(lambda x:x[0], emptyState))],

                outputs=[logits, *state],  # Graph output

                initializer=self.TensorList,  # initializer

                # did not work, needs to be external

            )

            modelDef = onnx.helper.make_model(
                graph_def, producer_name="rwkvstic",

            )

            modelDef.opset_import[0].version = opsVersion

            onnx.save(modelDef, exportname)

            # run model
            print("Model saved to: ", exportname, " and is ready to be run")
            print("Data type: ", dtype)
            print("Embedding size: ", embed)
            print("Number of layers: ", layers)
            print("external data: ", externalname)
            exit()

        self.postProcessModule = ppm
