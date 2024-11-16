# Manual TensorRT Conversion and Optimization - LLM Optimization

> This repo is an attempt to explore the TensorRT framework by NVIDIA.

**Objective: Convert an LLM to TensorRT without existing automated pipelines like TensorRT-LLM**


### LLM to TensorRT Conversion
Select any open-source LLM. Manually convert the chosen model to TensorRT without using TRT-LLM. Document the step-by-step conversion process, including
- Dynamic shape handling
- Making KV Cache static
- Network definition and layer creation
- Weight extraction and setting

### Benchmarking and Analysis
Compare this in the offline mode with HF transformers with FlashAttention and Frozen KV Cache. Provide a detailed analysis of performance improvements and any trade-offs.

### Deliverables:
Source code for both parts with comprehensive documentation. Technical report covering:
- Optimization strategies
- TensorRT conversion process and challenges
- Performance analysis and benchmarking results
- Recommendations for further optimization
Also add Jupyter notebook or script demonstrating benchmarking process and results.


### Note: Extra points for integrating this LLM with a simple DRY sampler implementatio

---
---

# OVERVIEW
INSERT IMAGE FLOW


# UNDERSTANDING CONCEPTS
> Given that almost all terms mentioned above are new for me 😅, let me provide notes on what all they mean. 

### ONNX
The ONNX Runtime library [Source](https://github.com/microsoft/onnxruntime) can load an exported ONNX model and perform inference. ONNX Runtime is optimised for running ONNX models efficiently, and it can take advantage of various hardware accelerators (e.g., GPU, TensorRT) to further improve the inference speed.


### Dynamic Shape Handling
In the context of language models, dynamic shapes refer to the ability to handle input sequences of varying lengths, rather than being limited to a fixed size. When you export the model to ONNX, you define the **`dynamic axes`** for the input tensors, such as `input_ids`, `attention_mask`, and `position_ids`. All other axes will be treated as static, and hence fixed at runtime. This **allows the ONNX model to accept inputs of different sequence lengths during inference, without the need to pre-define a maximum length**.

For example, the `input_ids` tensor has two dimensions: `batch_size` and `sequence_length`. By defining the `batch_size` dimension as dynamic (`{0: 'batch_size'}`), the ONNX model can accept input sequences of varying lengths, as long as the batch size remains the same. This is possible because when we set the dynamic shape of the input tensors, the **engine will automatically adjust its internal buffers** and computations to accommodate the new shape.

Source [HuggingFace Docs](https://huggingface.co/docs/optimum/en/exporters/onnx/package_reference/configuration#configuration-classes-for-onnx-exports)




### Weight Stripping Nice notes on model
Once the Tensor RT engine has been created, having the option to strip weights helps to create and optimize an engine without unnecessary weights. It is more fast and no duplicate weights are used. We use it while inferencing, when the engine is loaded and refit with onnx weights. I am not fully clear but it means though.  

Source [NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#weightless-build), [Official Repo](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/sample_weight_stripping)


### StaticCache
A Static Cache allows reusing the cached values from previous computations, rather than recomputing them for each new token. This significantly improves the inference speed of the model, as it avoids the overhead of creating and managing the cache dynamically.
The `StaticCache` class in the `transformers` library is used to initialize and manage the KV cache. **This class pre-allocates the cache tensors with a fixed size**, and the model can then efficiently access and update these cached values during inference.

However, I learnt that `StaticCache` by itself won't be faster -- it only shines together with `torch.compile`. [Source](https://github.com/huggingface/transformers/issues/33270#issuecomment-2444830657)

A comparision with the usual Dynamic Cache -

|                        | **Dynamic Cache**                                                                                 | **Static Cache**                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Purpose            | Automatically manages the cache for each forward pass. Suitable for single-pass inference where the cache is not reused across multiple calls. | Persists the cache across multiple forward passes. **Useful for generating sequences token by token, where the cache needs to be reused**. |
| Usage              | Typically used when you don't need to persist the cache between different inference calls.        | Requires manual management of the cache, ensuring that the cache is updated and reused correctly. |


> This understanding, along with [this Github issue](https://github.com/huggingface/transformers/issues/30670#issuecomment-2096809195) helps us to navigate into making token by token gerneation, instead of directly calling the `model.generate` method.


---
---

# THE PIVOT!
https://github.com/huggingface/transformers/issues/30670#issuecomment-2096809195. This. 
Why static cache and decode token by token
https://github.com/huggingface/transformers/issues/28981#issuecomment-2419754181 How we did for T5

---
---


# THE CODE



The dummy input tensors (`dummy_input_ids`, `dummy_attention_mask`, `dummy_position_ids`) are used to provide example inputs for the ONNX export process. The ONNX exporter needs these example inputs to understand the shape and data types of the model's inputs, so that it can properly define the input and output signatures of the ONNX model.


The wrapper you mentioned, `ModelWrapper`, is a simple class that defines the model's forward method explicitly. This is necessary because the original model's forward method expects a specific set of arguments, which may include optional or conflicting inputs (like `inputs_embeds`). By creating a wrapper, we can control the inputs and outputs more precisely for the ONNX export.



In large language models, the attention mechanism uses a cache to store the key and value tensors from previous attention computations. This cache is used to speed up the attention computation during text generation. 

For example, let's say you have a sequence of input tokens: "The quick brown fox". When the model processes the first token "The", it computes the key and value tensors for the attention mechanism. These tensors are then stored in the cache, along with the position of the token in the sequence (i.e., the **cache position**). 

When the model processes the next token "quick", it can reuse the cached key and value tensors for the previous token "The", instead of recomputing them. This is done by passing the cached key and value tensors, along with the new cache position (which is now 1), to the attention mechanism.

By reusing the cached values, the model can generate text much more efficiently, as it doesn't need to recompute the attention for each new token. This is why making the KV cache static is important - it allows the model to access and update the cache efficiently during inference.




# BENCHMARKING
Add tables




# On dynamic sizes not being handled well
The dynamic_axes in torch.onnx.export() should theoretically handle dynamic input shapes, but sometimes the internal operations (especially with the cache mechanism) can still expect fixed sizes. This is likely what's happening here


# Past Issues  with tinyllama
Even though this is not directly related to my issue, but surely there have been issues while converting small models like tinyllama to onnx. 
https://github.com/huggingface/optimum/issues/1606#issuecomment-1866507683



# Not all models are compatible with kv cache
https://github.com/huggingface/transformers/issues/28981!
It says llama and whisper only for now.

https://github.com/pytorch/pytorch/issues/74732

---

# Drawbacks of (blindly using!) TensorRT 

- Precision differences: TensorRT uses different numerical precision than PyTorch, which  lead to small differences in output of model. 

- Dynamic shapes: PyTorch models can have dynamic input shapes, meaning that input shape can vary from one inference to next. **TensorRT requires static input shapes**, meaning that the input shape must be known and fixed at time of engine creation. Input shape must be manually specified when creating the TensorRT engine.

- Memory usage: TensorRT engines require additional memory for storing intermediate results and optimization data. TensorRT also killed my kernels when I tried running on the CPU.

- TensorRT version: The version of TensorRT used for engine creation and inference should be compatible with the version of PyTorch used to create the original model. Otherwise, the conversion process may fail or the performance may be suboptimal.

Source [Hengtao Tantai's Blog](https://medium.com/@zergtant/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users-7cd4c30c97bc)


---

# Issues I faced

### With the ONNX export
By far the biggest issue I faced. 


### Versioning

<img src="readme-images/issues-tensorrt-1.png" alt="Local Image" width="800" height="150"/> \
However, installing these results in ERROR: No matching distribution found! This is because NVIDIA doesn't support anything before Version 8 now removed everything upto version 7 [check official releases!](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html). \
For a while it seemed like all the tutorials and blogs online were in version 7. Thankfully, it wasn't the case everywhere.


### The API keeps changing
Since this is a rapidly expanding library, frequent API changes make life slightly difficult. 
<img src="readme-images/issues-tensorrt-2.png" alt="Local Image" width="800" height="175"/>
<img src="readme-images/issues-tensorrt-3.png" alt="Local Image" width="800" height="150"/>


Here are a few attributes that caused by head to spin!
- `context_execute_async_v2` (deprecated) vs `context_execute_async_v3` (new)
- `engine.binding_is_input` (deprecated) vs `engine.get_tensor_mode` (new)
- `engine.max_batch_size` (deprecated) vs only supports the value 1 (new) 
- *a few more!*

Source [NVIDIA forums](https://forums.developer.nvidia.com/t/build-cuda-engine-throws-error/300198/3), [Github Issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/104), [More Issues](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/557#issuecomment-841523481)


### Not using local builds
Expectedly, the TRT library is built for GPUs. I resorted to using Colab and Kaggle. This had a few major downsides since I could not leverage popular libraries that would have made things simpler.

#### trtexec
The `trtexec` command line wrapper seems like a lightweight tool to convert `onnx` models to TensorRT format directly. However running it on Kaggle (even as a subprocess) doesn't seem possible. One has to build it from the repo and set correct path variables etc. Source [building trtexec.](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

#### onnx-tensorrt
Similarly, the `onnx-tensorrt` library allows to convert `onnx` models to TensorRT format directly. However, again, running on Kaggle systems doesnt seem possible :/ Building the repo locally seems to be the way which isn't possible for me. Source [building onnx-tensorrt.](https://github.com/onnx/onnx-tensorrt?tab=readme-ov-file#building)





# REFERENCES
The open-source community is amazing. Cannot praise them enough. Here are a few links that helped me on the way - 
- NVIDIA's [Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/quick-start-guide/index.html) and [Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#perform_inference_python) specially the inference sections, [Starter Notebooks](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/IntroNotebooks) but they used `trtexec`!, and their [repo](https://github.com/NVIDIA/TensorRT/tree/main/samples/python/sample_weight_stripping)
- Pytorch [docs](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export) 
- ONNX [docs](https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html)
- HuggingFace's [AMAZING blog](https://huggingface.co/docs/transformers/main/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile)
- Blogs on Medium by [Fateme Bafghi](https://medium.com/@fatemebfg/tensorrt-conversion-transforming-deep-learning-models-for-high-speed-inference-36548bdca46c) - great help to understand concepts, [Max Melichov](https://medium.com/@maxme006/how-to-create-a-tensorrt-engine-version-10-4-0-ec705013da7c), [Vilson Rodrigues](https://vilsonrodrigues.medium.com/a-friendly-introduction-to-tensorrt-building-engines-de8ae0b74038), [Hengtao Tanai](https://medium.com/@zergtant/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users-7cd4c30c97bc) - the benchmarking code is from his blogs!
- Github Repos by [Sithu Aung](https://github.com/sithu31296/PyTorch-ONNX-TRT/tree/master)