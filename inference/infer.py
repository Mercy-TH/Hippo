import time
import cv2
import onnxruntime
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import torch
from config.config import *

ONNX_SIM_MODEL_PATH = '../models/unetplusplus.onnx'
TENSORRT_ENGINE_PATH_PY = '../models/unetplusplus_py.engine'
BINDING_INDEX_NUMBER = 2


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def get_numpy_data():
    img_path = "/opt/projects/image_algorithm/src/segment/coco/images/000000000036/img/000000000036.jpg"
    # img_path = '/opt/projects/image_algorithm/src/segment/coco/cats/000000191016/img/000000191016.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    return img


def load_engine(engine_file_path):
    trt_logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_file_path, 'rb') as f:
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            print('load_engine ok.')
    return engine


def allocate_buffer(engine):
    binding_names = []
    for idx in range(BINDING_INDEX_NUMBER):
        bn = engine.get_binding_name(idx)
        if bn:
            binding_names.append(bn)
        else:
            break
    inputs = []
    outputs = []
    bindings = [None] * len(binding_names)
    stream = cuda.Stream()

    for binding in binding_names:
        binding_idx = engine[binding]
        if binding_idx == -1:
            print("Error Binding Names!")
            continue
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[binding_idx] = int(device_mem)

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def post_processing(outputs):
    outputs = outputs.reshape((3, 128, 128))
    outputs[outputs >= 0.5] = 255
    outputs[outputs < 0.5] = 0
    outputs = torch.Tensor(outputs).numpy()
    outputs = np.transpose(outputs, (1, 2, 0))
    outputs = cv2.resize(outputs, (128, 128))
    cv2.imshow('res', outputs)
    cv2.waitKey()
    cv2.destroyAllWindows()


def infer(engine, data_input, input_bufs, output_bufs, bindings, stream):
    context = engine.create_execution_context()
    input_bufs[0].host = data_input
    cuda.memcpy_htod_async(
        input_bufs[0].device,
        input_bufs[0].host,
        stream
    )
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(
        output_bufs[0].host,
        output_bufs[0].device,
        stream
    )
    stream.synchronize()
    outputs = output_bufs[0].host.copy()
    return outputs


def run():
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    data_input = get_numpy_data()
    data_input = np.ascontiguousarray(data_input, dtype=np.float32)
    engine = load_engine(TENSORRT_ENGINE_PATH_PY)
    input_bufs, output_bufs, bindings, stream = allocate_buffer(engine)
    try:
        start = time.time()
        outputs = infer(engine, data_input, input_bufs, output_bufs, bindings, stream)
        end = time.time()
        time_use_trt = end - start
        print(f"TRT use time {time_use_trt} s , FPS={1 / time_use_trt}")
    finally:
        cuda_ctx.pop()
    return outputs


if __name__ == '__main__':
    trt_outputs = run()
    post_processing(trt_outputs)
