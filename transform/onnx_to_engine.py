import os
import tensorrt as trt

ONNX_SIM_MODEL_PATH = '../models/unetplusplus_new.onnx'
TENSORRT_ENGINE_PATH_PY = '../models/unetplusplus_new.engine'


def build_engine(onnx_file_path, engine_file_path):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, trt_logger)
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())
        # if not parser.parse(model.read()):
        #     print('ERROR: Failed to parse the ONNX file.')
        #     for error in range(parser.num_errors):
        #         print(parser.get_error(error))
        #     return None
    print("Completed parsing ONNX file")
    builder.max_batch_size = 1

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                  engine_file_path)

    print("Creating Tensorrt Engine")

    config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.max_workspace_size = 2 << 30
    # if builder.platform_has_fast_fp16:
    #     config.set_flag(trt.BuilderFlag.FP16)
    if trt.__version__[0] == '7':
        engine = builder.build_engine(network, config)  # tensorRT-7
    else:
        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    return engine


if __name__ == "__main__":
    build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)
