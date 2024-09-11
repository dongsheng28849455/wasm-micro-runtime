/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wasi_nn_tensorflowlite.hpp"
#include "utils/logger.h"

#include "bh_platform.h"
#include "wasi_nn_types.h"
#include "wasm_export.h"

#if WASM_ENABLE_TFLITE_MICRO
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#else
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/error_reporter.h>

#if WASM_ENABLE_WASI_NN_GPU != 0
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

#if WASM_ENABLE_WASI_NN_EXTERNAL_DELEGATE != 0
#include <tensorflow/lite/delegates/external/external_delegate.h>
#endif
#endif

/* Maximum number of graphs per WASM instance */
#define MAX_GRAPHS_PER_INST 10
/* Maximum number of graph execution context per WASM instance*/
#define MAX_GRAPH_EXEC_CONTEXTS_PER_INST 10

#if WASM_ENABLE_TFLITE_MICRO != 0
/*
 * In order to use optimized tensorflow lite kernels, a signed int8_t quantized
 * model is preferred over the legacy unsigned model format. This means that
 * throughout this project, input images must be converted from unisgned to
 * signed format. The easiest and quickest way to convert from unsigned to
 * signed 8-bit integers is to subtract 128 from the unsigned value to get a
 * signed value.
*/
constexpr int scratchBufSize = 39 * 1024;

/* An area of memory to use for input, output, and intermediate arrays.*/
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;

/* Maybe we should move this to external*/
static uint8_t* wasi_nn_tensor_arena[MAX_GRAPH_EXEC_CONTEXTS_PER_INST] = {};

tflite::MicroMutableOpResolver<6> wasi_nn_micro_op_resolver;
#endif

typedef struct {
#if WASM_ENABLE_TFLITE_MICRO != 0
    /* tflite::MicroInterpreter do not have ctor for unique_ptr yet.*/
    tflite::MicroInterpreter* interpreter;
#else
    std::unique_ptr<tflite::Interpreter> interpreter;
#endif
} Interpreter;

typedef struct {
    char *model_pointer;
#if WASM_ENABLE_TFLITE_MICRO != 0
    /* tflite::Model do not have ctor for unique_ptr yet.*/
    tflite::Model* model;
#else
    std::unique_ptr<tflite::FlatBufferModel> model;
#endif
    execution_target target;
} Model;

typedef struct {
    uint32_t current_models;
    Model models[MAX_GRAPHS_PER_INST];
    uint32_t current_interpreters;
    Interpreter interpreters[MAX_GRAPH_EXEC_CONTEXTS_PER_INST];
    korp_mutex g_lock;
#if WASM_ENABLE_TFLITE_MICRO == 0
    TfLiteDelegate *delegate;
#endif
} TFLiteContext;

/* Utils */

static wasi_nn_error
initialize_g(TFLiteContext *tfl_ctx, graph *g)
{
    os_mutex_lock(&tfl_ctx->g_lock);
    if (tfl_ctx->current_models == MAX_GRAPHS_PER_INST) {
        os_mutex_unlock(&tfl_ctx->g_lock);
        NN_ERR_PRINTF("Excedded max graphs per WASM instance");
        return runtime_error;
    }
    *g = tfl_ctx->current_models++;
    os_mutex_unlock(&tfl_ctx->g_lock);
    return success;
}
static wasi_nn_error
initialize_graph_ctx(TFLiteContext *tfl_ctx, graph g,
                     graph_execution_context *ctx)
{
    os_mutex_lock(&tfl_ctx->g_lock);
    if (tfl_ctx->current_interpreters == MAX_GRAPH_EXEC_CONTEXTS_PER_INST) {
        os_mutex_unlock(&tfl_ctx->g_lock);
        NN_ERR_PRINTF("Excedded max graph execution context per WASM instance");
        return runtime_error;
    }
    *ctx = tfl_ctx->current_interpreters++;
    os_mutex_unlock(&tfl_ctx->g_lock);
    return success;
}

static wasi_nn_error
is_valid_graph(TFLiteContext *tfl_ctx, graph g)
{
    if (g >= MAX_GRAPHS_PER_INST) {
        NN_ERR_PRINTF("Invalid graph: %d >= %d.", g, MAX_GRAPHS_PER_INST);
        return runtime_error;
    }
    if (tfl_ctx->models[g].model_pointer == NULL) {
        NN_ERR_PRINTF("Context (model) non-initialized.");
        return runtime_error;
    }
    if (tfl_ctx->models[g].model == NULL) {
        NN_ERR_PRINTF("Context (tflite model) non-initialized.");
        return runtime_error;
    }
    return success;
}

static wasi_nn_error
is_valid_graph_execution_context(TFLiteContext *tfl_ctx,
                                 graph_execution_context ctx)
{
    if (ctx >= MAX_GRAPH_EXEC_CONTEXTS_PER_INST) {
        NN_ERR_PRINTF("Invalid graph execution context: %d >= %d", ctx,
                      MAX_GRAPH_EXEC_CONTEXTS_PER_INST);
        return runtime_error;
    }
    if (tfl_ctx->interpreters[ctx].interpreter == NULL) {
        NN_ERR_PRINTF("Context (interpreter) non-initialized.");
        return runtime_error;
    }
    return success;
}

/* WASI-NN (tensorflow) implementation */
__attribute__((visibility("default"))) wasi_nn_error
load(void *tflite_ctx, graph_builder_array *builder, graph_encoding encoding,
     execution_target target, graph *g)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    if (builder->size != 1) {
        NN_ERR_PRINTF("Unexpected builder format.");
        return invalid_argument;
    }

    if (encoding != tensorflowlite) {
        NN_ERR_PRINTF("Encoding is not tensorflowlite.");
        return invalid_argument;
    }

    if (target != cpu && target != gpu && target != tpu) {
        NN_ERR_PRINTF("Only CPU, GPU and TPU target is supported.");
        return invalid_argument;
    }

    wasi_nn_error res;
    if (success != (res = initialize_g(tfl_ctx, g)))
        return res;

    uint32_t size = builder->buf[0].size;

    // Save model
    tfl_ctx->models[*g].model_pointer = (char *)wasm_runtime_malloc(size);
    if (tfl_ctx->models[*g].model_pointer == NULL) {
        NN_ERR_PRINTF("Error when allocating memory for model.");
        return too_large;
    }

    bh_memcpy_s(tfl_ctx->models[*g].model_pointer, size, builder->buf[0].buf,
                size);

#if WASM_ENABLE_TFLITE_MICRO != 0
    /* Map the model into a usable data structure. This doesn't involve any*/
    /* copying or parsing, it's a very lightweight operation.*/
    tfl_ctx->models[*g].model =
        const_cast<tflite::Model*>(tflite::GetModel(tfl_ctx->models[*g].model_pointer));
#else
    // Save model flatbuffer
    tfl_ctx->models[*g].model =
        std::move(tflite::FlatBufferModel::BuildFromBuffer(
            tfl_ctx->models[*g].model_pointer, size, NULL));
#endif

    if (tfl_ctx->models[*g].model == NULL) {
        NN_ERR_PRINTF("Loading model error.");
        wasm_runtime_free(tfl_ctx->models[*g].model_pointer);
        tfl_ctx->models[*g].model_pointer = NULL;
        return too_large;
    }

#if WASM_ENABLE_TFLITE_MICRO != 0
    if (tfl_ctx->models[*g].model->version() != TFLITE_SCHEMA_VERSION) {
        NN_ERR_PRINTF("Model provided is schema version %d not equal to supported "
                "version %d.", tfl_ctx->models[*g].model->version(), TFLITE_SCHEMA_VERSION);
        free(tfl_ctx->models[*g].model_pointer);
        return unsupported_operation;
    }
#endif

    // Save target
    tfl_ctx->models[*g].target = target;
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name(void *tflite_ctx, const char *filename, uint32_t filename_len,
             graph *g)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    wasi_nn_error res = initialize_g(tfl_ctx, g);
    if (success != res)
        return res;

    // Load model
#if WASM_ENABLE_TFLITE_MICRO != 0
    tfl_ctx->models[*g].model =
        const_cast<tflite::Model*>(tflite::GetModel(tfl_ctx->models[*g].model_pointer));
#else
    tfl_ctx->models[*g].model =
        std::move(tflite::FlatBufferModel::BuildFromFile(filename, NULL));
#endif

    if (tfl_ctx->models[*g].model == NULL) {
        NN_ERR_PRINTF("Loading model error.");
        return too_large;
    }

#if WASM_ENABLE_TFLITE_MICRO != 0
    if (tfl_ctx->models[*g].model->version() != TFLITE_SCHEMA_VERSION) {
        NN_ERR_PRINTF("Model provided is schema version %d not equal to supported "
                "version %d.", tfl_ctx->models[*g].model->version(), TFLITE_SCHEMA_VERSION);
        free(tfl_ctx->models[*g].model_pointer);
        return unsupported_operation;
    }
#endif
    // Use CPU as default
    tfl_ctx->models[*g].target = cpu;
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
init_execution_context(void *tflite_ctx, graph g, graph_execution_context *ctx)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    wasi_nn_error res;
    if (success != (res = is_valid_graph(tfl_ctx, g)))
        return res;

    if (success != (res = initialize_graph_ctx(tfl_ctx, g, ctx)))
        return res;

#if WASM_ENABLE_TFLITE_MICRO != 0
    /* tflite::MicroMutableOpResolver<6> wasi_nn_micro_op_resolver*/
    /* Todo : maybe we can add more default op here? */
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_AVERAGE_POOL_2D))
        wasi_nn_micro_op_resolver.AddAveragePool2D();
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_CONV_2D))
        wasi_nn_micro_op_resolver.AddConv2D();
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D))
        wasi_nn_micro_op_resolver.AddDepthwiseConv2D();
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_RESHAPE))
        wasi_nn_micro_op_resolver.AddReshape();
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_SOFTMAX))
        wasi_nn_micro_op_resolver.AddSoftmax();
    if (!wasi_nn_micro_op_resolver.FindOp(tflite::BuiltinOperator_MAX_POOL_2D))
        wasi_nn_micro_op_resolver.AddMaxPool2D();

    if (wasi_nn_tensor_arena[g] != NULL) {
        wasm_runtime_free(wasi_nn_tensor_arena[g]);
        wasi_nn_tensor_arena[g] = NULL;
    }
    
    wasi_nn_tensor_arena[g] = (uint8_t *)wasm_runtime_malloc(kTensorArenaSize);
    if (wasi_nn_tensor_arena[g] == NULL) {
        NN_ERR_PRINTF("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return unsupported_operation;
    }

    /*tfl_ctx->interpreters[*ctx].interpreter = new tflite::MicroInterpreter(tfl_ctx->models[g].model,
                                                               wasi_nn_micro_op_resolver,
                                                               wasi_nn_tensor_arena[g],
                                                               kTensorArenaSize);*/
    /* use c-malloc/c-free instead of new/delete, because NUTTX cannot take template/operator as an symbol */
    tfl_ctx->interpreters[*ctx].interpreter = (tflite::MicroInterpreter*)wasm_runtime_malloc(sizeof(tflite::MicroInterpreter));
    tflite::MicroInterpreter static_interpreter(tfl_ctx->models[g].model, wasi_nn_micro_op_resolver,
		                                    wasi_nn_tensor_arena[g], kTensorArenaSize);
    memmove(tfl_ctx->interpreters[*ctx].interpreter, &static_interpreter, sizeof(tflite::MicroInterpreter));

#else
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder tflite_builder(*tfl_ctx->models[g].model,
                                              resolver);
    tflite_builder(&tfl_ctx->interpreters[*ctx].interpreter);
#endif

    if (tfl_ctx->interpreters[*ctx].interpreter == NULL) {
        NN_ERR_PRINTF("Error when generating the interpreter.");
        return too_large;
    }

    bool use_default = false;
    switch (tfl_ctx->models[g].target) {
        case gpu:
        {
#if WASM_ENABLE_WASI_NN_GPU != 0
            NN_WARN_PRINTF("GPU enabled.");
            // https://www.tensorflow.org/lite/performance/gpu
            TfLiteGpuDelegateOptionsV2 options =
                TfLiteGpuDelegateOptionsV2Default();
            options.inference_preference =
                TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
            options.inference_priority1 =
                TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            tfl_ctx->delegate = TfLiteGpuDelegateV2Create(&options);
            if (tfl_ctx->delegate == NULL) {
                NN_ERR_PRINTF("Error when generating GPU delegate.");
                use_default = true;
                return too_large;
            }
            if (tfl_ctx->interpreters[*ctx]
                    .interpreter->ModifyGraphWithDelegate(tfl_ctx->delegate)
                != kTfLiteOk) {
                NN_ERR_PRINTF("Error when enabling GPU delegate.");
                use_default = true;
            }
#else
            NN_WARN_PRINTF("GPU not enabled.");
            use_default = true;
#endif
            break;
        }
        case tpu:
        {
#if WASM_ENABLE_WASI_NN_EXTERNAL_DELEGATE != 0
            NN_WARN_PRINTF("external delegation enabled.");
            TfLiteExternalDelegateOptions options =
                TfLiteExternalDelegateOptionsDefault(
                    WASM_WASI_NN_EXTERNAL_DELEGATE_PATH);
            tfl_ctx->delegate = TfLiteExternalDelegateCreate(&options);
            if (tfl_ctx->delegate == NULL) {
                NN_ERR_PRINTF("Error when generating External delegate.");
                use_default = true;
                return too_large;
            }
            if (tfl_ctx->interpreters[*ctx]
                    .interpreter->ModifyGraphWithDelegate(tfl_ctx->delegate)
                != kTfLiteOk) {
                NN_ERR_PRINTF("Error when enabling External delegate.");
                use_default = true;
            }
#else
            NN_WARN_PRINTF("External delegate not enabled.");
            use_default = true;
#endif
            break;
        }
        default:
            use_default = true;
    }
    if (use_default)
        NN_WARN_PRINTF("Default encoding is CPU.");

    tfl_ctx->interpreters[*ctx].interpreter->AllocateTensors();
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
set_input(void *tflite_ctx, graph_execution_context ctx, uint32_t index,
          tensor *input_tensor)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    wasi_nn_error res;
    if (success != (res = is_valid_graph_execution_context(tfl_ctx, ctx)))
        return res;

    uint32_t num_tensors =
        tfl_ctx->interpreters[ctx].interpreter->inputs().size();
    NN_DBG_PRINTF("Number of tensors (%d)", num_tensors);
    if (index + 1 > num_tensors) {
        return runtime_error;
    }

    auto tensor = tfl_ctx->interpreters[ctx].interpreter->input_tensor(index);
    if (tensor == NULL) {
        NN_ERR_PRINTF("Missing memory");
        return too_large;
    }

    uint32_t model_tensor_size = 1;
    for (int i = 0; i < tensor->dims->size; ++i)
        model_tensor_size *= (uint32_t)tensor->dims->data[i];

    uint32_t input_tensor_size = 1;
    for (uint32_t i = 0; i < input_tensor->dimensions->size; i++)
        input_tensor_size *= (uint32_t)input_tensor->dimensions->buf[i];

    if (model_tensor_size != input_tensor_size) {
        NN_ERR_PRINTF("Input tensor shape from the model is different than the "
                      "one provided");
        return invalid_argument;
    }

    if (tensor->quantization.type == kTfLiteNoQuantization) {
        NN_DBG_PRINTF("No quantization information. Using float as default");
        float *it =
            tfl_ctx->interpreters[ctx].interpreter->typed_input_tensor<float>(
                index);

        int size = model_tensor_size * sizeof(float);
        bh_memcpy_s(it, size, input_tensor->data, size);
    }
    else { // TODO: Assumming uint8 quantized networks.
        TfLiteAffineQuantization *quant_info =
            (TfLiteAffineQuantization *)tensor->quantization.params;
        if (quant_info->scale->size != 1 || quant_info->zero_point->size != 1) {
            NN_ERR_PRINTF("Quantization per channel is not supported");
            return runtime_error;
        }
        uint8_t *it =
            tfl_ctx->interpreters[ctx].interpreter->typed_input_tensor<uint8_t>(
                index);

        float scale = quant_info->scale->data[0];
        float zero_point = (float)quant_info->zero_point->data[0];
        NN_DBG_PRINTF("input tensor: (scale, offset) = (%f, %f)", scale,
                      zero_point);

        float *input_tensor_f = (float *)input_tensor->data;
        for (uint32_t i = 0; i < model_tensor_size; ++i) {
            it[i] = (uint8_t)(input_tensor_f[i] / scale + zero_point);
        }
    }

    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
compute(void *tflite_ctx, graph_execution_context ctx)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    wasi_nn_error res;
    if (success != (res = is_valid_graph_execution_context(tfl_ctx, ctx)))
        return res;

    tfl_ctx->interpreters[ctx].interpreter->Invoke();
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
get_output(void *tflite_ctx, graph_execution_context ctx, uint32_t index,
           tensor_data output_tensor, uint32_t *output_tensor_size)
{
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    wasi_nn_error res;
    if (success != (res = is_valid_graph_execution_context(tfl_ctx, ctx)))
        return res;

    uint32_t num_output_tensors =
        tfl_ctx->interpreters[ctx].interpreter->outputs().size();
    NN_DBG_PRINTF("Number of tensors (%d)", num_output_tensors);

    if (index + 1 > num_output_tensors) {
        NN_ERR_PRINTF("Index %d is invalid.", index);
        return runtime_error;
    }

    auto tensor = tfl_ctx->interpreters[ctx].interpreter->output_tensor(index);
    if (tensor == NULL) {
        NN_ERR_PRINTF("Missing memory");
        return too_large;
    }

    uint32_t model_tensor_size = 1;
    for (int i = 0; i < (int)tensor->dims->size; ++i)
        model_tensor_size *= (uint32_t)tensor->dims->data[i];

    if (*output_tensor_size < model_tensor_size) {
        NN_ERR_PRINTF("Insufficient memory to copy tensor %d", index);
        return too_large;
    }

    if (tensor->quantization.type == kTfLiteNoQuantization) {
        NN_DBG_PRINTF("No quantization information. Using float as default");
        float *ot =
            tfl_ctx->interpreters[ctx].interpreter->typed_output_tensor<float>(
                index);

        int size = model_tensor_size * sizeof(float);
        bh_memcpy_s(output_tensor, size, ot, size);
    }
    else { // TODO: Assumming uint8 quantized networks.
        TfLiteAffineQuantization *quant_info =
            (TfLiteAffineQuantization *)tensor->quantization.params;
        if (quant_info->scale->size != 1 || quant_info->zero_point->size != 1) {
            NN_ERR_PRINTF("Quantization per channel is not supported");
            return runtime_error;
        }
        uint8_t *ot = tfl_ctx->interpreters[ctx]
                          .interpreter->typed_output_tensor<uint8_t>(index);

        float scale = quant_info->scale->data[0];
        float zero_point = (float)quant_info->zero_point->data[0];
        NN_DBG_PRINTF("output tensor: (scale, offset) = (%f, %f)", scale,
                      zero_point);

        float *output_tensor_f = (float *)output_tensor;
        for (uint32_t i = 0; i < model_tensor_size; ++i) {
            output_tensor_f[i] = (ot[i] - zero_point) * scale;
	    NN_DBG_PRINTF("output_f[%d]: %f", i, output_tensor_f[i]);
        }
    }

    *output_tensor_size = model_tensor_size;
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
init_backend(void **tflite_ctx)
{
#if WASM_ENABLE_TFLITE_MICRO != 0
    TFLiteContext *tfl_ctx = (TFLiteContext *)wasm_runtime_malloc(sizeof(TFLiteContext));
    TFLiteContext static_ctx;
    memmove(tfl_ctx, &static_ctx, sizeof(TFLiteContext));
#else
    TFLiteContext *tfl_ctx = new TFLiteContext();
#endif
    if (tfl_ctx == NULL) {
        NN_ERR_PRINTF("Error when allocating memory for tensorflowlite.");
        return runtime_error;
    }

    NN_DBG_PRINTF("Initializing models.");
    tfl_ctx->current_models = 0;
    for (int i = 0; i < MAX_GRAPHS_PER_INST; ++i) {
        tfl_ctx->models[i].model_pointer = NULL;
    }
    NN_DBG_PRINTF("Initializing interpreters.");
    tfl_ctx->current_interpreters = 0;

    if (os_mutex_init(&tfl_ctx->g_lock) != 0) {
        NN_ERR_PRINTF("Error while initializing the lock");
    }

#if WASM_ENABLE_TFLITE_MICRO == 0
    tfl_ctx->delegate = NULL;
#endif

    *tflite_ctx = (void *)tfl_ctx;
    return success;
}

__attribute__((visibility("default"))) wasi_nn_error
deinit_backend(void *tflite_ctx)
{
    /*
        TensorFlow Lite memory is internally managed by tensorflow

        Related issues:
        * https://github.com/tensorflow/tensorflow/issues/15880
    */
    TFLiteContext *tfl_ctx = (TFLiteContext *)tflite_ctx;

    NN_DBG_PRINTF("Freeing memory.");
    for (int i = 0; i < MAX_GRAPHS_PER_INST; ++i) {
#if WASM_ENABLE_TFLITE_MICRO != 0
        tfl_ctx->models[i].model = NULL;
#else
        tfl_ctx->models[i].model.reset();
#endif
        if (tfl_ctx->models[i].model_pointer) {
#if WASM_ENABLE_TFLITE_MICRO == 0
            if (tfl_ctx->delegate) {
                switch (tfl_ctx->models[i].target) {
                    case gpu:
                    {
#if WASM_ENABLE_WASI_NN_GPU != 0
                        TfLiteGpuDelegateV2Delete(tfl_ctx->delegate);
#else
                        NN_ERR_PRINTF("GPU delegate delete but not enabled.");
#endif
                        break;
                    }
                    case tpu:
                    {
#if WASM_ENABLE_WASI_NN_EXTERNAL_DELEGATE != 0
                        TfLiteExternalDelegateDelete(tfl_ctx->delegate);
#else
                        NN_ERR_PRINTF(
                            "External delegate delete but not enabled.");
#endif
                        break;
                    }
                    default:
                        break;
                }
            }
#endif
            wasm_runtime_free(tfl_ctx->models[i].model_pointer);
        }
        tfl_ctx->models[i].model_pointer = NULL;
    }
    for (int i = 0; i < MAX_GRAPH_EXEC_CONTEXTS_PER_INST; ++i) {
#if WASM_ENABLE_TFLITE_MICRO != 0
        if (tfl_ctx->interpreters[i].interpreter)
            tfl_ctx->interpreters[i].interpreter->Reset();
        wasm_runtime_free(tfl_ctx->interpreters[i].interpreter);
        tfl_ctx->interpreters[i].interpreter = NULL;
#else
        tfl_ctx->interpreters[i].interpreter.reset();
#endif
    }

#if WASM_ENABLE_TFLITE_MICRO != 0
    for (int i = 0; i < MAX_GRAPH_EXEC_CONTEXTS_PER_INST; ++i) {
        if (wasi_nn_tensor_arena[i])
            wasm_runtime_free(wasi_nn_tensor_arena[i]);
        wasi_nn_tensor_arena[i] = NULL;
    }
#endif

    os_mutex_destroy(&tfl_ctx->g_lock);
#if WASM_ENABLE_TFLITE_MICRO != 0
    wasm_runtime_free(tfl_ctx);
#else
    delete tfl_ctx;
#endif
    NN_DBG_PRINTF("Memory free'd.");
    return success;
}
