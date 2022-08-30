/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#include "tensorflow/core/kernels/conv_ops_gpu.h"

#include "absl/types/span.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA
namespace {

template <typename LaunchFunc, typename Sig>
StatusOr<std::vector<tensorflow::AutotuneResult>> AutotuneConvImpl(
    OpKernelContext* ctx,
    std::vector<std::unique_ptr<const se::dnn::OpRunner<Sig>>>& runners,
    bool actually_do_autotune, const LaunchFunc& launch_func,
    size_t scratch_size_limit, const se::RedzoneAllocator& rz_allocator) {
  auto* stream = ctx->op_device_context()->stream();

  se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                              stream);
  auto start = std::chrono::steady_clock::now();
  std::vector<tensorflow::AutotuneResult> results;
  VLOG(1) << "The number of runners to tune is: " << runners.size();
  // TODO(reedwm): Warn if determinism is enabled after autotune is run
  for (auto& runner : runners) {
    // TODO(zhengxq): profile each algorithm multiple times to better
    // accuracy.
    auto start_per_runner = std::chrono::steady_clock::now();
    se::RedzoneAllocator rz_scratch_allocator(
        stream, &tf_allocator_adapter, se::GpuAsmOpts(),
        /*memory_limit=*/scratch_size_limit);
    DnnScratchAllocator scratch_allocator(scratch_size_limit, ctx);
    se::ScratchAllocator* allocator_used =
        !RedzoneCheckDisabled()
            ? static_cast<se::ScratchAllocator*>(&rz_scratch_allocator)
            : static_cast<se::ScratchAllocator*>(&scratch_allocator);

    SE_ASSIGN_OR_RETURN(auto desc, runner->ToAlgorithmDesc());
    se::dnn::ProfileResult profile_result;
    VLOG(1) << "actually_do_autotune: "
            << (actually_do_autotune ? "true" : "false");
    Status cudnn_launch_status =
        actually_do_autotune
            ? launch_func(allocator_used, runner, &profile_result)
            : OkStatus();
    if (!actually_do_autotune) {
      // Make the result valid according to `is_valid`.
      profile_result.set_algorithm(desc);
      profile_result.set_elapsed_time_in_ms(0);
    }

    // We need to make sure the profiling results are one-to-one with the
    // "runners". So, we insert dummy results when the execution fails.
    results.emplace_back();
    auto& result = results.back();
    *result.mutable_algorithm() = desc.ToProto();
    if (cudnn_launch_status.ok() && profile_result.is_valid()) {
      result.set_scratch_bytes(
          !RedzoneCheckDisabled()
              ? rz_scratch_allocator.TotalAllocatedBytesExcludingRedzones()
              : scratch_allocator.TotalByteSize());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));

      CheckRedzones(rz_scratch_allocator, &result);
      CheckRedzones(rz_allocator, &result);
    } else {
      result.mutable_failure()->set_kind(AutotuneResult::UNKNOWN);
      result.mutable_failure()->set_msg(
          absl::StrCat("Profiling failure on CUDNN engine ", desc.ToString(),
                       ": ", cudnn_launch_status.ToString()));
    }
    auto end_per_runner = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milli_seconds_per_runner =
        end_per_runner - start_per_runner;
    VLOG(1) << "Iterating per runner takes " << elapsed_milli_seconds_per_runner.count() << "ms.";
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> elapsed_milli_seconds = end - start;
  VLOG(1) << "Iterating over all runners takes " << elapsed_milli_seconds.count() << "ms.";

  return results;
}
}  // namespace
#endif  // GOOGLE_CUDA

bool ComputeInNhwcEnabled(DataType data_type, se::Stream* stream,
                          bool is_conv2d) {
#if GOOGLE_CUDA
  // Tensor Core supports efficient convolution with fp16 for NVIDIA Volta+
  // GPUs and tf32 for Ampere+ GPUs in NHWC data layout. In all other
  // configurations it's more efficient to run computation in NCHW data format.
  bool use_nhwc_tf32 = data_type == DT_FLOAT &&
                       stream->GetCudaComputeCapability().IsAtLeast(
                           se::CudaComputeCapability::AMPERE) &&
                       tensorflow::tensor_float_32_execution_enabled();
  bool use_nhwc_fp16 =
      data_type == DT_HALF && stream->GetCudaComputeCapability().IsAtLeast(
                                  se::CudaComputeCapability::VOLTA);
  if (is_conv2d) {
    return use_nhwc_fp16 || use_nhwc_tf32;
  }
  return CUDNN_VERSION >= 8000 && (use_nhwc_fp16 || use_nhwc_tf32);
#else
  return false;
#endif  // GOOGLE_CUDA
}

// Finds the best convolution algorithm for the given ConvLaunch (cuda
// convolution on the stream) and parameters, by running all possible
// algorithms and measuring execution time.
template <typename T>
StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<T> input_ptr,
    se::DeviceMemory<T> filter_ptr, se::DeviceMemory<T> output_ptr,
    se::DeviceMemory<T> bias_ptr, se::DeviceMemory<T> side_input_ptr,
    int64_t scratch_size_limit) {
#if GOOGLE_CUDA
  AutotuneEntry<se::dnn::FusedConvOp> autotune_entry;
  auto* stream = ctx->op_device_context()->stream();

  if (!autotune_map->Find(params, &autotune_entry)) {
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());
    se::DeviceMemory<T> output_ptr_rz(
        WrapRedzoneBestEffort(&rz_allocator, output_ptr));

    std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
    auto element_type = se::dnn::ToDataType<T>::value;
    SE_RETURN_IF_ERROR(stream->parent()->GetFusedConvolveRunners(
        CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
        element_type, element_type, conv_scale, side_input_scale, stream,
        input_desc, filter_desc, bias_desc, output_desc, conv_desc,
        /*use_fallback=*/false, activation_mode, &runners));

    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::FusedConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       side_input_ptr, bias_ptr, output_ptr_rz);
    };

    auto start = std::chrono::steady_clock::now();
    SE_ASSIGN_OR_RETURN(
        auto results,
        AutotuneConvImpl(ctx, runners, cudnn_use_autotune, launch_func,
                         scratch_size_limit, rz_allocator));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milli_seconds =
        end - start;
    VLOG(1) << "Autotune fused conv takes " << elapsed_milli_seconds.count()
            << "ms";
    // Only log on an AutotuneConv cache miss.
    LogFusedConvForwardAutotuneResults(
        se::dnn::ToDataType<T>::value, input_ptr, filter_ptr, output_ptr,
        bias_ptr, side_input_ptr, input_desc, filter_desc, output_desc,
        conv_desc, conv_scale, side_input_scale, activation_mode,
        stream->parent(), results);

    // Two-level autotuning: Cudnn frontend supports two engine lists:
    // heuristics and fallback. Heuristics engines are normally faster.
    // To reduce autotuning time, we evaluate the fallback engines only when
    // none of the heuristics engines work.
    bool found_working_engine = false;
    for (auto& result : results) {
      if (!result.has_failure()) {
        found_working_engine = true;
        break;
      }
    }

    if (!CudnnUseFrontend() || found_working_engine) {
      TF_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
                              results, std::move(runners)));
    } else {
      LOG(WARNING)
          << "None of the algorithms provided by cuDNN frontend heuristics "
             "worked; trying fallback algorithms.  Conv: "
          << params.ToString();
      std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
          fallback_runners;
      SE_RETURN_IF_ERROR(stream->parent()->GetFusedConvolveRunners(
          CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
          element_type, element_type, conv_scale, side_input_scale, stream,
          input_desc, filter_desc, bias_desc, output_desc, conv_desc,
          /*use_fallback=*/true, activation_mode, &fallback_runners));

      auto start_fallback = std::chrono::steady_clock::now();
      SE_ASSIGN_OR_RETURN(
          auto fallback_results,
          AutotuneConvImpl(ctx, fallback_runners, cudnn_use_autotune,
                           launch_func, scratch_size_limit, rz_allocator));
      auto end_fallback = std::chrono::steady_clock::now();
      auto elapsed_micro_seconds_fallback = end - start;
      VLOG(1) << "Autotune fused conv in fallback takes "
              << elapsed_micro_seconds_fallback.count() << "ms";

      LogFusedConvForwardAutotuneResults(
          se::dnn::ToDataType<T>::value, input_ptr, filter_ptr, output_ptr,
          bias_ptr, side_input_ptr, input_desc, filter_desc, output_desc,
          conv_desc, conv_scale, side_input_scale, activation_mode,
          stream->parent(), fallback_results);

      TF_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
                              fallback_results, std::move(fallback_runners)));
    }

    VLOG(1) << "Insert autotune map in autotune fused conv";
    autotune_map->Insert(params, autotune_entry);
  }
  return autotune_entry;
#else
  return errors::Unimplemented(
      "Fused conv not implemented on non-CUDA platforms.");
#endif
}

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>>
AutotuneFusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<double> input_ptr,
    se::DeviceMemory<double> filter_ptr, se::DeviceMemory<double> output_ptr,
    se::DeviceMemory<double> bias_ptr, se::DeviceMemory<double> side_input_ptr,
    int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<float> input_ptr,
    se::DeviceMemory<float> filter_ptr, se::DeviceMemory<float> output_ptr,
    se::DeviceMemory<float> bias_ptr, se::DeviceMemory<float> side_input_ptr,
    int64_t scratch_size_limit);

template <typename T>
StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<T> input_ptr, const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<T> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc, se::DeviceMemory<T> output_ptr,
    int64_t scratch_size_limit) {
  AutotuneEntry<se::dnn::ConvOp> autotune_entry;

  auto* stream = ctx->op_device_context()->stream();

  /*
  // TODO(amoschenyq): Remove this after determining that missing batch entry
  // can be inferred from other batch entries available in autotune map
  // And a specific batch size is set here for debugging
  const char* tf_debug_cudnn_infer = std::getenv("TF_DEBUG_CUDNN_INFER");
  if (!autotune_map->Find(conv_parameters, &autotune_entry) &&
      conv_parameters.proto().batch() == 127 && tf_debug_cudnn_infer) {
    VLOG(1) << "Begin infer batch size is 127";
    const absl::Span<const int64_t> in_infer;
    int device_id = stream->parent()->device_ordinal();
    // TODO(amoschenyq): Simplify this by adding a member function in class
    ConParameters ConvParameters conv_parameters_for_infer = {
        128,
        conv_parameters.proto().in_depths(),
        conv_parameters.proto().in(),
        conv_parameters.proto().data_format(),
        conv_parameters.proto().out_depths(),
        conv_parameters.proto().filter(),
        conv_parameters.proto().dilation(),
        conv_parameters.proto().stride(),
        conv_parameters.proto().padding(),
        conv_parameters.proto().dtype(),
        device_id,
        conv_parameters.proto().group_count()};
    autotune_map->Find(conv_parameters_for_infer, &autotune_entry);
    VLOG(1) << "Infer autotune_entry result is:" <<
    autotune_entry.ToString(); return autotune_entry;
  }
  */

  if (!autotune_map->Find(conv_parameters, &autotune_entry)) {
    VLOG(1) << "Can not find autotune entry for conv parameters of "
            << conv_parameters.ToString();
    profiler::ScopedAnnotation annotation("cudnn_autotuning");

#if GOOGLE_CUDA
    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());

    // TODO(awpr): second-guess whether it's okay that this profiles
    // convolutions on uninitialized memory.
    switch (kind) {
      case se::dnn::ConvolutionKind::FORWARD:
      case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
        VLOG(1) << "ConvolutionKind is FORWARD or FORWARD_BIAS_ACTIVATION";
        output_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, output_ptr));
        break;
      case se::dnn::ConvolutionKind::BACKWARD_DATA:
        VLOG(1) << "ConvolutionKind is BACKWARD_DATA";
        input_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, input_ptr));
        break;
      case se::dnn::ConvolutionKind::BACKWARD_FILTER:
        VLOG(1) << "ConvolutionKind is BACKWARD_FILTER";
        filter_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, filter_ptr));
        break;
      default:
        return errors::InvalidArgument(
            absl::StrFormat("Unknown ConvolutionKind %d", kind));
    }

    const auto element_type = se::dnn::ToDataType<T>::value;
    std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
    auto start = std::chrono::steady_clock::now();
    TF_RETURN_IF_ERROR(stream->parent()->GetConvolveRunners(
        CudnnUseFrontend(), kind, element_type, element_type, stream,
        input_desc, input_ptr, filter_desc, filter_ptr, output_desc, output_ptr,
        conv_desc, /*use_fallback=*/false, &rz_allocator, &runners));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_milli_seconds =
        end - start;
    VLOG(1) << "Getting unfused conv runners takes "
            << elapsed_milli_seconds.count() << "ms. "
            << "The number of runners is " << runners.size() << ".";
    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::ConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      VLOG(1) << "Lauching lambda function for runner " << runner->ToString();
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       output_ptr);
    };

    start = std::chrono::steady_clock::now();
    SE_ASSIGN_OR_RETURN(
        auto results,
        AutotuneConvImpl(ctx, runners, cudnn_use_autotune, launch_func,
                         scratch_size_limit, rz_allocator));
    end = std::chrono::steady_clock::now();
    elapsed_milli_seconds = end - start;
    VLOG(1) << "Autotune unfused conv in heuristics takes "
            << elapsed_milli_seconds.count() << "ms";

    LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                           filter_ptr, output_ptr, input_desc, filter_desc,
                           output_desc, conv_desc, stream->parent(), results);

    // Two-level autotuning: Cudnn frontend supports two engine lists:
    // heuristics and fallback. Heuristics engines are normally faster.
    // To reduce autotuning time, we evaluate the fallback engines only when
    // none of the heuristics engines work.
    bool found_working_engine = false;
    for (auto& result : results) {
      if (!result.has_failure()) {
        found_working_engine = true;
        break;
      }
    }

    VLOG(1) << "Cudnn use frontend: " << (CudnnUseFrontend() ? "true" : "false")
            << " found working engine: "
            << (found_working_engine ? "true" : "false");
    if (!CudnnUseFrontend() || found_working_engine) {
      VLOG(1) << "Get autotune_entry from heuristics results";
      SE_ASSIGN_OR_RETURN(
          autotune_entry,
          BestCudnnConvAlgorithm<se::dnn::ConvOp>(results, std::move(runners)));
    } else {
      VLOG(1) << "None of the algorithms provided by cuDNN frontend heuristics "
                 "worked; trying fallback algorithms.  Conv: "
              << conv_parameters.ToString();
      std::vector<std::unique_ptr<const se::dnn::ConvRunner>> fallback_runners;
      TF_RETURN_IF_ERROR(stream->parent()->GetConvolveRunners(
          CudnnUseFrontend(), kind, element_type, element_type, stream,
          input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
          output_ptr, conv_desc, /*use_fallback=*/true, &rz_allocator,
          &fallback_runners));

      SE_ASSIGN_OR_RETURN(
          auto fallback_results,
          AutotuneConvImpl(ctx, fallback_runners, cudnn_use_autotune,
                           launch_func, scratch_size_limit, rz_allocator));

      LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                             filter_ptr, output_ptr, input_desc, filter_desc,
                             output_desc, conv_desc, stream->parent(),
                             fallback_results);

      SE_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::ConvOp>(
                              fallback_results, std::move(fallback_runners)));
    }

#elif TENSORFLOW_USE_ROCM
    DnnScratchAllocator scratch_allocator(scratch_size_limit, ctx);

    std::vector<se::dnn::ProfileResult> algorithms;
    if (!stream->parent()->GetMIOpenConvolveAlgorithms(
            kind, se::dnn::ToDataType<T>::value, stream, input_desc, input_ptr,
            filter_desc, filter_ptr, output_desc, output_ptr, conv_desc,
            &scratch_allocator, &algorithms)) {
      return errors::Unknown(
          "Failed to get convolution algorithm. This is probably "
          "because MIOpen failed to initialize, so try looking to "
          "see if a warning log message was printed above.");
    }

    std::vector<tensorflow::AutotuneResult> results;
    if (algorithms.size() == 1) {
      auto profile_result = algorithms[0];
      results.emplace_back();
      auto& result = results.back();
      *result.mutable_algorithm() = profile_result.algorithm().ToProto();

      result.set_scratch_bytes(profile_result.scratch_size());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    } else {
      for (auto miopen_algorithm : algorithms) {
        auto profile_algorithm = miopen_algorithm.algorithm();
        se::dnn::ProfileResult profile_result;
        auto miopen_launch_status = stream->ConvolveWithAlgorithm(
            kind, input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
            output_ptr, conv_desc, &scratch_allocator,
            se::dnn::AlgorithmConfig(profile_algorithm,
                                     miopen_algorithm.scratch_size()),
            &profile_result);
        if (miopen_launch_status.ok() && profile_result.is_valid()) {
          results.emplace_back();
          auto& result = results.back();
          *result.mutable_algorithm() = profile_algorithm.ToProto();

          result.set_scratch_bytes(scratch_allocator.TotalByteSize());
          *result.mutable_run_time() = proto_utils::ToDurationProto(
              absl::Milliseconds(profile_result.elapsed_time_in_ms()));
        }
      }
    }
    LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                           filter_ptr, output_ptr, input_desc, filter_desc,
                           output_desc, conv_desc, stream->parent(), results);

    SE_ASSIGN_OR_RETURN(auto algo_desc, BestCudnnConvAlgorithm(results));
    autotune_entry = AutotuneEntry<se::dnn::ConvOp>(algo_desc);
#endif

    VLOG(1) << "Insert autotune map in autotune unfused conv";
    autotune_map->Insert(conv_parameters, autotune_entry);
  }

  VLOG(1) << "autotune entry is: " << autotune_entry.ToString();

  return autotune_entry;
}

template StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<double> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<double> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<double> output_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<float> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<float> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<float> output_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::ConvOp>>
AutotuneUnfusedConv<Eigen::half>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<Eigen::half> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<Eigen::half> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<Eigen::half> output_ptr, int64_t scratch_size_limit);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
