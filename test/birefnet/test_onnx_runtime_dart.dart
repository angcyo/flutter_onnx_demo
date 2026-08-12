import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_onnx_demo/src/birefnet_helper.dart';
import 'package:image/image.dart' as img;
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/11
///
/// # 2026-8-11
/// ```
/// package:onnx_runtime_dart/src/onnx_graph.dart 793:9      OnnxGraphExecutor._execNodes
/// package:onnx_runtime_dart/src/onnx_graph.dart 554:5      OnnxGraphExecutor.run
/// package:onnx_runtime_dart/onnx_runtime_dart.dart 108:17  OnnxModel.run
/// test\test_onnx_runtime_dart.dart 34:21                   main
///
/// Failed to load "E:/projects/flutter/flutter_onnx_demo/test/test_onnx_runtime_dart.dart": Bad state: ONNX graph execution failed at node "/squeeze_module/squeeze_module.0/dec_att/aspp1/atrous_conv/GatherND" (op=GatherND, input shapes: [1, 1, 34, 34, 64], [1, 1, 1024, 2]): 'package:onnx_runtime_dart/src/onnx_ops.dart': Failed assertion: line 802 pos 10: 'batchDims == 0': is not true.
/// ```
void main() {
  // Resolves companion external-data files (large models) automatically.
  //final model = loadOnnxModel(r"E:\dnn\BiRefNet_lite-ONNX\model.onnx");
  final model = loadOnnxModel(r"E:\dnn\BiRefNet_lite-ONNX\model_fp16.onnx");
  final bytes = File(r"E:\temp\export_1024x1024.png").readAsBytesSync();
  // 解码 PNG/JPEG/WebP 等
  final image = img.decodeImage(bytes)!;

  // --------------------------------------------------
  // 4. RGB -> Float32 NCHW
  //
  // [1, 3, 1024, 1024]
  //
  // RRRRR...
  // GGGGG...
  // BBBBB...
  // --------------------------------------------------

  // 1024*1024*3=3,145,728
  final inputData = BiRefNetHelper.imageToFloat32NCHW(image);

  final out = model.run(
    {
      model.inputSpecs.first.name: Tensor.float(
        Float32List.fromList(inputData),
        model.inputSpecs.first.shape /*1 3 1024 1024*/,
      ),
    },
    [model.outputNames.first],
  );

  print(out[model.outputNames.first]!.asFloatList());
  print("...");
}
