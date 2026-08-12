import 'dart:io';

import 'package:betto_onnxrt/betto_onnxrt.dart';
import 'package:flutter_onnx_demo/src/birefnet/birefnet_helper.dart';
import 'package:image/image.dart' as img;

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/11
///
/// # 2026-8-11
/// ```
/// Failed to load "E:/projects/flutter/flutter_onnx_demo/test/test_betto_onnxrt.dart": Bad state: ORT API version 22 is not supported by this library. Update VERSION_ONNX in betto_onnxrt and rebuild the native library.
/// ```
void main() async {
  // 1. Load the ORT runtime (opens the native library staged by the hook).
  final runtime = await OnnxRuntime.load();

  // 2. Create a session from model bytes.
  final modelBytes = File(
    r"E:\dnn\BiRefNet_lite-ONNX\model_fp16.onnx",
  ).readAsBytesSync();
  final session = runtime.createSession(modelBytes);

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

  // 3. Build input tensors and run inference.
  final inputIds = OnnxTensor.fromFloat32([1, 3, 1024, 1024], inputData);
  final outputs = session.run(
    inputs: {'input_ids': inputIds},
    outputNames: ['last_hidden_state'],
  );

  // 4. Read the output.
  final embeddings = outputs.first.asFloat32();

  // 5. Clean up.
  session.dispose();
  runtime.dispose();

  print("...");
}
