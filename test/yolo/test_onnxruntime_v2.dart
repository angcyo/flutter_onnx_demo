import 'dart:io';

import 'package:flutter_onnx_demo/src/birefnet/birefnet_helper.dart';
import 'package:flutter_onnx_demo/src/yolo/yolo_helper.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/11
///
/// # 2026-8-11
/// ```
/// Succeed
/// ```
void main() async {
  OrtEnv.instance.init();

  //模型输入大小
  final inputSize = 1024;
  //final modelPath = r"E:\dnn\BiRefNet_lite-ONNX\model_fp16.onnx";
  final modelPath = r"E:\dnn\hispark-modelzoo-yolov8s-obb\yolov8s-obb.onnx";
  //final inputImagePath = r"E:\temp\4d8deaf7-ba4f-4b59-a522-f76f873924f8.png";
  //final inputImagePath = r"E:\temp\export_1024x1024.png";
  final inputImagePath = r"test/.output/output_1786522445634.png";
  final modelBytes = File(modelPath).readAsBytesSync();
  final sessionOptions = OrtSessionOptions();
  // 🚀 NEW: Automatically use GPU acceleration if available!
  // This will try GPU providers first, then fall back to CPU
  sessionOptions.appendDefaultProviders();
  final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

  final bytes = File(inputImagePath).readAsBytesSync();
  // 解码 PNG/JPEG/WebP 等
  // 原始图片, 支持任意大小
  final original = img.decodeImage(bytes);
  if (original == null) {
    throw Exception('无法解析图片');
  }

  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final originalWidth = original.width;
  final originalHeight = original.height;

  print(
    'Original image: '
    '$originalWidth x $originalHeight',
  );

  // --------------------------------------------------
  // 2. 转 RGB
  // --------------------------------------------------

  final rgbImage = original.convert(numChannels: 3);

  // --------------------------------------------------
  // 3. Resize 到 1024 x 1024
  // --------------------------------------------------
  final resized = BiRefNetHelper.resizeImage(rgbImage, inputSize, inputSize);
  await img.encodePngFile(
    'test/.output/output_resized_$timestamp.png',
    resized,
  ); // Save the resized as PNG

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
  final inputData = BiRefNetHelper.imageToFloat32NCHW(resized);
  print(
    'Input tensor: '
    '[1, 3, $inputSize, $inputSize]',
  );

  final shape = [1, 3, inputSize, inputSize];
  final inputOrt = OrtValueTensor.createTensorWithDataList(inputData, shape);
  //input_image / images
  final inputs = {session.inputNames.first: inputOrt};
  final runOptions = OrtRunOptions();
  final outputs = await session.runAsync(runOptions, inputs);
  inputOrt.release();
  runOptions.release();

  // --------------------------------------------------
  // 9. 获取输出
  //
  // [1, 1, 1024, 1024]
  // --------------------------------------------------

  final output = outputs?.firstOrNull;

  if (output == null) {
    throw Exception('BiRefNet 输出为空');
  }
  //print(out[model.outputNames.first]!.asFloatList());
  //[output_image] / [output0]
  print(session.outputNames);

  //输出的是前景/背景的置信度 logits
  // logit
  //  ↓
  // sigmoid
  //  ↓
  // 0.0 ~ 1.0
  //  ↓
  // ×255
  //  ↓
  // Alpha
  //[1, 1, 1024, 1024]
  //[[[1024],..[1024]]]
  final outputValue = output.value;

  print('Output type: ${outputValue.runtimeType}');

  final scale =
      inputSize /
      (originalWidth > originalHeight ? originalWidth : originalHeight);

  final padX = (inputSize - resized.width) / 2;
  final padY = (inputSize - resized.height) / 2;
  final result = YoloHelper.decodeObb(
    outputValue,
    padX: padX,
    padY: padY,
    scale: scale,
  );
  print(
    result
        .map((e) => "${e.cx},${e.cy},${e.width},${e.height},${e.angle}")
        .join(","),
  );

  outputs?.forEach((element) {
    //print(element?.value);
    element?.release();
  });
  OrtEnv.instance.release();
}
