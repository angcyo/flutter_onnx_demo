import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_onnx_demo/src/birefnet/birefnet_helper.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_plus/onnxruntime_plus.dart';

import '../basics.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/11
///
/// # 2026-8-11
/// ```
/// Succeed
/// ```
Future<Uint8List?> testOnnxruntimePlusMain() async {
  OrtEnv.instance.init();

  final inputSize = 1024;
  final shape = [1, 3, inputSize, inputSize];
  final modelPath = isMobile
      ? "assets/model_fp16.onnx"
      : r"E:\dnn\BiRefNet_lite-ONNX\model_fp16.onnx";
  final inputImagePath = isMobile
      ? "assets/input.png"
      : r"E:\temp\export_1024x1024.png";
  //final inputImagePath = r"E:\temp\4d8deaf7-ba4f-4b59-a522-f76f873924f8.png";

  final inputImageBytes = isMobile
      ? await readAssetsBytes(inputImagePath)
      : File(inputImagePath).readAsBytesSync();
  final original = img.decodeImage(inputImageBytes);
  if (original == null) {
    throw Exception('无法解析图片');
  }

  final modelBytes = isMobile
      ? await readAssetsBytes(modelPath)
      : File(modelPath).readAsBytesSync();
  final sessionOptions = OrtSessionOptions();
  final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

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
  /*await img.encodePngFile(
    'test/.output/output_resized_$timestamp.png',
    resized,
  ); // Save the resized as PNG*/

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

  final inputOrt = OrtValueTensor.createTensorWithDataList(inputData, shape);
  //input_image
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
  //[output_image]
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

  // --------------------------------------------------
  // 10. 转换 output
  // --------------------------------------------------
  final logits = BiRefNetHelper.extractOutput(outputValue, inputSize);
  print('Output values: ${logits.length}');
  final mask = BiRefNetHelper.sigmoid(logits, inputSize);
  // --------------------------------------------------
  // 12. Resize mask 到原图尺寸
  // --------------------------------------------------
  final maskImage = BiRefNetHelper.maskToImage(mask, inputSize, inputSize);
  final resizedMask = BiRefNetHelper.resizeImage(
    maskImage,
    originalWidth,
    originalHeight,
  );

  // --------------------------------------------------
  // 13. 应用 Alpha
  // --------------------------------------------------
  final result = BiRefNetHelper.applyMask(original, resizedMask);
  /*final outputPath = 'test/.output/output_$timestamp.png';
  await img.encodePngFile(outputPath, result); // Save the result as PNG
  print('Save result to: $outputPath');*/

  outputs?.forEach((element) {
    //print(element?.value);
    element?.release();
  });
  OrtEnv.instance.release();

  //--
  final pngBytes = img.encodePng(result);
  //return decodeImageFromList(pngBytes);
  return pngBytes;
}
