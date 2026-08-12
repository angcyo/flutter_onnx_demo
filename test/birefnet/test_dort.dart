import 'dart:io';

import 'package:dort/dort.dart';
import 'package:flutter_onnx_demo/src/birefnet/birefnet_helper.dart';
import 'package:image/image.dart' as img;

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/12
///
/// # 2026-8-11
/// ```
/// Succeed
/// ```
void main() async {
  final timestamp = DateTime.now().millisecondsSinceEpoch;

  final inputSize = 1024;
  final modelPath = r"E:\dnn\BiRefNet_lite-ONNX\model_fp16.onnx";
  final shape = [1, 3, inputSize, inputSize];

  final session = Session.load(modelPath);

  final inputImagePath = r"E:\temp\export_1024x1024.png";
  final original = img.decodeImage(File(inputImagePath).readAsBytesSync());
  if (original == null) {
    throw Exception('无法解析图片');
  }
  final rgbImage = original.convert(numChannels: 3);
  final resized = BiRefNetHelper.resizeImage(rgbImage, inputSize, inputSize);
  final inputData = BiRefNetHelper.imageToFloat32NCHW(resized);

  final outputs = session.run([Tensor('input_image', inputData, shape)]);

  final outputValue = outputs.first; // Float32List
  //print(outputValue);
  final logits = BiRefNetHelper.extractOutput(outputValue, inputSize);
  final mask = BiRefNetHelper.sigmoid(logits, inputSize);
  final maskImage = BiRefNetHelper.maskToImage(mask, inputSize, inputSize);
  final resizedMask = BiRefNetHelper.resizeImage(
    maskImage,
    original.width,
    original.height,
  );
  final result = BiRefNetHelper.applyMask(original, resizedMask);
  final outputPath = 'test/.output/output_$timestamp.png';
  await img.encodePngFile(outputPath, result); // Save the result as PNG
  print('Save result to: $outputPath');

  session.dispose();
}
