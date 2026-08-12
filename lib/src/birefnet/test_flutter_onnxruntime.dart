import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/cupertino.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnx_demo/src/birefnet/birefnet_helper.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import '../basics.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/12
///
/// # 2026-8-12
/// ```
/// flutter_onnxruntime\windows\flutter_onnxruntime_plugin.cpp(718,1): error C2220: 以下警告被视为错误 [E:\projects\flutter\flutter_onnx_demo\build\windows\x64\plugins\flutter_onnxruntime\flutter_onnxruntime_plugin.vcxproj]
///
/// ---
///
/// # flutter_onnxruntime_plugin.cpp(718,1): error C2220: 以下警告被视为错误
/// # 强制 MSVC 编译器对所有源文件（包括引用的插件）统一采用 UTF-8 编码解析
/// if (MSVC)
///   add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
///   add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
/// endif()
///
/// ```
Future<Uint8List?> testFlutterOnnxruntimeMain() async {
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
  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final rgbImage = original.convert(numChannels: 3);
  final resized = BiRefNetHelper.resizeImage(rgbImage, inputSize, inputSize);
  final inputData = BiRefNetHelper.imageToFloat32NCHW(resized);

  // create inference session
  final ort = OnnxRuntime();
  final session = isMobile
      ? await ort.createSessionFromAsset(modelPath)
      : await ort.createSession(modelPath);

  // specify input with data and shape
  final inputs = {
    session.inputNames.first: await OrtValue.fromList(inputData, shape),
  };

  // start the inference
  final outputs = await session.run(inputs);
  final outputValue = await outputs[session.outputNames.first]!.asList();

  // print output data
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

  for (final element in outputs.values) {
    //print(element?.asList());
    element.dispose();
  }
  session.close();

  //--
  final pngBytes = img.encodePng(result);
  //return decodeImageFromList(pngBytes);
  return pngBytes;
}

Future<ui.Image> decodeImageFromList(Uint8List bytes) async {
  final ui.ImmutableBuffer buffer = await ui.ImmutableBuffer.fromUint8List(
    bytes,
  );
  final ui.Codec codec = await PaintingBinding.instance
      .instantiateImageCodecWithSize(buffer);
  final ui.FrameInfo frameInfo;
  try {
    frameInfo = await codec.getNextFrame();
  } finally {
    codec.dispose();
  }
  return frameInfo.image;
}
