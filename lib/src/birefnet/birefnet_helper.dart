import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/11
///
/// https://huggingface.co/ZhengPeng7/BiRefNet_lite
///
/// https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1
///
/// https://huggingface.co/onnx-community/BiRefNet_lite-ONNX
///
class BiRefNetHelper {
  BiRefNetHelper._();

  //MARK: - input

  /// RGB Image -> Float32 NCHW
  /// ```
  /// //Mean Centering（均值中心化）
  /// final meanR = 0.485;
  /// final meanG = 0.456;
  /// final meanB = 0.406;
  ///
  /// //Standardization（标准化）
  /// final stdR = 0.229;
  /// final stdG = 0.224;
  /// final stdB = 0.225;
  /// ```
  /// 1024*1024*3=3,145,728
  static Float32List imageToFloat32NCHW(
    img.Image image, {
    //Mean Centering（均值中心化）
    double meanR = 0.485,
    double meanG = 0.456,
    double meanB = 0.406,
    //Standardization（标准化）
    double stdR = 0.229,
    double stdG = 0.224,
    double stdB = 0.225,
  }) {
    final width = image.width;
    final height = image.height;

    final planeSize = width * height;

    final result = Float32List(3 * planeSize);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);

        final r = pixel.r.toDouble() / 255.0;
        final g = pixel.g.toDouble() / 255.0;
        final b = pixel.b.toDouble() / 255.0;

        final index = y * width + x;

        // NCHW
        //
        // R channel
        result[index] = (r - meanR) / stdR;

        // G channel
        result[planeSize + index] = (g - meanG) / stdG;

        // B channel
        result[planeSize * 2 + index] = (b - meanB) / stdB;
      }
    }

    return result;
  }

  //MARK: - output

  /// 从 ONNX output 中提取 Float32, 对应 logits 的值
  ///
  /// ```
  /// 输出的是前景/背景的置信度 logits
  ///  logit
  ///   ↓
  ///  sigmoid
  ///   ↓
  ///  0.0 ~ 1.0
  ///   ↓
  ///  ×255
  ///   ↓
  ///  Alpha
  /// ```
  static Float32List extractOutput(dynamic value, int inputSize) {
    /*
     * 不同版本的 onnxruntime Dart binding
     * 对 output.value 的包装方式可能不同。
     *
     * 常见情况：
     *
     * List<List<List<List<double>>>>
     *
     * 或：
     *
     * List<double>
     */

    final result = Float32List(inputSize * inputSize);

    int index = 0;

    void walk(dynamic data) {
      if (data is List) {
        for (final item in data) {
          walk(item);
        }
      } else if (data is num) {
        if (index < result.length) {
          result[index++] = data.toDouble();
        }
      }
    }

    walk(value);

    if (index != result.length) {
      throw Exception(
        'BiRefNet output size error: '
        'expected ${result.length}, got $index',
      );
    }

    return result;
  }

  /// 将 logits 转换成 Sigmoid
  /// logits -> 0~1
  static Float32List sigmoid(Float32List logits, int inputSize) {
    final mask = Float32List(inputSize * inputSize);
    for (int i = 0; i < mask.length; i++) {
      final x = logits[i];

      // sigmoid
      final value = 1.0 / (1.0 + math.exp(-x));

      mask[i] = value;
    }
    return mask;
  }

  /// Mask Float32 -> 灰度图片
  /// ```
  /// final resizedMask = img.copyResize(
  ///   maskImage,
  ///   width: originalWidth,
  ///   height: originalHeight,
  ///   interpolation: img.Interpolation.linear,
  /// );
  /// ```
  static img.Image maskToImage(Float32List mask, int width, int height) {
    final result = img.Image(width: width, height: height, numChannels: 1);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final index = y * width + x;

        final value = mask[index].clamp(0.0, 1.0);

        final gray = (value * 255.0).round();

        result.setPixelRgb(x, y, gray, gray, gray);
      }
    }

    return result;
  }

  /// 将 mask 设置到原始图片 Alpha
  static img.Image applyMask(img.Image original, img.Image mask) {
    final result = original.convert(numChannels: 4);

    for (int y = 0; y < result.height; y++) {
      for (int x = 0; x < result.width; x++) {
        final maskPixel = mask.getPixel(x, y);

        final alpha = maskPixel.r.toDouble().round().clamp(0, 255);

        final pixel = result.getPixel(x, y);

        result.setPixelRgba(
          x,
          y,
          pixel.r.toInt(),
          pixel.g.toInt(),
          pixel.b.toInt(),
          alpha,
        );
      }
    }

    return result;
  }

  //MARK: - api

  /// 调整图片大小
  static img.Image resizeImage(
    img.Image original,
    int width,
    int height, {
    bool? maintainAspect,
    img.Interpolation interpolation = .linear,
  }) {
    return img.copyResize(
      original,
      width: width,
      height: height,
      maintainAspect: maintainAspect,
      interpolation: img.Interpolation.linear,
    );
  }
}
