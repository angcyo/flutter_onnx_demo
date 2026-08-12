import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/12
///
class YoloHelper {
  YoloHelper._();

  /// YOLOv8n-OBB 官方 DOTA 模型有 15 类
  static const int numClasses = 15;

  static const double confThreshold = 0.25;

  static const double iouThreshold = 0.45;

  //MARK: input

  /// 预处理图像
  static PreprocessResult preprocessImage(img.Image image, int inputSize) {
    final originalWidth = image.width;
    final originalHeight = image.height;

    final scale =
        inputSize /
        (originalWidth > originalHeight ? originalWidth : originalHeight);

    final resizedWidth = (originalWidth * scale).round();
    final resizedHeight = (originalHeight * scale).round();

    final resized = img.copyResize(
      image,
      width: resizedWidth,
      height: resizedHeight,
      interpolation: img.Interpolation.linear,
    );

    final canvas = img.Image(width: inputSize, height: inputSize);

    // YOLO letterbox 默认填充 114
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));

    final padX = (inputSize - resizedWidth) / 2;
    final padY = (inputSize - resizedHeight) / 2;

    img.compositeImage(canvas, resized, dstX: padX.round(), dstY: padY.round());

    final area = inputSize * inputSize;

    final output = Float32List(3 * area);

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = canvas.getPixel(x, y);

        final index = y * inputSize + x;

        output[index] = pixel.r / 255.0;

        output[area + index] = pixel.g / 255.0;

        output[area * 2 + index] = pixel.b / 255.0;
      }
    }

    return PreprocessResult(
      input: output,
      scale: scale,
      padX: padX,
      padY: padY,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );
  }

  //MARK: output

  /// 解码YOLOv8的输出数据
  /// 旋转目标检测（Oriented Object Detection / OBB）。
  static List<RotatedObject> decodeObb(
    dynamic output, {
    double padX = 0,
    double padY = 0,
    double scale = 1.0,
  }) {
    /*
     * YOLOv8 OBB exported output:
     *
     * [1, 4 + numClasses + 1, numPredictions]
     *
     * 4 -> cx cy w h
     * numClasses -> class score
     * 1 -> angle
     *
     * 对 YOLOv8n-OBB:
     *
     * [1, 20, 8400]
     *
     * 因此：
     *
     * 20 = 4 + 15 + 1
     */

    final values = _flatten(output);

    final channels = 4 + numClasses + 1;

    final numPredictions = values.length ~/ channels;

    final candidates = <RotatedObject>[];

    for (int i = 0; i < numPredictions; i++) {
      final cx = values[i];
      final cy = values[numPredictions + i];
      final w = values[numPredictions * 2 + i];
      final h = values[numPredictions * 3 + i];

      int bestClass = -1;
      double bestScore = 0;

      for (int c = 0; c < numClasses; c++) {
        final score = values[numPredictions * (4 + c) + i];

        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < confThreshold) {
        continue;
      }

      final rawAngle = values[numPredictions * (4 + numClasses) + i];

      /*
       * YOLOv8 OBB angle decode:
       *
       * angle = (sigmoid(rawAngle) - 0.25) * PI
       */
      final angle = (_sigmoid(rawAngle) - 0.25) * pi;

      /*
       * 模型坐标 -> 原图坐标
       *
       * 模型输入经过：
       *
       * original
       *   ↓
       * scale
       *   ↓
       * letterbox
       *
       * 所以反向：
       *
       * x_original = (x - padX) / scale
       */
      final originalCx = (cx - padX) / scale;

      final originalCy = (cy - padY) / scale;

      final originalW = w / scale;

      final originalH = h / scale;

      if (originalW <= 0 || originalH <= 0) {
        continue;
      }

      candidates.add(
        RotatedObject(
          cx: originalCx,
          cy: originalCy,
          width: originalW,
          height: originalH,
          angle: angle,
          confidence: bestScore,
          classId: bestClass,
        ),
      );
    }

    return _nmsRotated(candidates);
  }

  static double _sigmoid(double x) {
    if (x >= 0) {
      final z = exp(-x);
      return 1 / (1 + z);
    } else {
      final z = exp(x);
      return z / (1 + z);
    }
  }

  static Float32List _flatten(dynamic value) {
    if (value is Float32List) {
      return value;
    }

    if (value is List) {
      final result = <double>[];

      void recursive(dynamic v) {
        if (v is List) {
          for (final item in v) {
            recursive(item);
          }
        } else {
          result.add((v as num).toDouble());
        }
      }

      recursive(value);

      return Float32List.fromList(result);
    }

    throw Exception('Unsupported ONNX output type: ${value.runtimeType}');
  }

  static List<RotatedObject> _nmsRotated(List<RotatedObject> objects) {
    /*
     * 这里为了保持示例完整，先使用
     * axis-aligned IoU 做 NMS。
     *
     * 如果物体旋转角度比较大，
     * 建议后续改成真正的 rotated IoU NMS。
     */

    final sorted = [...objects]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <RotatedObject>[];

    for (final object in sorted) {
      bool keep = true;

      for (final selectedObject in selected) {
        if (object.classId != selectedObject.classId) {
          continue;
        }

        final iou = _axisAlignedIoU(object, selectedObject);

        if (iou > iouThreshold) {
          keep = false;
          break;
        }
      }

      if (keep) {
        selected.add(object);
      }
    }

    return selected;
  }

  static double _axisAlignedIoU(RotatedObject a, RotatedObject b) {
    final aLeft = a.cx - a.width / 2;
    final aRight = a.cx + a.width / 2;
    final aTop = a.cy - a.height / 2;
    final aBottom = a.cy + a.height / 2;

    final bLeft = b.cx - b.width / 2;
    final bRight = b.cx + b.width / 2;
    final bTop = b.cy - b.height / 2;
    final bBottom = b.cy + b.height / 2;

    final left = max(aLeft, bLeft);
    final right = min(aRight, bRight);
    final top = max(aTop, bTop);
    final bottom = min(aBottom, bBottom);

    final intersectionWidth = max(0.0, right - left);

    final intersectionHeight = max(0.0, bottom - top);

    final intersection = intersectionWidth * intersectionHeight;

    final areaA = a.width * a.height;
    final areaB = b.width * b.height;

    final union = areaA + areaB - intersection;

    if (union <= 0) {
      return 0;
    }

    return intersection / union;
  }
}

class PreprocessResult {
  final Float32List input;

  final double scale;
  final double padX;
  final double padY;

  final int originalWidth;
  final int originalHeight;

  PreprocessResult({
    required this.input,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.originalWidth,
    required this.originalHeight,
  });
}

class RotatedObject {
  final double cx;
  final double cy;
  final double width;
  final double height;

  /// 弧度
  final double angle;

  /// 置信度
  final double confidence;

  /// 类别ID
  final int classId;

  RotatedObject({
    required this.cx,
    required this.cy,
    required this.width,
    required this.height,
    required this.angle,
    required this.confidence,
    required this.classId,
  });

  List<Point<double>> get corners {
    final cosA = cos(angle);
    final sinA = sin(angle);

    final hw = width / 2;
    final hh = height / 2;

    final points = <Point<double>>[];

    final localPoints = [
      Point<double>(-hw, -hh),
      Point<double>(hw, -hh),
      Point<double>(hw, hh),
      Point<double>(-hw, hh),
    ];

    for (final p in localPoints) {
      final x = p.x * cosA - p.y * sinA + cx;
      final y = p.x * sinA + p.y * cosA + cy;

      points.add(Point<double>(x, y));
    }

    return points;
  }

  @override
  String toString() {
    return 'RotatedObject('
        'cx=$cx, '
        'cy=$cy, '
        'w=$width, '
        'h=$height, '
        'angle=$angle, '
        'conf=$confidence, '
        'classId=$classId'
        ')';
  }
}
