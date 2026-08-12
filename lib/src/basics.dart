import 'dart:io';

import 'package:flutter/services.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/12
///
///
bool get isMobile => Platform.isIOS || Platform.isAndroid;

bool get isMacOS => Platform.isMacOS;

Future<Uint8List> readAssetsBytes(String key) async =>
    (await rootBundle.load(key)).buffer.asUint8List();
