import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

///
/// @author <a href="mailto:angcyo@126.com">angcyo</a>
/// @date 2026/08/12
///
///
bool get isMobile => Platform.isIOS || Platform.isAndroid;

bool get isMacOS => Platform.isMacOS;

Future<Uint8List> readAssetsBytes(String key) async =>
    (await rootBundle.load(key)).buffer.asUint8List();

Future<Directory> cacheDirectory() async {
  Directory? directory;
  try {
    if (defaultTargetPlatform == .android) {
      // android: /storage/emulated/0/Android/data/com.angcyo.flutter3_abc/cache
      try {
        directory = (await getExternalCacheDirectories())?.firstOrNull;
      } catch (e) {}
    } else if (defaultTargetPlatform == .iOS ||
        defaultTargetPlatform == .macOS) {
      try {
        // android: /data/user/0/com.angcyo.flutter3_abc/cache
        // ios:
        // macos: /Users/angcyo/Library/Containers/com.laserabc.laserabcFactoryTools/Data/Library/Caches/G01%20V41.01.0004.ydb
        directory = await getTemporaryDirectory();
      } catch (e) {}
    } else if (defaultTargetPlatform == .windows) {
      try {
        //C:\Users\Administrator\AppData\Roaming\com.angcyo.flutter3.desktop.abc\flutter3_desktop_abc_pn
        // C:\Users\ADMINI~1\AppData\Local\Temp
        //final dir = await getTemporaryDirectory();

        // C:\Users\Administrator\AppData\Local\com.laser.abc.beeb.desktop.app\Laserabc Beeb Desktop
        //final cache = await getApplicationCacheDirectory();

        // C:\Users\Administrator\Downloads
        //final downloads = await getDownloadsDirectory();

        //Windows: getLibraryPath() has not been implemented.
        //final library = await getLibraryDirectory();

        //C:\Users\Administrator\AppData\Local\com.laser.abc.beeb.desktop.app\Laserabc Beeb Desktop
        directory = await getApplicationCacheDirectory();
      } catch (e) {
        //l.e(e);
        try {
          //C:\Users\Administrator\AppData\Roaming\com.laser.abc.beeb.desktop.app\Laserabc Beeb Desktop
          directory = await getApplicationSupportDirectory();
        } catch (e) {}
      }
    }
    directory ??= await getTemporaryDirectory();
  } catch (e) {
    //l.e(e);
  }
  //Directory: '/data/user/0/com.angcyo.lp.image.ffi.lp_image_handle_ffi_example/code_cache'
  return directory ??
      Directory.systemTemp; //C:\Users\ADMINI~1\AppData\Local\Temp
}

Future<String> cacheFilePath(String fileName) async {
  final folder = await cacheDirectory();
  try {
    folder.createSync(recursive: true);
  } catch (e) {
    //print(e);
  }
  return p.join(folder.path, fileName);
}
