# LiteRT Studio Android reference app

This minimal Android app exercises exported `.litertlm` models with LiteRT-LM's
Kotlin SDK. It deliberately keeps model files out of the APK.

1. Build with `gradle assembleDebug`.
2. Install `app/build/outputs/apk/debug/app-debug.apk`.
3. Open the app and choose a `.litertlm` file through Android's file picker.
4. Select CPU or GPU and run a prompt.

The selected model is copied into private app storage. Engine initialization and
generation run on an IO coroutine, and the engine and conversation are always closed.
