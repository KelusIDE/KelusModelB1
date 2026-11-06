# Kani for Android and iOS

The mobile app lives here. It packages Kani's own model exports, React interface, and Capacitor native projects. Model training and the web/CLI releases are separate.

## Android updates from GitHub

Kani **0.5.0 has no updater**. Install the latest APK over it once; do **not** uninstall first. The application ID remains `com.kelus.kani`, and the signing certificate matches the supplied 0.5.0 APK, so Android can preserve app data. Versions 0.5.1 and later contain the updater.

[Latest Android release](https://github.com/KelusIDE/KelusModelB1/releases/latest)

1. On launch, Kani checks `https://github.com/KelusIDE/KelusModelB1/releases/latest/download/kani-update.json`. It checks again after six hours when returning to the app. **Settings → Updates** checks immediately.
2. A newer Android `versionCode` shows an offer. **Later** dismisses it for the session. Nothing downloads until **Update** is tapped.
3. The native downloader follows HTTPS redirects only through GitHub's asset hosts. It checks declared size, SHA-256, application ID, signing certificate, and version; files are staged before being offered to the installer. Those checks run again before installation.
4. Android may ask the person to enable **Allow from this source** for Kani. Returning without allowing it leaves the app usable; it does not reopen settings in a loop.
5. Android's installer still requires confirmation. Cancelling returns to the update offer and reuses the completed download. Updates replace the existing app rather than creating another Kani.

Automatic **checking**, user-confirmed **installation**. This is a sideload update channel, not a silent installer. iOS does not use this APK updater; its native project still needs Xcode, signing, and an Apple-supported distribution route.

## Build and release

Use Node 22+, JDK 21, and Android SDK 36. On this Mac, the build script detects the SDK and JDK paths. `npm ci` restores pinned dependencies.

```sh
cd mobile
npm ci
npm test
npm run apk
```

The resulting APK is `artifacts/kani-<version>-debug.apk`. Existing packaged models live in ignored `public/models/`; preserve these when rebuilding. `npm run model` packages model exports according to `scripts/package-model.mjs`.

For each release, increase **both** `version` in `package.json` and `versionCode` / `versionName` in `android/app/build.gradle`; run `npm install --package-lock-only --ignore-scripts` to update the lockfile version. Keep app data keys and the application ID stable.

```sh
# Local build, signature/version verification, APK + update manifest + release notes; no publication.
npm run release:prepare -- "What changed"

# Runs tests and builds again, verifies the actual APK, then publishes via signed-in gh.
npm run release:publish -- "What changed" "Another change"
```

The publisher reads the version **inside the APK**, verifies its signature against `signing-certificate.json`, and refuses an existing published tag or a version no newer than the public channel. It uploads the APK and `kani-update.json` to a draft, verifies GitHub's asset checksums/sizes, then publishes and verifies the public latest manifest. It does not replace old published assets. A failed upload leaves the draft unpublished and the previous latest intact.

If a draft upload was interrupted, inspect it with `gh release view v<version> --repo KelusIDE/KelusModelB1 --json isDraft,assets`. The publisher will only reuse a draft whose two assets exactly match the verified local build; it will not silently overwrite different assets. Complete a missing upload using `gh release upload` only after checking the draft and local manifest. If the local build changed, use a new version instead.

Keep `kani-update.json` attached to the release marked **Latest**. Publishing an unrelated release as Latest without that asset breaks this existing channel.

**Signing:** the supplied 0.5.0 and current sideload builds use this Mac's Android debug certificate. Preserve the existing private key in `~/.android/debug.keystore` securely; never commit or upload it. A different machine's debug key cannot update these installations. `signing-certificate.json` contains only its public fingerprint. Switching to a production key requires a separate migration plan.

Releases are built from the local mobile working tree. This repository currently has uncommitted mobile/source work; GitHub's tag on `main` does not capture that source. The release notes disclose this. The publisher does not commit or push unrelated work.

## Verification

`npm test` covers the mobile model/history paths plus updater lifecycle and release validation. Native checks:

```sh
cd android
./gradlew :app:testDebugUnitTest --no-daemon --max-workers=2
```

Before distributing beyond testing, try a real in-place upgrade on Android: keep a saved chat, install the new APK over the old one, check the saved chat, then exercise denial/allow/cancel in Android's installer. Automated tests and APK verification do not substitute for this device test.

## iOS

The iPhone/iPad project uses **CocoaPods**, with CocoaPods 1.17.0 pinned in `Gemfile.lock` and native dependencies pinned in `ios/App/Podfile.lock`. Open **`ios/App/App.xcworkspace`**, not the `.xcodeproj`. The app remains `com.kelus.kani`; its current version is 0.5.3 (build 8).

Requirements: Node 22+, a modern Ruby with Bundler (this setup was verified with Ruby 4.0.6 / Bundler 4.0.16), **full Xcode 26+ with the iOS SDK**, and iOS **15.5+** on the device. The higher device minimum is required by the existing ML Kit plugins. See [Capacitor's environment requirements](https://capacitorjs.com/docs/getting-started/environment-setup) and [ML Kit's iOS requirements](https://developers.google.com/ml-kit/migration/ios).

```sh
cd mobile
npm ci
bundle config set --local path vendor/bundle
bundle install
npm run sync:ios
npm run ios
```

After installing Xcode, launch it once, install its iOS platform support, and select it under **Xcode → Settings → Locations → Command Line Tools**. In the workspace, select the **App** scheme, choose your team under **Signing & Capabilities**, connect an iPhone, and Run. For a distributable build, select a device destination and use **Product → Archive → Distribute App** with the appropriate Apple signing/distribution setup. This does not silently install or update iOS apps through the Android GitHub APK updater.

For a command-line compile check:

```sh
npm run ios:build
```

This checks for full Xcode, builds/syncs the interface, and compiles an **unsigned device `.app`** at `artifacts/ios-derived/Build/Products/Debug-iphoneos/App.app`. It is not an installable IPA. Use Xcode signing to install/export. A device build is the default because the installed ML Kit binaries exclude arm64 simulators; do not remove that exclusion to force a simulator build.

To package a Release device build for a sideload tool to re-sign:

```sh
npm run ios:build -- --ipa
```

The output is `artifacts/kani-VERSION-unsigned.ipa`, containing `Payload/Kani.app`.
It has no Apple provisioning profile or distribution signature. Import it into
your sideload tool to sign and install; the unsigned file does not install by itself.

For the 0.5.4 parity build, the isolated GitHub macOS workflow pins the native
version to 0.5.4/build 9 and verifies the published APK's SHA-256 before
extracting its web assets. `--apk FILE` stages those assets in a new directory
and syncs them into the iOS project without rebuilding the web UI. Its receipt
records every original asset hash; the workflow verifies the assets again in
the final IPA. Native bridge placeholders are regenerated for iOS. Later
uncommitted mobile work is excluded from that version's build branch.

Firebase's **Kani iOS** app is registered in the existing `kelus-ide` project with the same bundle ID. `GoogleService-Info.plist` is included in the app target, and its reversed client ID is registered for Google sign-in callbacks in both the scene and app delegates. Firebase's plugin initializes the native SDK from that file. The Google subspec stays outside `capacitor_pods` in the Podfile, so `cap sync` preserves it. No service-account key belongs in this app.

**Verified on this Mac:** CocoaPods installed all 51 pods; the dependency lock matches the installed pods; the web build and mobile tests pass; project/plist validation and Swift syntax parsing pass. **Not yet verified:** compiling/linking or running on an iPhone, because this Mac only has Command Line Tools, not full Xcode. The initial `cap sync ios` installed pods but failed at its final `xcodebuild clean`; standalone `bundle exec pod install --project-directory=ios/App` and `npx cap copy ios` succeeded. Build logs are in ignored `artifacts/ios-*.log`.

`npm run sync:android` and `npm run apk` only sync Android, so they do not require CocoaPods or Xcode. `npm run sync` still syncs both platforms. When releasing, also update iOS `MARKETING_VERSION` and `CURRENT_PROJECT_VERSION` in the Xcode target.

**Migration rollback:** the original SPM project/configuration is preserved locally in `artifacts/ios-spm-backup-20261025-232333/`. Before rolling back, save any newer iOS changes. Run `bundle exec pod deintegrate ios/App/App.xcodeproj`, move the CocoaPods workspace/Podfile/lock aside, and restore the backed-up `ios/` files (including `App/CapApp-SPM`) and `package.json`. Keep the current assets/models. The old SPM setup does not include the speech-recognition or ML Kit plugins. The Firebase registration can remain; it does not affect Android or the website.

References: [GitHub release asset links](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases), [draft release publishing](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository), [Android install permission](https://developer.android.com/reference/android/content/pm/PackageManager#canRequestPackageInstalls()).
