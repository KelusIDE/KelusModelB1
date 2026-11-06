package com.kelus.kani;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import androidx.core.content.FileProvider;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


@CapacitorPlugin(name = "Updater")
public class UpdaterPlugin extends Plugin {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    @PluginMethod
    public void fetchManifest(PluginCall call) {
        String url = call.getString("url", "");
        if (!UpdatePolicy.MANIFEST.equals(url)) { call.reject("Updates come only from Kani's release page."); return; }
        worker.execute(() -> {
            try {
                HttpURLConnection connection = open(url);
                try (InputStream in = connection.getInputStream()) {
                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    byte[] buffer = new byte[8192];
                    for (int n; (n = in.read(buffer)) != -1;) {
                        out.write(buffer, 0, n);
                        if (out.size() > 65536) throw new Exception("Update information is too large.");
                    }
                    JSObject result = new JSObject();
                    result.put("text", new String(out.toByteArray(), StandardCharsets.UTF_8));
                    call.resolve(result);
                } finally { connection.disconnect(); }
            } catch (Exception e) { call.reject("Could not check for updates. Check your connection.", e); }
        });
    }

    @PluginMethod
    public void download(PluginCall call) {
        String url = call.getString("url", ""), sha256 = call.getString("sha256", "");
        Integer version = call.getInt("versionCode");
        Long size = UpdatePolicy.whole(call.getData().opt("size"));
        if (!UpdatePolicy.apkUrl(url) || !sha256.matches("[0-9a-f]{64}") || version == null || version <= 0
                || size == null || !UpdatePolicy.validSize(size)) { call.reject("That update information is invalid."); return; }
        worker.execute(() -> {
            File dir = new File(getContext().getCacheDir(), "updates");
            File part = new File(dir, "kani-" + version + ".download.apk"), file = new File(dir, "kani-" + version + ".apk");
            try {
                clean(dir);
                if (!dir.isDirectory() && !dir.mkdirs()) throw new Exception("No space for the update.");
                if (dir.getUsableSpace() < size + 16L * 1024 * 1024) throw new Exception("Free some storage before downloading this update.");
                HttpURLConnection connection = open(url);
                long loaded = 0, reported = -1;
                try {
                    long contentLength = connection.getContentLengthLong();
                    if (contentLength > 0 && contentLength != size) throw new Exception("The update download has the wrong size.");
                    try (InputStream in = connection.getInputStream(); FileOutputStream out = new FileOutputStream(part)) {
                        byte[] buffer = new byte[65536];
                        for (int n; (n = in.read(buffer)) != -1;) {
                            loaded += n;
                            if (loaded > size) throw new Exception("The update download is larger than expected.");
                            out.write(buffer, 0, n);
                            long percent = loaded * 100 / size;
                            if (percent != reported) {
                                reported = percent;
                                JSObject progress = new JSObject(); progress.put("loaded", loaded); progress.put("total", size);
                                notifyListeners("progress", progress);
                            }
                        }
                        out.getFD().sync();
                    }
                } finally { connection.disconnect(); }
                validate(part, sha256, size, version);
                if (!part.renameTo(file)) throw new Exception("Could not save the downloaded update.");
                JSObject result = new JSObject(); result.put("path", file.getAbsolutePath()); call.resolve(result);
            } catch (Exception e) {
                part.delete(); file.delete();
                call.reject(e.getMessage() != null ? e.getMessage() : "The update could not be downloaded.", e);
            }
        });
    }

    @PluginMethod
    public void install(PluginCall call) {
        String path = call.getString("path", ""), sha256 = call.getString("sha256", "");
        Integer version = call.getInt("versionCode"); Long size = UpdatePolicy.whole(call.getData().opt("size"));
        if (!sha256.matches("[0-9a-f]{64}") || version == null || version <= 0 || size == null || !UpdatePolicy.validSize(size)) {
            call.reject("Download the update first."); return;
        }
        worker.execute(() -> {
            try {
                File file = new File(path).getCanonicalFile();
                File dir = new File(getContext().getCacheDir(), "updates").getCanonicalFile();
                if (!file.isFile() || !dir.equals(file.getParentFile()) || !file.getName().equals("kani-" + version + ".apk")) {
                    throw new Exception("Download the update first.");
                }

                validate(file, sha256, size, version);
                getActivity().runOnUiThread(() -> {
                    try {
                        Context context = getContext(); JSObject result = new JSObject();
                        if (!allowed()) {
                            Intent settings = new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:" + context.getPackageName()));
                            getActivity().startActivity(settings);
                            result.put("started", false); result.put("permission", true);
                        } else {
                            Uri uri = FileProvider.getUriForFile(context, context.getPackageName() + ".fileprovider", file);
                            Intent intent = new Intent(Intent.ACTION_VIEW).setDataAndType(uri, "application/vnd.android.package-archive");
                            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                            getActivity().startActivity(intent);
                            result.put("started", true); result.put("permission", false);
                        }
                        call.resolve(result);
                    } catch (Exception e) { call.reject("Android could not open the installer. Try again.", e); }
                });
            } catch (Exception e) { call.reject(e.getMessage() != null ? e.getMessage() : "Download the update again.", e); }
        });
    }

    @PluginMethod
    public void cleanup(PluginCall call) {
        worker.execute(() -> { clean(new File(getContext().getCacheDir(), "updates")); call.resolve(); });
    }

    @PluginMethod
    public void canInstall(PluginCall call) {
        JSObject result = new JSObject(); result.put("allowed", allowed()); call.resolve(result);
    }

    private boolean allowed() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.O || getContext().getPackageManager().canRequestPackageInstalls();
    }

    private void validate(File file, String hash, long size, int version) throws Exception {
        if (file.length() != size) throw new Exception("The download is incomplete. Try again.");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(file)) {
            byte[] bytes = new byte[65536];
            for (int n; (n = in.read(bytes)) != -1;) digest.update(bytes, 0, n);
        }
        if (!hex(digest.digest()).equals(hash)) throw new Exception("The download was damaged. Try again.");
        PackageManager pm = getContext().getPackageManager();
        int flags = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? PackageManager.GET_SIGNING_CERTIFICATES : PackageManager.GET_SIGNATURES;
        PackageInfo update = pm.getPackageArchiveInfo(file.getPath(), flags);
        PackageInfo installed = pm.getPackageInfo(getContext().getPackageName(), flags);
        if (update == null || !getContext().getPackageName().equals(update.packageName)) throw new Exception("That file is not Kani.");
        long next = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? update.getLongVersionCode() : update.versionCode;
        long current = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? installed.getLongVersionCode() : installed.versionCode;
        if (next != version || next <= current) throw new Exception("That update is not newer than this Kani.");
        if (!UpdatePolicy.sameSigners(signers(installed), signers(update))) throw new Exception("That update is not signed like this Kani.");
    }

    private static byte[][] signers(PackageInfo info) {
        Signature[] signatures = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
            ? info.signingInfo == null ? null : info.signingInfo.getApkContentsSigners() : info.signatures;
        if (signatures == null) return null;
        byte[][] bytes = new byte[signatures.length][];
        for (int i = 0; i < signatures.length; i++) bytes[i] = signatures[i].toByteArray();
        return bytes;
    }

    private static HttpURLConnection open(String address) throws Exception {
        URL url = new URL(address);
        for (int redirects = 0; redirects < 6; redirects++) {
            if (!UpdatePolicy.downloadHost(url.toString())) throw new Exception("Untrusted update download address.");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setInstanceFollowRedirects(false); connection.setUseCaches(false);
            connection.setConnectTimeout(15000); connection.setReadTimeout(30000);
            connection.setRequestProperty("User-Agent", "Kani-Updater");
            connection.setRequestProperty("Accept-Encoding", "identity");
            connection.setRequestProperty("Cache-Control", "no-cache");
            try {
                int status = connection.getResponseCode();
                if (status == 200) return connection;
                if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
                    String location = connection.getHeaderField("Location");
                    if (location == null) throw new Exception("Missing update redirect.");
                    url = new URL(url, location); connection.disconnect(); continue;
                }
                throw new Exception("Update server returned HTTP " + status + ".");
            } catch (Exception e) { connection.disconnect(); throw e; }
        }
        throw new Exception("Too many update redirects.");
    }

    private static void clean(File dir) {
        File[] files = dir.listFiles();
        if (files != null) for (File file : files) if (file.isFile()) file.delete();
    }
    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder();
        for (byte b : bytes) out.append(String.format("%02x", b));
        return out.toString();
    }
    @Override protected void handleOnDestroy() { worker.shutdown(); }
}
