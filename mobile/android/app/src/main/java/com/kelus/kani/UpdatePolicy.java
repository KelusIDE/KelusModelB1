package com.kelus.kani;

import java.net.URI;
import java.util.Arrays;


final class UpdatePolicy {
    static final String RELEASES = "https://github.com/KelusIDE/KelusModelB1/releases/";
    static final String MANIFEST = RELEASES + "latest/download/kani-update.json";
    static final long MAX_APK_BYTES = 1024L * 1024 * 1024;

    static boolean apkUrl(String value) {
        try {
            URI uri = new URI(value);
            return "https".equals(uri.getScheme()) && "github.com".equals(uri.getHost()) && uri.getPort() == -1
                && uri.getRawUserInfo() == null && uri.getRawQuery() == null && uri.getRawFragment() == null
                && uri.getRawPath().matches("/KelusIDE/KelusModelB1/releases/download/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\\.apk")
                && uri.normalize().equals(uri);
        } catch (Exception e) { return false; }
    }

    static boolean downloadHost(String value) {
        try {
            URI uri = new URI(value);
            return "https".equals(uri.getScheme()) && uri.getRawUserInfo() == null && uri.getPort() == -1
                && Arrays.asList("github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com",
                    "github-releases.githubusercontent.com").contains(uri.getHost());
        } catch (Exception e) { return false; }
    }

    static boolean validSize(long size) { return size > 0 && size <= MAX_APK_BYTES; }





    static Long whole(Object value) {
        if (value instanceof Integer || value instanceof Long) return ((Number) value).longValue();
        if (value instanceof Double && (Double) value == Math.rint((Double) value) && Math.abs((Double) value) < 9007199254740992.0) return ((Double) value).longValue();
        return null;
    }
    static boolean sameSigners(byte[][] ours, byte[][] theirs) {
        if (ours == null || theirs == null || ours.length == 0 || ours.length != theirs.length) return false;
        boolean[] matched = new boolean[theirs.length];
        for (byte[] signer : ours) {
            if (signer == null || signer.length == 0) return false;
            boolean found = false;
            for (int i = 0; i < theirs.length; i++) {
                if (!matched[i] && Arrays.equals(signer, theirs[i])) { matched[i] = true; found = true; break; }
            }
            if (!found) return false;
        }
        return true;
    }
}
