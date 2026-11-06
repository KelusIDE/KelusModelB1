package com.kelus.kani;
import org.junit.Test;
import static org.junit.Assert.*;

public class UpdatePolicyTest {
    @Test public void onlyThisRepositorysApksAreAccepted() {
        assertTrue(UpdatePolicy.apkUrl(UpdatePolicy.RELEASES + "download/v0.5.3/kani-0.5.3.apk"));
        for (String url : new String[] { "http://github.com/KelusIDE/KelusModelB1/releases/download/v1/kani.apk",
            UpdatePolicy.RELEASES + "download/../../other/kani.apk", UpdatePolicy.RELEASES + "download/%2e%2e/kani.apk",
            UpdatePolicy.RELEASES + "download/v1/kani.apk?redirect=elsewhere", UpdatePolicy.RELEASES + "download/v1/kani.apk#anything",
            "https://github.com.evil.test/KelusIDE/KelusModelB1/releases/download/v1/kani.apk" }) assertFalse(url, UpdatePolicy.apkUrl(url));
    }
    @Test public void redirectsCannotDowngradeOrLeaveGitHubDownloads() {
        assertTrue(UpdatePolicy.downloadHost("https://release-assets.githubusercontent.com/asset?signature=abc"));
        assertFalse(UpdatePolicy.downloadHost("http://release-assets.githubusercontent.com/asset"));
        assertFalse(UpdatePolicy.downloadHost("https://github.com.evil.test/asset"));
        assertFalse(UpdatePolicy.downloadHost("https://user:password@github.com/asset"));
    }
    @Test public void absentOrDifferentSignersAreRejectedAndOrderingIsIrrelevant() {
        byte[][] signers = { {1, 2}, {3, 4} };
        assertTrue(UpdatePolicy.sameSigners(signers, new byte[][] {{3, 4}, {1, 2}}));
        assertFalse(UpdatePolicy.sameSigners(signers, null));
        assertFalse(UpdatePolicy.sameSigners(null, signers));
        assertFalse(UpdatePolicy.sameSigners(new byte[0][], new byte[0][]));
        assertFalse(UpdatePolicy.sameSigners(signers, new byte[][] {{1, 2}, {1, 2}}));
    }
    @Test public void sizesArriveAsWholeNumbersOfAnyWidth() {
        assertEquals(Long.valueOf(170525989L), UpdatePolicy.whole(170525989));
        assertEquals(Long.valueOf(3000000000L), UpdatePolicy.whole(3000000000L));
        assertEquals(Long.valueOf(170525989L), UpdatePolicy.whole(170525989.0));
        assertNull(UpdatePolicy.whole(1.5));
        assertNull(UpdatePolicy.whole("170525989"));
        assertNull(UpdatePolicy.whole(null));
    }
    @Test public void downloadsHaveABoundedPositiveSize() {
        assertTrue(UpdatePolicy.validSize(170000000));
        assertFalse(UpdatePolicy.validSize(0));
        assertFalse(UpdatePolicy.validSize(UpdatePolicy.MAX_APK_BYTES + 1));
    }
}
