"""Shared URL normalization and mirror mappings for git repository URLs."""

GH_MIRRORS = {
    "https://cgit.freedesktop.org/NetworkManager/NetworkManager/": "https://github.com/NetworkManager/NetworkManager",
    "https://cgit.freedesktop.org/udisks/": "https://github.com/storaged-project/udisks",
    "https://gitlab.freedesktop.org/udisks/": "https://github.com/storaged-project/udisks",
    "https://git.videolan.org/ffmpeg.git/": "https://github.com/FFmpeg/FFmpeg",
    "https://git.videolan.org/ffmpeg.git": "https://github.com/FFmpeg/FFmpeg",
    "https://git.videolan.org/gitweb.cgi/ffmpeg.git/": "https://github.com/FFmpeg/FFmpeg",
    "https://gitlab.freedesktop.org/accountsservice/": "https://gitlab.freedesktop.org/accountsservice/accountsservice",
    "https://gitlab.freedesktop.org/cairo/": "https://gitlab.freedesktop.org/cairo/cairo",
    "https://gitlab.freedesktop.org/drm/drm-misc/": "https://gitlab.freedesktop.org/drm/drm-misc/kernel/",
    "https://github.com/GNOME/libgfbgraph/": "https://gitlab.gnome.org/Archive/libgfbgraph",
    "https://github.com/openssl/openssl/git/openssl.git": "https://github.com/openssl/openssl",
    "https://github.com/openssl/openssl/openssl.git": "https://github.com/openssl/openssl",
    "https://github.com/tsingsee/EasyPlayerPro-Win/": "https://github.com/tsingsee/EasyPlayer-RTMP-Win",
    "https://gitlab.freedesktop.org/exempi/": "https://gitlab.freedesktop.org/libopenraw/exempi",
    "https://gitlab.freedesktop.org/fontconfig/": "https://gitlab.freedesktop.org/fontconfig/fontconfig",
    "https://gitlab.freedesktop.org/gypsy/": "https://gitlab.freedesktop.org/archived-projects/gypsy",
    "https://gitlab.freedesktop.org/harfbuzz.old/": "https://github.com/harfbuzz/harfbuzz",
    "https://gitlab.freedesktop.org/harfbuzz/": "https://github.com/harfbuzz/harfbuzz",
    "https://gitlab.freedesktop.org/libbsd/": "https://gitlab.freedesktop.org/libbsd/libbsd",
    "https://gitlab.freedesktop.org/libreoffice/binfilter/": "https://git.libreoffice.org/binfilter",
    "https://gitlab.freedesktop.org/libreoffice/core/": "https://git.libreoffice.org/core",
    "https://gitlab.freedesktop.org/libreoffice/filters/": "https://git.libreoffice.org/core",
    "https://gitlab.freedesktop.org/polkit/": "https://github.com/polkit-org/polkit",
    "https://gitlab.freedesktop.org/systemd/systemd/": "https://github.com/systemd/systemd",
    "https://gitlab.freedesktop.org/virglrenderer/": "git@ssh.gitlab.freedesktop.org:virgl/virglrenderer.git",
    "https://gitlab.freedesktop.org/pixman/": "https://gitlab.freedesktop.org/pixman/pixman",
    "https://sourceware.org/git/gitweb.cgiglibc.git": "https://sourceware.org/git/glibc.git",
    "https://sourceware.org/git/gitweb.cgibinutils-gdb.git": "https://sourceware.org/git/binutils-gdb.git",
    "https://kernel.googlesource.com/pub/scm/git.git/": "https://kernel.googlesource.com/pub/scm/git/git.git/",
    "https://git.openssl.org": "https://github.com/openssl/openssl",
    "https://git.openssl.org/openssl.git": "https://github.com/openssl/openssl",
    "https://git.openssl.org/git/openssl.git": "https://github.com/openssl/openssl",
    "https://chromium.googlesource.com/chromium/src/": "https://github.com/chromium/chromium",
}


def normalize_git_url(git_url: str) -> str:
    """Extract the repo base URL from a megavul git_url (which includes /commit/hash)."""
    for sep in ("/commit/", "/commits/", "/+/"):
        idx = git_url.find(sep)
        if idx != -1:
            git_url = git_url[:idx]
            break
    return git_url.rstrip("/")


def normalize_project_url(url: str) -> str:
    """Normalize a git hosting URL to a canonical form, applying mirror mappings."""
    url = url.replace("http://", "https://")
    parts = url.replace("https://", "").split("/", 1)
    hosting = "https://" + parts[0]
    repo_path = parts[1] if len(parts) > 1 else ""
    hosting = hosting.replace("git.kernel.org", "kernel.googlesource.com/pub/scm")
    repo_path = repo_path.replace("cgit", "git")
    for sep in ("commit", "+", ";h=", ";a="):
        repo_path = repo_path.split(sep)[0]
    repo_path = repo_path.replace("/-/", "/")
    project_url = f"{hosting}/{repo_path}"
    project_url = project_url.replace("https://git.freedesktop.org/", "https://gitlab.freedesktop.org/")
    project_url = project_url.replace("https://cgit.freedesktop.org/", "https://gitlab.freedesktop.org/")
    project_url = project_url.replace("gitweb/?p=", "git/")
    project_url = project_url.replace("/pub/scm/git/", "/pub/scm/")
    project_url = project_url.replace("/pub/scm/pub/scm/", "/pub/scm/")
    project_url = project_url.replace("?p=", "")
    project_url = project_url.replace("?a=", "")
    project_url = project_url.replace("%2F", "/")
    for old, new in GH_MIRRORS.items():
        if project_url.startswith(old):
            project_url = new + project_url[len(old):]
            break
    return project_url


def url_to_dst_folder(url: str) -> str:
    """Convert a URL to a filesystem-safe folder name."""
    url = normalize_project_url(url)
    folder = url.replace("https://", "").replace("http://", "")
    folder = folder.replace("/", "_").strip("_").replace(".", "_")
    return folder
