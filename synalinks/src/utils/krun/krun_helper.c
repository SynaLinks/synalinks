// License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
//
// synalinks-krun: boot one libkrun microVM, run one command in it, exit with
// its exit code. The macOS backend of MirageSandbox spawns this per Python
// execution (a fresh VM per `python3`, as Mirage spawns a fresh interpreter).
//
// It exists as a separate binary, rather than ctypes calls from Python, because
// creating a VM needs the `com.apple.security.hypervisor` entitlement on the
// *calling executable*, and the Python interpreter does not carry it. This
// binary is ad-hoc signed with that entitlement and nothing else.
//
// Secure by default, the caller opts *in* to every capability:
//   * the root filesystem is exposed read-only;
//   * the guest gets no network: TSI (libkrun's transparent proxying of guest
//     sockets through the host network) is disabled unless --net is passed;
//   * the guest environment is exactly the --env entries, never the host's
//     (a NULL envp would copy the host environment, API keys included);
//   * host sockets are reachable only through explicit --vsock port mappings;
//   * with --profile, the helper confines *itself* with that Seatbelt profile
//     before creating the VM, so a guest that escapes into this process (a
//     bug in libkrun's device emulation) is still boxed in on the host.
//
// Usage:
//   synalinks-krun --root DIR [--share TAG=DIR]... [--share-ro TAG=DIR]...
//                  [--vsock PORT=SOCKET]...
//                  [--cpus N] [--mem MIB] [--env K=V]... [--rlimit R=C:M]...
//                  [--mountpoint PATH]... [--workdir DIR] [--net]
//                  [--profile SBPL_FILE] [--root-writable] -- EXEC [ARG]...
//
// --root-writable is for provisioning only (compiling the image's bytecode
// once, as trusted code); every sandboxed run gets the read-only root.

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libkrun.h>
#include <sandbox.h>

#define MAX_ITEMS 64

static void die(const char *what, int rc) {
    fprintf(stderr, "synalinks-krun: %s failed: %s\n", what, strerror(-rc));
    exit(125);
}

// Top-level directories already added. Re-adding one does not fail: it replaces
// the first, so two paths under the same top level (every /private/... pair)
// must share it.
static char *added[MAX_ITEMS];
static int n_added = 0;

// Add the top-level directory of PATH (absolute) to the read-only root as an
// empty virtual directory. The guest mounts a tmpfs on it and creates the rest
// of PATH there, so it can mount a share at the host's own path. Only the top
// level is virtual: nested virtual directories do not survive a sibling mount.
static void add_mountpoint(int ctx, const char *path) {
    char top[256];
    const char *slash = path[0] == '/' ? strchr(path + 1, '/') : NULL;
    size_t len = slash ? (size_t)(slash - path - 1) : strlen(path) - 1;
    if (path[0] != '/' || len == 0 || len >= sizeof(top)) {
        fprintf(stderr, "synalinks-krun: --mountpoint must be an absolute path\n");
        exit(2);
    }
    memcpy(top, path + 1, len);
    top[len] = '\0';
    for (int k = 0; k < n_added; k++)
        if (strcmp(added[k], top) == 0) return;
    if (n_added == MAX_ITEMS) die("--mountpoint", -E2BIG);
    int rc = krun_fs_add_overlay_dir(ctx, KRUN_FS_ROOT_TAG, top, 040755);
    if (rc && rc != -EEXIST) die("--mountpoint", rc);
    added[n_added++] = strdup(top);
}

// Confine this process with the Seatbelt profile in PATH, before anything else.
// The profile is read first: once it applies, even its own file may be denied.
static void apply_profile(const char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) die("--profile", -errno);
    static char sbpl[1 << 16];
    size_t n = fread(sbpl, 1, sizeof(sbpl) - 1, f);
    fclose(f);
    if (n == sizeof(sbpl) - 1) die("--profile", -E2BIG);
    sbpl[n] = '\0';
    char *err = NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    // Deprecated without replacement, yet the documented way to sandbox a
    // non-App-Store process; Chrome, Nix and Bazel rely on it too.
    if (sandbox_init(sbpl, 0, &err) != 0) {
#pragma clang diagnostic pop
        fprintf(stderr, "synalinks-krun: --profile failed: %s\n", err ? err : "unknown");
        exit(125);
    }
}

static void usage(void) {
    fprintf(stderr,
            "usage: synalinks-krun --root DIR [--share TAG=DIR]... [--share-ro TAG=DIR]... "
            "[--vsock PORT=SOCKET]... [--cpus N] [--mem MIB] [--env K=V]... "
            "[--rlimit R=C:M]... [--mountpoint PATH]... [--workdir DIR] [--net] "
            "[--profile SBPL_FILE] [--root-writable] -- EXEC [ARG]...\n");
    exit(2);
}

// Split "KEY=VALUE" in place; returns VALUE, or exits on a malformed item.
static char *split_eq(char *item) {
    char *eq = strchr(item, '=');
    if (eq == NULL || eq == item) usage();
    *eq = '\0';
    return eq + 1;
}

int main(int argc, char **argv) {
    const char *root = NULL, *workdir = NULL, *profile = NULL;
    char *shares[MAX_ITEMS], *ro_shares[MAX_ITEMS], *vsocks[MAX_ITEMS], *mountpoints[MAX_ITEMS];
    const char *envs[MAX_ITEMS + 1], *rlimits[MAX_ITEMS + 1];
    int n_shares = 0, n_ro_shares = 0, n_vsocks = 0, n_envs = 0, n_rlimits = 0, n_mountpoints = 0;
    int cpus = 1, mem = 512;
    bool net = false, root_writable = false;
    int i = 1;

    for (; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--") == 0) { i++; break; }
        if (i + 1 >= argc && strcmp(a, "--net") != 0 && strcmp(a, "--root-writable") != 0)
            usage();
        if (strcmp(a, "--root") == 0) root = argv[++i];
        else if (strcmp(a, "--workdir") == 0) workdir = argv[++i];
        else if (strcmp(a, "--profile") == 0) profile = argv[++i];
        else if (strcmp(a, "--cpus") == 0) cpus = atoi(argv[++i]);
        else if (strcmp(a, "--mem") == 0) mem = atoi(argv[++i]);
        else if (strcmp(a, "--net") == 0) net = true;
        else if (strcmp(a, "--root-writable") == 0) root_writable = true;
        else if (strcmp(a, "--share") == 0 && n_shares < MAX_ITEMS) shares[n_shares++] = argv[++i];
        else if (strcmp(a, "--share-ro") == 0 && n_ro_shares < MAX_ITEMS)
            ro_shares[n_ro_shares++] = argv[++i];
        else if (strcmp(a, "--vsock") == 0 && n_vsocks < MAX_ITEMS) vsocks[n_vsocks++] = argv[++i];
        else if (strcmp(a, "--env") == 0 && n_envs < MAX_ITEMS) envs[n_envs++] = argv[++i];
        else if (strcmp(a, "--rlimit") == 0 && n_rlimits < MAX_ITEMS) rlimits[n_rlimits++] = argv[++i];
        else if (strcmp(a, "--mountpoint") == 0 && n_mountpoints < MAX_ITEMS)
            mountpoints[n_mountpoints++] = argv[++i];
        else usage();
    }
    if (root == NULL || i >= argc || cpus < 1 || cpus > 255 || mem < 64) usage();
    envs[n_envs] = NULL;
    rlimits[n_rlimits] = NULL;
    if (profile) apply_profile(profile);

    int rc, ctx = krun_create_ctx();
    if (ctx < 0) die("krun_create_ctx", ctx);
    if ((rc = krun_set_vm_config(ctx, (uint8_t)cpus, (uint32_t)mem))) die("krun_set_vm_config", rc);
    if ((rc = krun_add_virtiofs3(ctx, KRUN_FS_ROOT_TAG, root, 0, !root_writable)))
        die("root filesystem", rc);
    for (int m = 0; m < n_mountpoints; m++) add_mountpoint(ctx, mountpoints[m]);
    for (int s = 0; s < n_shares; s++) {
        char *path = split_eq(shares[s]);
        if ((rc = krun_add_virtiofs(ctx, shares[s], path))) die("--share", rc);
    }
    for (int s = 0; s < n_ro_shares; s++) {
        char *path = split_eq(ro_shares[s]);
        if ((rc = krun_add_virtiofs3(ctx, ro_shares[s], path, 0, true))) die("--share-ro", rc);
    }
    // No implicit vsock: its TSI would proxy guest sockets through the host.
    if ((rc = krun_disable_implicit_vsock(ctx))) die("krun_disable_implicit_vsock", rc);
    if ((rc = krun_add_vsock(ctx, net ? KRUN_TSI_HIJACK_INET : 0))) die("krun_add_vsock", rc);
    for (int v = 0; v < n_vsocks; v++) {
        char *path = split_eq(vsocks[v]);
        if ((rc = krun_add_vsock_port(ctx, (uint32_t)atoi(vsocks[v]), path))) die("--vsock", rc);
    }
    if (n_rlimits && (rc = krun_set_rlimits(ctx, rlimits))) die("--rlimit", rc);
    if (workdir && (rc = krun_set_workdir(ctx, workdir))) die("--workdir", rc);
    if ((rc = krun_set_exec(ctx, argv[i], (const char *const *)&argv[i + 1], envs)))
        die("krun_set_exec", rc);
    rc = krun_start_enter(ctx);  // only returns on a configuration error
    die("krun_start_enter", rc);
    return 125;
}
