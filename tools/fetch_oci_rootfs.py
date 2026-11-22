#!/usr/bin/env python3
"""Stream an OCI/Docker image registry into a plain rootfs directory.

Why this exists: the official SpacemiT x64 emulation image is published as an
11.5 GB compressed image (19.4 GB unpacked).  A host that keeps both compressed
blobs and unpacked snapshots (``docker pull``) needs roughly twice the unpacked
size; this tool only keeps the unpacked rootfs and therefore fits on a 30 GB
disk.  It also allows running the image without a Docker daemon.

It handles the layer whiteout entries that ``docker load``/``umoci`` normally
apply.  Layers are streamed, so no layer tarball is written to disk.

Example:
    sudo python3 tools/fetch_oci_rootfs.py \
        --repo flagtree/flagtree-spacemit-py312-torch2.10.0-ubuntu24.04 \
        --tag 202607-3.6-base \
        --dest /var/tmp/sm-rootfs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request

DEFAULT_REGISTRY = "https://harbor.baai.ac.cn"


def free_gb(path: str) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def fetch_token(registry: str, repo: str) -> str:
    query = urllib.parse.urlencode(
        {"service": "harbor-registry", "scope": f"repository:{repo}:pull"}
    )
    with urllib.request.urlopen(f"{registry}/service/token?{query}", timeout=60) as resp:
        return json.load(resp)["token"]


def open_url(url: str, token: str, accept: str | None = None, timeout: int = 600):
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    return urllib.request.urlopen(request, timeout=timeout)


def remove_path(path: str) -> None:
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except FileNotFoundError:
        pass


def safe_target(dest: str, name: str) -> str | None:
    parts = [part for part in name.lstrip("/").split("/") if part not in ("", ".")]
    if ".." in parts:
        return None
    return os.path.join(dest, *parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.5,
        help="abort when the destination filesystem drops below this value",
    )
    args = parser.parse_args(argv)

    token = fetch_token(args.registry, args.repo)
    manifest_url = f"{args.registry}/v2/{args.repo}/manifests/{args.tag}"
    with open_url(
        manifest_url,
        token,
        accept="application/vnd.docker.distribution.manifest.v2+json",
    ) as resp:
        manifest = json.load(resp)
    layers = manifest["layers"]
    total = sum(layer["size"] for layer in layers) / 1e9
    print(f"layers={len(layers)} compressed_GB={total:.2f}", flush=True)

    os.makedirs(args.dest, exist_ok=True)
    started = time.time()
    for index, layer in enumerate(layers):
        if free_gb(args.dest) < args.min_free_gb:
            print(f"ABORT: free space below {args.min_free_gb} GB", flush=True)
            return 3
        layer_started = time.time()
        blob_url = f"{args.registry}/v2/{args.repo}/blobs/{layer['digest']}"
        with open_url(blob_url, token) as resp:
            archive = tarfile.open(fileobj=resp, mode="r|gz")
            entries = 0
            for member in archive:
                name = member.name
                base = os.path.basename(name.rstrip("/"))
                if base.startswith(".wh."):
                    parent = os.path.dirname(safe_target(args.dest, name) or "")
                    if base == ".wh..wh..opq":
                        if os.path.isdir(parent):
                            for child in os.listdir(parent):
                                remove_path(os.path.join(parent, child))
                    else:
                        remove_path(os.path.join(parent, base[len(".wh.") :]))
                    continue
                target = safe_target(args.dest, name)
                if target is None:
                    continue
                member.name = target
                link = member.linkname
                if member.issym():
                    # Symlink targets are resolved inside the rootfs, so an
                    # absolute target must stay absolute.  Prefixing it with the
                    # extraction directory would break every chroot lookup.
                    pass
                elif link and link.startswith("/"):
                    member.linkname = link.lstrip("/")
                elif link and ".." in link.split("/"):
                    member.linkname = os.path.normpath(
                        os.path.join(os.path.dirname(target), link)
                    )
                try:
                    archive.extract(
                        member,
                        args.dest,
                        set_attrs=True,
                        numeric_owner=True,
                        filter="fully_trusted",
                    )
                except (OSError, tarfile.ExtractError) as exc:
                    print(f"  skip {target}: {exc}", flush=True)
                entries += 1
                if entries % 20000 == 0 and free_gb(args.dest) < args.min_free_gb:
                    print("ABORT mid-layer: free space exhausted", flush=True)
                    return 3
        print(
            f"layer[{index}] {layer['size'] / 1e9:.2f}GB compressed, {entries} entries, "
            f"{time.time() - layer_started:.0f}s, free={free_gb(args.dest):.1f}GB",
            flush=True,
        )

    print(
        f"DONE rootfs={args.dest} elapsed={time.time() - started:.0f}s "
        f"free={free_gb(args.dest):.1f}GB",
        flush=True,
    )
    for probe in ("usr/bin/python3", "bin/bash", "etc/os-release"):
        print(f"probe {probe}: exists={os.path.exists(os.path.join(args.dest, probe))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())