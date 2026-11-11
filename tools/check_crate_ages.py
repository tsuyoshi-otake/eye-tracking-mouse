#!/usr/bin/env python3
"""Fail if any crate version locked in Cargo.lock was published too recently.

Policy: a dependency version is adopted only once it has been public for at least
MIN_AGE_DAYS days, so a freshly published (possibly compromised) release never
reaches a build. Uses only the standard library; reads publish dates from the
crates.io API.

Usage: python -I tools/check_crate_ages.py
"""
import datetime
import json
import pathlib
import re
import sys
import time
import urllib.request

MIN_AGE_DAYS = 3
ROOT = pathlib.Path(__file__).resolve().parent.parent


def locked_packages(lock_text):
    """Yield (name, version) for every registry package in a Cargo.lock."""
    for block in lock_text.split("[[package]]")[1:]:
        name = re.search(r'^name = "(.+)"$', block, re.M)
        version = re.search(r'^version = "(.+)"$', block, re.M)
        source = re.search(r'^source = "(.+)"$', block, re.M)
        if name and version and source and source.group(1).startswith("registry+"):
            yield name.group(1), version.group(1)


def published_at(name, version):
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{name}/{version}",
        headers={"User-Agent": "crate-age-check"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        created = json.load(response)["version"]["created_at"]
    return datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    limit = datetime.timedelta(days=MIN_AGE_DAYS)
    lock_text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    too_new = 0
    checked = 0
    for name, version in sorted(locked_packages(lock_text)):
        age = now - published_at(name, version)
        ok = age >= limit
        too_new += not ok
        checked += 1
        print(f"{'OK ' if ok else 'NEW'} {name} {version}: published {age.days} days ago")
        time.sleep(1.0)  # crates.io asks automated clients for at most 1 request/second
    if checked == 0:
        print("FAIL: no registry packages found in Cargo.lock")
        return 1
    if too_new:
        print(f"FAIL: {too_new} of {checked} locked crates are younger than {MIN_AGE_DAYS} days")
        return 1
    print(f"PASS: all {checked} locked crates are at least {MIN_AGE_DAYS} days old")
    return 0


if __name__ == "__main__":
    sys.exit(main())
