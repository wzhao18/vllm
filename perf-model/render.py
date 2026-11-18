"""Inline out/*.json into sol_template.html -> k31vr/sol.html (self-contained, no network)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def _dst():
    """k31vr/sol.html. solmodel/ may be a symlink (k31vr/solmodel -> perf-model), so resolve via the repo layout."""
    up = os.path.dirname(HERE)
    if os.path.basename(up) == "k31vr":
        return os.path.join(up, "sol.html")
    if os.path.isdir(os.path.join(up, "k31vr")):
        return os.path.join(up, "k31vr", "sol.html")
    return os.path.join(up, "sol.html")


DST = os.environ.get("SOL_HTML") or _dst()


def main():
    data = {}
    for name in ("meta", "prefill", "decode", "optables"):
        with open(os.path.join(HERE, "out", name + ".json")) as f:
            data[name] = json.load(f)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "sol_template.html")) as f:
        tpl = f.read()
    assert "/*__DATA__*/" in tpl
    html = tpl.replace("/*__DATA__*/", "var DATA = " + blob + ";", 1)
    with open(DST, "w") as f:
        f.write(html)
    print("wrote %s (%.1f MB)" % (DST, os.path.getsize(DST) / 1e6))


if __name__ == "__main__":
    main()
