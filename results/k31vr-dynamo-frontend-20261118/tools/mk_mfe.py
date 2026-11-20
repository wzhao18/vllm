"""mk_mfe.py <src render dir> <dst render dir> <n_frontends>: copy a render, switch the Dynamo frontend to nginx + N frontends
with consistent-hash session affinity on X-Dynamo-Session-ID. Every other key unchanged (asserted by a parsed-YAML diff)."""
import copy, json, os, shutil, sys, yaml
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
NGINX = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/images/nginx_1.27.4-aarch64.sqsh"
assert not os.path.exists(dst), dst
shutil.copytree(src, dst, symlinks=True)
sn, dn = os.path.basename(src.rstrip("/")), os.path.basename(dst.rstrip("/"))
cfgp = os.path.join(dst, "render/config.yaml")
old = yaml.safe_load(open(cfgp))
txt = open(cfgp).read()
anchor = "  enable_multiple_frontends: false\n"
assert txt.count(anchor) == 1
txt = txt.replace(anchor, "  enable_multiple_frontends: true\n"
                  f"  num_additional_frontends: {n - 1}\n"
                  f"  nginx_container: {NGINX}\n"
                  "  nginx_session_affinity: true\n"
                  "  nginx_session_affinity_header: X-Dynamo-Session-ID\n")
txt = txt.replace(sn, dn)
open(cfgp, "w").write(txt)
new = yaml.safe_load(txt)
o2 = json.loads(json.dumps(old).replace(sn, dn))
fe = new["frontend"]
for k in ("enable_multiple_frontends", "num_additional_frontends", "nginx_container", "nginx_session_affinity", "nginx_session_affinity_header"):
    o2["frontend"].pop(k, None); fe_v = fe.pop(k)
assert o2 == new, "unexpected diff"
for f in ("render/render.json", "render/sbatch_script.sh", "hybrid.yaml"):
    p = os.path.join(dst, f)
    if os.path.exists(p):
        t = open(p).read(); open(p, "w").write(t.replace(sn, dn))
open(os.path.join(dst, "MFE.txt"), "w").write(
    f"{dn} = {sn} + frontend.enable_multiple_frontends true, {n} Dynamo frontends (num_additional_frontends {n-1}) on the\n"
    f"non-head nodes behind nginx on the head node (nginx 1.27.4 aarch64 sqsh {NGINX}),\n"
    "nginx_session_affinity true (consistent hash of X-Dynamo-Session-ID: a conversation stays on one frontend, so its\n"
    "K3 token cache and router session affinity are preserved). Every other key unchanged (parsed-YAML diff asserted by mk_mfe.py).\n"
    "Purpose: C512 frontend-bottleneck validation (single Python vLLM chat-processor frontend saturated at ~13 req/s).\n")
print("ok", dst)
