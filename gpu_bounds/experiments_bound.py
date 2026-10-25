import hashlib, json


def grid(phase):
    jobs = []

    def add(group, **kw):
        s = dict(
            group=group,
            family="Const",
            g=48,
            blocksize=32,
            rhs=4,
            rhs_exp=0,
            corr_exp=None,
            seed=301,
            global_fmt="fp32",
            algorithm="qr",
            t=4,
            p=2,
            maxit=200,
            minit=201,
            patience=201,
            tol=1e-12,
            sample_every=10,
        )
        s.update(kw)
        jobs.append(s)

    fmts = ["fp64", "fp32", "fp16", "bf16"]
    if phase == "smoke":
        for f in fmts:
            add("smoke", g=4, rhs=2, global_fmt=f, maxit=6, minit=7)
    elif phase == "pilot":
        for f in fmts:
            add("pilot", g=16, global_fmt=f, maxit=40, minit=41)
    elif phase == "large":
        for f in fmts:
            add("large", g=96, global_fmt=f)
    elif phase == "main":
        for seed in [301, 302, 303]:
            for g in [24, 48]:
                for fam in ["Const", "Dis", "Ani"]:
                    for f in fmts:
                        add("A", seed=seed, g=g, family=fam, global_fmt=f)
            for rhs, corr in [(4, None), (4, -24), (8, None)]:
                for f in ["fp32", "fp16"]:
                    add("B", seed=seed, rhs=rhs, corr_exp=corr, global_fmt=f)
            for scale in [0, -20]:
                for f in fmts:
                    add("C", seed=seed, rhs_exp=scale, global_fmt=f)
    else:
        raise ValueError(phase)
    out = {}
    for s in jobs:
        group = s.pop("group")
        ident = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
        if ident in out:
            out[ident]["groups"].append(group)
        else:
            out[ident] = dict(s, id=ident, groups=[group])
    return list(out.values())
