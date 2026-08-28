"""Phase 4: does the Knowledge Base actually help?

The plan says not to assume it does. This runs the same search loop twice under the
same iteration budget -- once proposing blind, once proposing from
knowledge_base.yaml -- and compares score reached, iterations to beat the baseline,
convergence, and wasted trials.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
  The KB was derived from this dataset, so it is guaranteed to win on score. That
  is circular and is NOT the claim. The defensible claim is the transfer cost:
  how many iterations and how much wall-clock a blind search needs to reach what
  the KB hands over on iteration 1. That number is meaningful because the blind
  arm searches a space nobody tuned against the answer.

THE BLIND SEARCH SPACE IS PRE-REGISTERED, NOT INVENTED HERE
  It is exactly what KNOWLEDGE_BASE_PLAN.md Phase 1a/1d proposed before any
  experiment ran: k in 8..128, lr around the baseline's 0.001, L2 sweep, the four
  losses, affinity features on/off. Using the plan's own prior is what keeps this
  from being a straw man of my own construction.

COST ACCOUNTING
  Identical (config, seed) pairs are memoized against experiments.jsonl and a local
  cache, because the same config recurs across arms and restarts. A cache hit still
  consumes an iteration and still contributes its REAL measured runtime to the
  wall-clock total, so the reported cost is what the search would actually have
  paid.

Run:  python experiments/p4_kb_ablation.py --iterations 15 --restarts 3
"""
import os, sys, json, time, argparse, hashlib, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import yaml
from explib import dataset as D, features as F, fm, harness as H, history as HI
sys.path.insert(0, os.path.dirname(__file__))
from sweep import build_columns, AFFINITY_SPECS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KB_PATH = os.path.join(ROOT, 'knowledge_base.yaml')
CACHE_PATH = os.path.join(ROOT, 'cache', 'p4_runs.json')

BASELINE = H.CALIBRATION['official_baseline']['valid']      # 0.6016
POPULARITY = H.CALIBRATION['item_popularity']['valid']           # 0.5807
EPS, NCONV = 0.002, 3                                       # official convergence rule

# ---------------------------------------------------------------- search spaces
# Pre-registered from KNOWLEDGE_BASE_PLAN.md, before any result was known.
BLIND_SPACE = {
    'loss':     ['pointwise', 'bpr', 'listwise', 'hybrid'],
    'k':        [8, 16, 32, 64, 128],
    'lr':       [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'l2':       [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    'affinity': [False, True],
}


def kb_space(kb):
    """Derive the KB arm's proposal space from the KB itself, not from hindsight."""
    dead = ' '.join(str(d.get('what', '')) for d in kb['dead_ends']).lower()
    losses = [l for l in ['pointwise', 'bpr', 'listwise', 'hybrid'] if l not in dead]
    kmax = kb['validated_search_space']['k']['validated_range'][1]
    ks = [k for k in [1, 2, 4, 6, 8, 16] if k <= kmax]
    ks = [k for k in ks if k != 1]                 # KB: do_not_use_k1 (seed variance)
    lo, hi = kb['validated_search_space']['l2']['validated_range']
    l2s = [x for x in [1e-7, 1e-6, 1e-5] if lo <= x <= hi]
    lrs = {}
    for name in losses:
        entry = kb['validated_search_space']['learning_rate'].get(name)
        if entry and entry.get('validated_range'):
            a, b = entry['validated_range']
            lrs[name] = sorted({x for x in [0.0001, 0.0002, 0.0005, 0.001] if a <= x <= b})
        else:
            lrs[name] = [0.001]
    return {'loss': losses, 'k': ks, 'l2': l2s, 'lr_by_loss': lrs,
            'affinity': [False]}          # KB marks every affinity variant skip/dead


# ------------------------------------------------------------------- execution
class Runner:
    """Trains a config, memoizing identical (config, seed) pairs."""

    def __init__(self, enc_plain, enc_aff, dim_plain, dim_aff):
        self.enc = {False: enc_plain, True: enc_aff}
        self.dim = {False: dim_plain, True: dim_aff}
        self.cache = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, encoding='utf-8') as fh:
                    self.cache = json.load(fh)
            except json.JSONDecodeError:
                # a half-written cache is a cache miss, not a fatal error
                print('warning: cache unreadable, rebuilding from the log')
        self._seed_from_log()
        self.fresh = 0
        self.hits = 0

    @staticmethod
    def key(cfg, seed):
        c = {k: cfg[k] for k in ('loss', 'k', 'lr', 'l2', 'affinity')}
        c['seed'] = seed
        return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]

    def _seed_from_log(self):
        """Reuse Phase 1 runs where the config matches exactly -- same math, same result."""
        for r in H.read_log():
            c = r.get('config') or {}
            if c.get('model') != 'fm' or 'loss' not in c:
                continue
            if c.get('fields') != list(F.BASELINE_FIELDS) or c.get('affinity'):
                continue
            if not r['metrics'].get('valid'):
                continue
            cfg = {'loss': c['loss'], 'k': c['k'], 'lr': c['lr'],
                   'l2': c['l2'], 'affinity': False}
            k = self.key(cfg, c.get('seed', 0))
            self.cache.setdefault(k, {'valid': float(r['metrics']['valid']['primary']),
                                      'test': float(r['metrics']['test']['primary']),
                                      'seconds': r['seconds'], 'from': r['exp_id']})

    def save(self):
        # write-then-rename: a crash mid-write must not corrupt the cache
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self.cache, fh)
        os.replace(tmp, CACHE_PATH)

    def run(self, cfg, seed):
        k = self.key(cfg, seed)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        aff = cfg['affinity']
        epochs, patience = (60, 6) if cfg['loss'] in ('bpr', 'listwise') else (40, 4)
        t0 = time.time()
        m, info = fm.train(self.enc[aff], self.dim[aff], loss=cfg['loss'], k=cfg['k'],
                           lr=cfg['lr'], l2=cfg['l2'], epochs=epochs, patience=patience,
                           seed=seed, evaluator=H.score, verbose=False)
        X, y, u = self.enc[aff]['valid']
        va = H.score(u, y, m.predict(X))
        Xt, yt, ut = self.enc[aff]['test']
        te = H.score(ut, yt, m.predict(Xt))
        # H.score returns numpy scalars; keep the cache JSON-serializable
        rec = {'valid': float(va['primary']), 'test': float(te['primary']),
               'seconds': round(time.time() - t0, 1), 'from': 'fresh'}
        self.cache[k] = rec
        self.fresh += 1
        return rec


# ------------------------------------------------------------------- proposers
class BlindProposer:
    """Uniform over the plan's pre-registered space. No memory beyond avoiding repeats."""
    name = 'blind'

    def __init__(self, rng):
        self.rng = rng
        self.seen = set()

    def propose(self, history):
        for _ in range(200):
            cfg = {k: self.rng.choice(v) for k, v in BLIND_SPACE.items()}
            cfg['k'] = int(cfg['k'])
            cfg['affinity'] = bool(cfg['affinity'])
            key = json.dumps(cfg, sort_keys=True)
            if key not in self.seen:
                self.seen.add(key)
                return cfg, 0
        return cfg, 0


class KBProposer:
    """Reads the KB: starts at the recommended config, filters dead ends, applies
    the replication rule and the diagnostic stop condition."""
    name = 'kb'

    def __init__(self, rng, kb):
        self.rng = rng
        self.kb = kb
        self.space = kb_space(kb)
        self.seen = set()
        rec = next(m for m in kb['candidate_models'] if m.get('recommended'))
        self.start = rec['recommended_config']
        self.started = False
        self.replicated = set()

    def propose(self, history):
        # 1. architecture_ladder: open on the recommended operating point
        if not self.started:
            self.started = True
            c = self.start
            cfg = {'loss': c['loss'], 'k': c['k'], 'lr': c['lr'],
                   'l2': c['l2'], 'affinity': False}
            self.seen.add(json.dumps(cfg, sort_keys=True))
            return cfg, 0

        # 2. replication_rule: a config within 0.0015 of best gets more seeds
        if history:
            best = max(h['valid'] for h in history)
            for h in history:
                key = json.dumps(h['cfg'], sort_keys=True)
                if h['valid'] >= best - 0.0015 and (key, h['seed'] + 1) not in self.replicated:
                    self.replicated.add((key, h['seed'] + 1))
                    if h['seed'] + 1 < 3:
                        return dict(h['cfg']), h['seed'] + 1

        # 3. otherwise sample inside the validated space (dead ends already excluded)
        for _ in range(200):
            loss = self.rng.choice(self.space['loss'])
            cfg = {'loss': loss,
                   'k': int(self.rng.choice(self.space['k'])),
                   'lr': float(self.rng.choice(self.space['lr_by_loss'][loss])),
                   'l2': float(self.rng.choice(self.space['l2'])),
                   'affinity': False}
            key = json.dumps(cfg, sort_keys=True)
            if key not in self.seen:
                self.seen.add(key)
                return cfg, 0
        return cfg, 0


# ----------------------------------------------------------------------- loop
def search(proposer, runner, iterations):
    history, best_curve = [], []
    best, stall, conv_iter = -1.0, 0, None
    for it in range(1, iterations + 1):
        cfg, seed = proposer.propose(history)
        rec = runner.run(cfg, seed)
        history.append({'iter': it, 'cfg': cfg, 'seed': seed, **rec})
        if rec['valid'] > best + EPS:
            best, stall = max(best, rec['valid']), 0
        else:
            best = max(best, rec['valid'])
            stall += 1
            if stall >= NCONV and conv_iter is None:
                conv_iter = it
        best_curve.append(best)
    beat = next((h['iter'] for h in history if h['valid'] > BASELINE), None)
    return {
        'best_valid': round(best, 5),
        'best_test': round(max(h['test'] for h in history), 5),
        'iters_to_beat_baseline': beat,
        'iters_to_converge': conv_iter,
        'below_baseline_trials': sum(1 for h in history if h['valid'] <= BASELINE),
        'broken_trials': sum(1 for h in history if h['valid'] < POPULARITY),
        'wall_clock_s': round(sum(h['seconds'] for h in history), 1),
        'best_curve': [round(x, 5) for x in best_curve],
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=15)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(ROOT, 'phase4_ablation.json'))
    a = ap.parse_args()

    kb = yaml.safe_load(open(KB_PATH, encoding='utf-8'))
    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc_plain, dim_plain, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)

    # the blind arm may propose affinity features, so that encoding must exist
    cols = build_columns(logs, masks, {'author_id', 'tag', 'dur_bucket'})
    specs = [(f'aff_{n}', [cols[c] if c in cols else logs[c] for c in AFFINITY_SPECS[n]])
             for n in ('user_tab', 'user_dur', 'video')]
    extra, _, _ = HI.build_affinity_fields(logs, masks, specs, mode='causal')
    enc_aff, dim_aff, _ = F.encode_splits(logs, masks,
                                          list(F.BASELINE_FIELDS) + list(extra),
                                          extra_cols=extra)

    runner = Runner(enc_plain, enc_aff, dim_plain, dim_aff)
    print(f'seeded cache with {len(runner.cache)} runs from experiments.jsonl\n')

    results = {'blind': [], 'kb': []}
    for r in range(a.restarts):
        for arm in ('blind', 'kb'):
            rng = random.Random(1000 + r)
            prop = BlindProposer(rng) if arm == 'blind' else KBProposer(rng, kb)
            t0 = time.time()
            out = search(prop, runner, a.iterations)
            out['restart'] = r
            results[arm].append(out)
            print(f"[{arm:5s} restart {r}] best {out['best_valid']:.4f} | "
                  f"beat baseline @ iter {out['iters_to_beat_baseline']} | "
                  f"converged @ {out['iters_to_converge']} | "
                  f"{out['below_baseline_trials']}/{a.iterations} below baseline | "
                  f"{out['broken_trials']} broken | {out['wall_clock_s']:.0f}s search cost "
                  f"({time.time()-t0:.0f}s real)")
            runner.save()

    summary = {}
    for arm, runs in results.items():
        f = lambda key: [x[key] for x in runs if x[key] is not None]
        summary[arm] = {
            'best_valid_mean': round(float(np.mean([x['best_valid'] for x in runs])), 5),
            'best_valid_min': round(min(x['best_valid'] for x in runs), 5),
            'best_test_mean': round(float(np.mean([x['best_test'] for x in runs])), 5),
            'iters_to_beat_baseline': f('iters_to_beat_baseline'),
            'never_beat_baseline': sum(1 for x in runs if x['iters_to_beat_baseline'] is None),
            'below_baseline_trials_mean': round(float(np.mean(
                [x['below_baseline_trials'] for x in runs])), 2),
            'broken_trials_mean': round(float(np.mean([x['broken_trials'] for x in runs])), 2),
            'wall_clock_s_mean': round(float(np.mean([x['wall_clock_s'] for x in runs])), 1),
        }

    print('\n' + '=' * 78)
    print(f'PHASE 4 — KB ABLATION  ({a.iterations} iterations x {a.restarts} restarts)')
    print('=' * 78)
    hdr = f"{'metric':34s} {'blind':>16} {'with KB':>16}"
    print(hdr); print('-' * len(hdr))
    rows = [
        ('best valid reached (mean)', 'best_valid_mean', '{:.4f}'),
        ('best valid reached (worst restart)', 'best_valid_min', '{:.4f}'),
        ('best test reached (mean)', 'best_test_mean', '{:.4f}'),
        ('restarts never beating baseline', 'never_beat_baseline', '{}'),
        ('trials at/below baseline (mean)', 'below_baseline_trials_mean', '{}'),
        ('broken trials, below popularity', 'broken_trials_mean', '{}'),
        ('search wall-clock, seconds (mean)', 'wall_clock_s_mean', '{:.0f}'),
    ]
    for label, key, fmtstr in rows:
        print(f"{label:34s} {fmtstr.format(summary['blind'][key]):>16} "
              f"{fmtstr.format(summary['kb'][key]):>16}")
    print(f"{'iterations to beat baseline':34s} "
          f"{str(summary['blind']['iters_to_beat_baseline']):>16} "
          f"{str(summary['kb']['iters_to_beat_baseline']):>16}")
    print(f"\ncache: {runner.hits} hits, {runner.fresh} fresh runs")

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump({'config': vars(a), 'summary': summary, 'runs': results}, fh, indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
