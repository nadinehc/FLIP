#!/usr/bin/env python
"""
Runs FLIP against a GAN-generated trigger, compared to the paper's hand-coded
sinusoidal trigger.

This is the trigger-axis experiment. FLIP itself is label-only throughout --
the victim always receives clean images and flipped labels, through the
untouched `generate_labels` / `train_user` modules. What varies between
settings is the *trigger* the experts are backdoored with and that PTA is
measured against:

  1xs       the paper's sinusoidal stripes (StripePoisoner), the baseline
  gan_epsN  a generator trained by `train_trigger`, bounded at N/255

Pipeline per setting:

  0. (gan only) train the trigger generator: G against a discriminator and a
     surrogate classifier. Outputs a frozen generator.pth.
  1. Train expert trajectories on data poisoned with *this* trigger.
  2. Optimize labels (label-only FLIP) and select flips at each budget m.
  3. Train a victim on clean images with flipped labels; record CTA / PTA and
     run the kmeans / PCA / SPECTRE defenses.

Experts are NOT shared across settings here, unlike sweep/run_sweep.py: the
experts are trained on trigger-poisoned data, so a different trigger means
different experts. That is the dominant cost.

Every step is skipped if its outputs already exist, so an interrupted sweep
resumes by re-running the same command. Use --force to recompute.

Examples:
    python sweep/run_trigger_sweep.py --tag trig
    python sweep/run_trigger_sweep.py --tag trig --dry-run
    python sweep/run_trigger_sweep.py --tag trigsmoke --smoke
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sweep import (BASE_SETTING, DEFENSES, REPO_ROOT, Runner,  # noqa: E402
                       log)

DEFAULT_EPSILONS = [2, 4, 8]
DEFAULT_BUDGETS = [150, 300, 500, 1000, 1500]

# The baseline trigger is the paper's; BASE_SETTING carries the rest of the
# Table 1 CIFAR-10 configuration (r32p, cifar, sgd, source 9 -> target 4).
BASELINE_TRIGGER = BASE_SETTING['poisoner']


class TriggerRunner(Runner):
    """
    Sweeps over triggers rather than over attack-side perturbations.

    Inherits the plumbing (config writing, step running and skipping, logging,
    preflight) from Runner and replaces everything that assumed the generator
    lived on the attack side.
    """

    # -- settings ---------------------------------------------------------

    def settings(self):
        '''The paper's trigger, plus one learned trigger per epsilon.'''
        yield {'name': BASELINE_TRIGGER, 'learned': False, 'eps_255': None,
               'poisoner': BASELINE_TRIGGER, 'arch': None}
        for eps in self.args.epsilons:
            yield {'name': f'{self.args.arch}_eps{eps}', 'learned': True,
                   'eps_255': eps, 'poisoner': 'gan',
                   'arch': self.args.arch}

    @staticmethod
    def parse_setting(name):
        """
        Rebuilds a setting dict from an experiment directory's name.

        The name carries the architecture ("vit_eps8", "conv_eps8") so that
        sweeps run with different generators do not collide in one results
        table, and so the table says which generator produced each row.
        """
        for arch in ('vit', 'conv'):
            prefix = f'{arch}_eps'
            if name.startswith(prefix):
                try:
                    return {'name': name, 'learned': True,
                            'eps_255': int(name[len(prefix):]),
                            'poisoner': 'gan', 'arch': arch}
                except ValueError:
                    return None
        return {'name': name, 'learned': False, 'eps_255': None,
                'poisoner': name, 'arch': None}

    # -- directories ------------------------------------------------------

    def trigger_dir(self, setting):
        return (REPO_ROOT / 'experiments' /
                f"sweep_{self.tag}_trigger_{setting['name']}")

    def setting_ckpt_dir(self, setting):
        '''Experts are per-setting: a different trigger means different experts.'''
        return self.ckpt_dir / setting['name']

    def generator_path(self, setting):
        '''Repo-relative path to this setting's frozen trigger generator.'''
        if not setting['learned']:
            return None
        return (os.path.relpath(self.trigger_dir(setting), REPO_ROOT)
                + '/generator.pth')

    def expert_paths(self, setting):
        '''Format strings the label module uses to find this setting's experts.'''
        rel = os.path.relpath(self.setting_ckpt_dir(setting), REPO_ROOT)
        return (f"{rel}/{{}}/model_{{}}_{{}}.pth",
                f"{rel}/{{}}/model_{{}}_{{}}_opt.pth")

    # -- stage 0: the trigger ---------------------------------------------

    def adaptive_config(self):
        '''
        Image-adaptivity options for the trigger. Only present in the config
        when asked for, so a default run writes exactly the config it always
        did (TOML has no null, hence no `adaptive_floor = None`).
        '''
        cfg = {}
        if self.args.adaptive_floor is not None:
            cfg['adaptive_floor'] = self.args.adaptive_floor
        if self.args.lambda_div > 0:
            cfg['lambda_div'] = self.args.lambda_div
            cfg['div_tau'] = self.args.div_tau
        return cfg

    def train_trigger(self, setting):
        '''Trains the GAN trigger. A no-op for the hand-coded baseline.'''
        if not setting['learned']:
            return True

        exp_name = f"sweep_{self.tag}_trigger_{setting['name']}"
        out_dir = os.path.relpath(self.trigger_dir(setting), REPO_ROOT) + '/'

        config = {'train_trigger': {
            'output_dir': out_dir,
            'model': BASE_SETTING['model'],
            'dataset': BASE_SETTING['dataset'],
            'source_label': BASE_SETTING['source_label'],
            'target_label': BASE_SETTING['target_label'],
            'trigger_config': {
                **self.adaptive_config(),
                'epsilon_255': setting['eps_255'],
                'norm': self.args.norm,
                'arch': self.args.arch,
                'dim': self.args.vit_dim,
                'depth': self.args.vit_depth,
                'heads': self.args.vit_heads,
                'patch_size': self.args.vit_patch,
                'width': self.args.generator_width,
                'd_width': self.args.discriminator_width,
                'g_lr': self.args.generator_lr,
                'd_lr': self.args.discriminator_lr,
                'lambda_bd': self.args.lambda_bd,
                'lambda_tv': self.args.lambda_tv,
                'lambda_adv': self.args.lambda_adv,
                'epochs': self.args.trigger_epochs,
                'warmup': self.args.trigger_warmup,
                'quantize': True,
            },
        }}
        self.write_config(exp_name, config)

        expect = [self.trigger_dir(setting) / 'generator.pth']
        return self.run_experiment(exp_name, f"trigger_{setting['name']}",
                                   expect=expect)

    # -- stage 1: experts, per setting ------------------------------------

    def train_experts(self, setting):
        '''Trains this setting's experts on trigger-poisoned data.'''
        ok = True
        for i in range(self.args.experts):
            exp_name = f"sweep_{self.tag}_expert_{setting['name']}_{i}"
            out_dir = (os.path.relpath(self.setting_ckpt_dir(setting) / str(i),
                                       REPO_ROOT) + '/')

            block = {
                'output_dir': out_dir,
                'model': BASE_SETTING['model'],
                'trainer': BASE_SETTING['trainer'],
                'dataset': BASE_SETTING['dataset'],
                'source_label': BASE_SETTING['source_label'],
                'target_label': BASE_SETTING['target_label'],
                'poisoner': setting['poisoner'],
                'epochs': self.args.expert_epochs,
                'checkpoint_iters': self.args.checkpoint_iters,
            }
            if setting['learned']:
                block['generator'] = self.generator_path(setting)

            self.write_config(exp_name, {'train_expert': block})

            last = (self.setting_ckpt_dir(setting) / str(i) /
                    f"model_{self.args.expert_epochs}_"
                    f"{self.args.checkpoint_iters}.pth")
            ok &= self.run_experiment(
                exp_name, f"expert_{setting['name']}_{i}", expect=[last])
        return ok

    def check_experts(self, setting):
        '''Verifies this setting's expert checkpoints exist before stage 2.'''
        ckpt = self.setting_ckpt_dir(setting)
        missing = [i for i in range(self.args.experts)
                   if not (ckpt / str(i) /
                           f"model_{self.args.expert_epochs}_"
                           f"{self.args.checkpoint_iters}.pth").exists()]
        if not missing:
            return True
        shown = ', '.join(str(i) for i in missing[:8])
        log(f"EXPERT CHECK FAILED for {setting['name']}: "
            f"{len(missing)} of {self.args.experts} trajectories have no final "
            f"checkpoint under {ckpt} (missing: {shown}"
            f"{', ...' if len(missing) > 8 else ''})")
        log("sampling a missing trajectory is a FileNotFoundError in stage 2; "
            "pass --experts <n> matching what was trained, or let stage 1 run")
        return False

    # -- stage 2: label-only FLIP -----------------------------------------

    def generate_labels(self, setting):
        """
        Label-only FLIP, always through the untouched `generate_labels` module.

        The only thing the trigger changes here is which poisoner is built, so
        that the attack's internal poisoned-data view and its PTA evaluation
        use the same trigger the experts were backdoored with.
        """
        exp_name = f"sweep_{self.tag}_attack_{setting['name']}"
        out_dir = os.path.relpath(self.attack_dir(setting), REPO_ROOT) + '/'
        input_pths, opt_pths = self.expert_paths(setting)

        block = {
            'input_pths': input_pths,
            'opt_pths': opt_pths,
            'expert_model': BASE_SETTING['model'],
            'trainer': BASE_SETTING['trainer'],
            'dataset': BASE_SETTING['dataset'],
            'source_label': BASE_SETTING['source_label'],
            'target_label': BASE_SETTING['target_label'],
            'poisoner': setting['poisoner'],
            'output_dir': out_dir,
            'lambda': 0.0,
            # `max` is capped at the number of epochs actually checkpointed:
            # extract_experts samples a trajectory epoch in [1, max], and
            # asking for an epoch past the end of training is a
            # FileNotFoundError at the first batch.
            'expert_config': {
                'experts': self.args.experts,
                'min': 0,
                'max': min(15, self.args.expert_epochs),
            },
            'attack_config': {
                'iterations': self.args.attack_iterations,
                'one_hot_temp': 5,
                'label_kwargs': {'lr': 150, 'momentum': 0.5},
            },
        }
        if setting['learned']:
            block['generator'] = self.generator_path(setting)

        config = {'generate_labels': block}
        config['select_flips'] = {
            'budgets': self.args.budgets,
            'input_label_glob': out_dir + 'labels.npy',
            'true_labels': out_dir + 'true.npy',
            'output_dir': out_dir,
        }
        self.write_config(exp_name, config)

        expect = [self.attack_dir(setting) / f"{m}.npy"
                  for m in self.args.budgets]
        return self.run_experiment(exp_name, f"attack_{setting['name']}",
                                   expect=expect)

    # -- stage 3: victim and defenses -------------------------------------

    def train_user_and_defend(self, setting, budget):
        """
        Victim training on clean images with flipped labels, then the defenses.

        No generator reaches `run_defenses`: the victim's images are clean, so
        the defenses inspect exactly the dataset the victim was handed. The
        generator does reach `train_user`, but only so that its PTA evaluation
        stamps the right trigger on the test set.
        """
        exp_name = f"sweep_{self.tag}_user_{setting['name']}_m{budget}"
        attack_dir = os.path.relpath(self.attack_dir(setting), REPO_ROOT) + '/'
        out_dir = os.path.relpath(self.user_dir(setting, budget),
                                  REPO_ROOT) + '/'

        block = {
            'input_labels': f"{attack_dir}{budget}.npy",
            'true_labels': f"{attack_dir}true.npy",
            'user_model': BASE_SETTING['model'],
            'trainer': BASE_SETTING['trainer'],
            'dataset': BASE_SETTING['dataset'],
            'source_label': BASE_SETTING['source_label'],
            'target_label': BASE_SETTING['target_label'],
            'poisoner': setting['poisoner'],
            'output_dir': out_dir,
            'soft': False,
            'alpha': 0.0,
        }
        if self.args.user_epochs is not None:
            block['epochs'] = self.args.user_epochs
        if self.args.user_milestones is not None:
            block['scheduler_kwargs'] = {
                'milestones': self.args.user_milestones}
        if setting['learned']:
            block['generator'] = self.generator_path(setting)

        config = {'train_user': block}
        config['run_defenses'] = {
            'model_pth': out_dir + 'model.pth',
            'input_labels': f"{attack_dir}{budget}.npy",
            'true_labels': f"{attack_dir}true.npy",
            'output_dir': out_dir,
            'model': BASE_SETTING['model'],
            'dataset': BASE_SETTING['dataset'],
            'target_label': BASE_SETTING['target_label'],
            'defenses': DEFENSES,
            'expansion': self.args.expansion,
        }
        self.write_config(exp_name, config)

        expect = [self.user_dir(setting, budget) / 'caccs.npy',
                  self.user_dir(setting, budget) / 'defenses.json']
        return self.run_experiment(exp_name,
                                   f"user_{setting['name']}_m{budget}",
                                   expect=expect)

    # -- results ----------------------------------------------------------

    def collect(self):
        '''Reads every finished cell into a list of flat result rows.'''
        rows = []
        for setting, budget in self.result_cells():
            d = self.user_dir(setting, budget)
            row = {
                'trigger': setting['name'],
                'trigger_kind': setting['arch'] or 'hand-coded',
                'epsilon_255': setting['eps_255'],
                'norm': self.args.norm if setting['learned'] else '',
                'm': budget,
            }

            try:
                row['cta'] = float(np.load(d / 'caccs.npy')[-1][0])
                row['pta'] = float(np.load(d / 'paccs.npy')[-1][0])
            except (FileNotFoundError, IndexError):
                row['cta'] = row['pta'] = None

            stats = self.trigger_dir(setting) / 'trigger_stats.json'
            if setting['learned'] and stats.exists():
                with open(stats) as f:
                    row['surrogate_asr'] = json.load(f).get('surrogate_asr')
            else:
                row['surrogate_asr'] = None
            row['delta_cos'] = None
            if setting['learned'] and stats.exists():
                with open(stats) as f:
                    row['delta_cos'] = json.load(f).get('delta_cos')

            def_path = d / 'defenses.json'
            if def_path.exists():
                with open(def_path) as f:
                    defenses = json.load(f)
                row['n_flipped'] = defenses.get('n_flipped_total')
                row['n_poison_in_target'] = defenses.get('n_poison_in_target')
                for name in DEFENSES:
                    m = defenses['defenses'].get(name, {})
                    row[f'{name}_detect'] = m.get('detection_rate')
                    row[f'{name}_fpr'] = m.get('false_positive_rate')
                    row[f'{name}_auroc'] = m.get('auroc')
            else:
                row['n_flipped'] = row['n_poison_in_target'] = None
                for name in DEFENSES:
                    row[f'{name}_detect'] = None
                    row[f'{name}_fpr'] = None
                    row[f'{name}_auroc'] = None

            rows.append(row)
        return rows

    def flip_table(self):
        """
        Counts, per cell, which true class each flipped label was moved to.

        `select_flips` writes the true label everywhere except at the flipped
        indices, so an argmax disagreement identifies the flips exactly. It
        picks the best non-true class, which is usually but not necessarily the
        target -- so this table is the place the difference shows up, and it is
        why `n_flipped` and `n_poison_in_target` can disagree in results.csv.
        """
        rows = []
        for setting, budget in self.result_cells():
            labels_path = self.attack_dir(setting) / f"{budget}.npy"
            true_path = self.attack_dir(setting) / 'true.npy'
            if not (labels_path.exists() and true_path.exists()):
                continue

            poisoned = np.load(labels_path).argmax(axis=1)
            true = np.load(true_path).argmax(axis=1)
            flipped = np.where(poisoned != true)[0]

            pairs = {}
            for i in flipped:
                pairs[(int(true[i]), int(poisoned[i]))] = \
                    pairs.get((int(true[i]), int(poisoned[i])), 0) + 1

            for (src, dst), count in sorted(pairs.items(),
                                            key=lambda kv: -kv[1]):
                rows.append({
                    'trigger': setting['name'],
                    'epsilon_255': setting['eps_255'],
                    'm': budget,
                    'true_class': src,
                    'flipped_to': dst,
                    'count': count,
                    'share_of_flips': count / max(len(flipped), 1),
                    'is_target': dst == BASE_SETTING['target_label'],
                })
        return rows

    def write_flip_table(self, rows):
        if not rows:
            log("no flip tables to write (no attack outputs found)")
            return

        csv_path = self.out_root / 'flip_table.csv'
        fields = ['trigger', 'epsilon_255', 'm', 'true_class', 'flipped_to',
                  'count', 'share_of_flips', 'is_target']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        md_path = self.out_root / 'flip_table.md'
        with open(md_path, 'w') as f:
            f.write(f"# Flipped labels by class (`{self.tag}`)\n\n")
            f.write(f"Source class {BASE_SETTING['source_label']}, target "
                    f"class {BASE_SETTING['target_label']}. `select_flips` "
                    f"picks the best non-true class, so flips need not all "
                    f"land on the target.\n\n")
            f.write("| trigger | eps (/255) | m | true class | flipped to | "
                    "count | share | on target |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(
                    f"| {r['trigger']} "
                    f"| {r['epsilon_255'] if r['epsilon_255'] else '--'} "
                    f"| {r['m']} | {r['true_class']} | {r['flipped_to']} "
                    f"| {r['count']} | {100 * r['share_of_flips']:.1f}% "
                    f"| {'yes' if r['is_target'] else 'no'} |\n")

        log(f"wrote {csv_path}")
        log(f"wrote {md_path}")

    def write_results(self, rows):
        fields = ['trigger', 'trigger_kind', 'epsilon_255', 'norm', 'm',
                  'cta', 'pta', 'surrogate_asr', 'delta_cos', 'n_flipped',
                  'n_poison_in_target']
        for name in DEFENSES:
            fields += [f'{name}_detect', f'{name}_fpr', f'{name}_auroc']

        csv_path = self.out_root / 'results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        def fmt(v, pct=False, places=3):
            if v is None:
                return '--'
            return f"{100 * v:.2f}" if pct else f"{v:.{places}f}"

        md_path = self.out_root / 'results.md'
        with open(md_path, 'w') as f:
            f.write(f"# FLIP with a GAN trigger vs the paper's trigger "
                    f"(`{self.tag}`)\n\n")
            f.write(f"- Setting: {BASE_SETTING['dataset']}, "
                    f"{BASE_SETTING['model']}, source "
                    f"{BASE_SETTING['source_label']} -> target "
                    f"{BASE_SETTING['target_label']}\n")
            f.write("- The attack is label-only in every row: the victim gets "
                    "clean images and flipped labels. Only the trigger "
                    "differs.\n")
            f.write(f"- Experts: {self.args.experts} x "
                    f"{self.args.expert_epochs} epochs, retrained per trigger; "
                    f"attack iterations: {self.args.attack_iterations}\n")
            f.write(f"- GAN trigger: {self.args.norm} bound, "
                    f"{self.args.trigger_epochs} epochs, lambda_bd="
                    f"{self.args.lambda_bd}, lambda_adv="
                    f"{self.args.lambda_adv}\n")
            f.write("- `surrogate ASR` is the trigger's attack success against "
                    "the surrogate used to train it, i.e. an upper reference "
                    "for what FLIP could distil.\n")
            f.write(f"- Defenses given a {self.args.expansion}x removal budget "
                    f"over the true poison count\n")
            f.write("- CTA / PTA / detection / FPR in %, AUROC in [0, 1]\n")
            f.write("- Defense implementations are reimplementations; see "
                    "`modules/run_defenses/utils.py`\n\n")

            f.write("| trigger | eps (/255) | m | CTA | PTA | surrogate ASR | "
                    "kmeans det | kmeans AUROC | PCA det | PCA AUROC | "
                    "SPECTRE det | SPECTRE AUROC |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for row in rows:
                f.write(
                    f"| {row['trigger']} "
                    f"| {row['epsilon_255'] if row['epsilon_255'] else '--'} "
                    f"| {row['m']} "
                    f"| {fmt(row['cta'], pct=True)} "
                    f"| {fmt(row['pta'], pct=True)} "
                    f"| {fmt(row['surrogate_asr'], pct=True)} "
                    f"| {fmt(row['kmeans_detect'], pct=True)} "
                    f"| {fmt(row['kmeans_auroc'])} "
                    f"| {fmt(row['pca_detect'], pct=True)} "
                    f"| {fmt(row['pca_auroc'])} "
                    f"| {fmt(row['spectre_detect'], pct=True)} "
                    f"| {fmt(row['spectre_auroc'])} |\n"
                )

            if self.failures:
                f.write(f"\n## Failed steps ({len(self.failures)})\n\n")
                for name in self.failures:
                    f.write(f"- `{name}` -- see `logs/{name}.log`\n")

        log(f"wrote {csv_path}")
        log(f"wrote {md_path}")

    # -- cost -------------------------------------------------------------

    def estimate(self):
        """
        Rough cost in epoch-equivalents (one pass over CIFAR-10 with r32p).

        Experts are per-setting here, which is the dominant term. The trigger
        GAN trains three networks per batch, so its epochs are counted at 3x.
        """
        n_settings = 1 + len(self.args.epsilons)
        user_epochs = self.args.user_epochs or 200

        triggers = len(self.args.epsilons) * self.args.trigger_epochs * 3
        experts = n_settings * self.args.experts * self.args.expert_epochs
        attacks = n_settings * self.args.attack_iterations * 4
        users = n_settings * len(self.args.budgets) * user_epochs

        return {
            'triggers': triggers,
            'experts': experts,
            'attacks': attacks,
            'users': users,
            'total': triggers + experts + attacks + users,
            'n_user_models': n_settings * len(self.args.budgets),
        }

    def log_estimate(self):
        e = self.estimate()
        log(f"cost estimate: {e['total']} epoch-equivalents "
            f"(~{e['triggers']} trigger GAN + {e['experts']} experts + "
            f"~{e['attacks']} attack + {e['users']} user, over "
            f"{e['n_user_models']} user models)")
        spe = self.args.seconds_per_epoch
        if spe:
            log(f"at {spe:.1f} s/epoch that is ~{e['total'] * spe / 3600:.1f} h "
                f"wall clock")
        else:
            log("pass --seconds-per-epoch for a wall-clock figure")

    # -- driver -----------------------------------------------------------

    def report_trigger(self, setting):
        '''Prints the numbers that say whether a trigger is worth a sweep.'''
        if not setting['learned']:
            return
        stats = self.trigger_dir(setting) / 'trigger_stats.json'
        if not stats.exists():
            return
        with open(stats) as f:
            s = json.load(f)
        log(f"  {setting['name']}: surrogate ASR "
            f"{100 * s.get('surrogate_asr', float('nan')):.1f}%   "
            f"cross-image cosine {s.get('delta_cos', float('nan')):.3f} "
            f"(1 = same pattern on every image)")

    def run(self):
        start = time.time()
        log(f"sweep tag={self.tag}  triggers=[{BASELINE_TRIGGER}] + "
            f"gan at {self.args.epsilons} (/255)  budgets={self.args.budgets}")
        log(f"outputs under {self.out_root}")
        log("attack is label-only in every setting (vanilla generate_labels / "
            "train_user)")

        if not self.args.dry_run and not self.args.skip_preflight:
            if not self.preflight():
                log("aborting before any step ran")
                return 2

        self.log_estimate()

        if self.args.collect_only:
            self.write_results(self.collect())
            self.write_flip_table(self.flip_table())
            return 0

        for setting in self.settings():
            name = setting['name']
            log(f"=== setting {name} ===")

            if not self.train_trigger(setting):
                log(f"skipping {name}: trigger training failed")
                continue

            if self.args.trigger_only:
                self.report_trigger(setting)
                continue

            if not self.args.skip_experts:
                self.train_experts(setting)
            else:
                log(f"stage 1 skipped for {name} (--skip-experts)")

            if not self.args.dry_run and not self.check_experts(setting):
                log(f"skipping {name}: experts incomplete")
                continue

            if not self.generate_labels(setting):
                log(f"skipping {name}: label optimization failed")
                continue

            for budget in self.args.budgets:
                self.train_user_and_defend(setting, budget)

        if self.args.dry_run:
            log("dry run complete; no results to collect")
            return 0

        if self.args.trigger_only:
            log("trigger-only run complete; no experts, attack or victim "
                "were run")
            return 0

        log("collecting results")
        self.write_results(self.collect())
        self.write_flip_table(self.flip_table())

        elapsed = timedelta(seconds=int(time.time() - start))
        if self.failures:
            log(f"sweep finished in {elapsed} with "
                f"{len(self.failures)} failed step(s): "
                f"{', '.join(self.failures)}")
            return 1
        log(f"sweep finished in {elapsed}; all steps succeeded")
        return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FLIP with a GAN-generated trigger vs the paper's trigger.")
    p.add_argument('--tag', default='trig',
                   help="Names the output directory and the generated configs.")
    p.add_argument('--python', default=sys.executable,
                   help="Interpreter used for each step.")
    p.add_argument('--epsilons', type=int, nargs='+', default=DEFAULT_EPSILONS,
                   help="GAN trigger bounds, in /255 units.")
    p.add_argument('--budgets', type=int, nargs='+', default=DEFAULT_BUDGETS,
                   help="Numbers of flipped labels (m).")
    p.add_argument('--norm', choices=['linf', 'l2'], default='linf',
                   help="Norm the trigger bound is measured in.")

    p.add_argument('--arch', choices=['vit', 'conv'], default='vit',
                   help="Trigger generator architecture.")
    p.add_argument('--vit-dim', type=int, default=192)
    p.add_argument('--vit-depth', type=int, default=4)
    p.add_argument('--vit-heads', type=int, default=3)
    p.add_argument('--vit-patch', type=int, default=4)
    p.add_argument('--trigger-epochs', type=int, default=10,
                   help="Epochs of trigger training.")
    p.add_argument('--trigger-warmup', type=int, default=1,
                   help="Surrogate-only epochs before G starts training.")
    p.add_argument('--generator-width', type=int, default=32,
                   help="Base channel count, --arch conv only.")
    p.add_argument('--discriminator-width', type=int, default=32)
    p.add_argument('--generator-lr', type=float, default=1e-3)
    p.add_argument('--discriminator-lr', type=float, default=2e-4)
    p.add_argument('--lambda-bd', type=float, default=1.0,
                   help="Weight on backdoor efficacy in G's loss.")
    p.add_argument('--lambda-tv', type=float, default=0.01,
                   help="Weight on the total-variation (smoothness) penalty, "
                        "which is what keeps the trigger from being visible "
                        "pixel-scale speckle.")
    p.add_argument('--lambda-adv', type=float, default=0.0,
                   help="Weight on fooling a discriminator. 0 drops D "
                        "entirely, which is the default now that the ViT "
                        "replaces the GAN.")

    p.add_argument('--adaptive-floor', type=float, default=None,
                   help="Make the trigger image-adaptive: scale the bound per "
                        "pixel by a texture map, so flat regions get only "
                        "this share (0-1) of epsilon. Unset = off, which "
                        "reproduces the earlier runs.")
    p.add_argument('--lambda-div', type=float, default=0.0,
                   help="Weight on the cross-image similarity penalty, which "
                        "stops the generator emitting one pattern for every "
                        "image. 0 = off.")
    p.add_argument('--div-tau', type=float, default=0.3,
                   help="Cosine similarity above which --lambda-div bites.")
    p.add_argument('--trigger-only', action='store_true',
                   help="Train and report the triggers, then stop. A few "
                        "minutes, and tells you whether a configuration "
                        "learned a working trigger before you spend hours on "
                        "experts.")

    p.add_argument('--experts', type=int, default=50,
                   help="Expert trajectories PER TRIGGER.")
    p.add_argument('--expert-epochs', type=int, default=20)
    p.add_argument('--checkpoint-iters', type=int, default=50)
    p.add_argument('--attack-iterations', type=int, default=15)
    p.add_argument('--user-epochs', type=int, default=None)
    p.add_argument('--user-milestones', type=int, nargs='+', default=None)
    p.add_argument('--expansion', type=float, default=1.5)
    p.add_argument('--seconds-per-epoch', type=float, default=None)

    p.add_argument('--skip-experts', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-preflight', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help="Tiny end-to-end pipeline check.")
    p.add_argument('--seed', type=int, default=None,
                   help="Seeds python, numpy and torch in every step (trigger, "
                        "experts, attack, victim). Use a distinct --tag per "
                        "seed, e.g. --tag adapt_s1, or steps will be skipped "
                        "as already done. Omit for the previous unseeded "
                        "behaviour.")

    args = p.parse_args(argv)

    if args.seed is not None:
        # Child processes (run_experiment.py) inherit this and seed themselves.
        os.environ['FLIP_SEED'] = str(args.seed)

    if args.smoke:
        args.experts = 1
        args.expert_epochs = 2
        args.attack_iterations = 1
        args.user_epochs = 2
        args.user_milestones = [1]
        args.budgets = [150]
        args.epsilons = [8]
        args.trigger_epochs = 2
        args.trigger_warmup = 1

    return args


def main(argv=None):
    return TriggerRunner(parse_args(argv)).run()


if __name__ == '__main__':
    sys.exit(main())
