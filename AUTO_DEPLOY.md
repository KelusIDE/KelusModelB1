# Automatic B1 release

The local watcher is authorized to publish **only Firebase Hosting site `kelus-b1` in project `kelus-ide`**. It runs every five minutes in a detached background process while this Mac is awake. It publishes the prepared static site and exported model; it does not deploy Firestore rules or change account configuration.

**Pretraining completion is not a release.** The current 8.69M run is foundation pretraining. A varied instruction dataset and instruction training still need to be completed, followed by genuine browser verification. The watcher does not create that missing dataset, train an instruction model, or perform human/browser verification itself. Until those stages are complete, v3 stays live. It can continue after this chat closes. After a reboot or logout, restart the watcher with **Start Auto Deploy.command** and resume training with **Train Foundation.command**. Neither process automatically restarts after a reboot.

## Check or cancel

Double-click **Auto Deploy Status.command** or **Stop Auto Deploy.command** in this project. Terminal equivalents:

```sh
cd /Users/hashtagpro/Downloads/KelusModelB1
python3 -m kelus.auto_deploy --status
python3 -m kelus.auto_deploy --disable
```

State: `training-runs/auto-deploy/status.json`; process log: `training-runs/auto-deploy/watcher.log`. `--status` checks the process lock so a stale status file is not mistaken for a running watcher. Start/restart with:

```sh
python3 -m kelus.auto_deploy --start
```

Duplicate starters are harmless. Disabling stops future deployments; it cannot cancel an upload already in progress. The watcher exits on deployment success or an operational error; resolve errors and restart it explicitly. Missing model/verification evidence simply keeps it waiting. A macOS LaunchAgent was tested but could not open the Downloads project; it was unloaded and its unused plist retained in the state folder. No macOS privacy settings were changed.

## Completing the release pipeline

After finishing instruction training from this foundation run's `best.npz` or `latest.npz`, prepare the selected immutable checkpoint. This command attests that instruction training is finished; do not run it on a live checkpoint:

```sh
python3 -m kelus.auto_deploy --prepare /absolute/path/to/instruction.best.npz \
  --data /absolute/path/to/instruction.jsonl --training-complete
```

Preparation freezes the checkpoint, dataset, existing site and Firebase Hosting configuration in `training-runs/auto-deploy/releases/<checkpoint SHA256>/`. It verifies ancestry and dataset identity, checks exact readiness-prompt leakage, runs the unit suite, new-spec coding/website structure checks, original regressions and wider benchmark, then checks the actual browser export against Python. It does not execute generated code. Logs and reports stay with the release. A failed preparation does not replace the pending candidate. A failed attempt's directory must be inspected and moved aside before retrying that same checkpoint.

Required scores: new-spec Python 4/4; all website structure checks; original bug regressions 5/5, wording 12/15 or better, screenshot sequence 5/5; wider benchmark at least 158/199 and context regressions at least 179/184. These are narrow development checks, not proof of general coding ability. The canonical-answer benchmark can reject differently worded correct replies; inspect failures rather than silently lowering the gate.

Next, use a real browser to inspect the exact generated website from `readiness.json` (also saved verbatim as `website-reply.txt`). Confirm the visible counter starts at 0 and increases on two clicks. Preview the frozen B1 site bundle, and confirm it loads the new model and answers a chat request. Keep notes/screenshots as evidence. After those checks actually pass, write `website-verification.json` beside `release.json`:

```json
{
  "checkpoint_sha256": "value from release.json",
  "bundle_sha256": "value from release.json",
  "reply_sha256": "SHA256 of the UTF-8 readiness.json website.reply text, including any code fences",
  "tester": "person or agent that performed the checks",
  "notes": "Actual observations and paths to supporting screenshots/records",
  "checks": {
    "counter_starts_at_zero": true,
    "counter_increments_twice": true,
    "heading_and_button_visible": true,
    "b1_preview_loads_and_replies": true
  }
}
```

This is a local verification record, not an automated browser test or a security signature. Do not create a passing record without performing the checks. The watcher rejects evidence or reports belonging to another model, website response or bundle. **Once all prerequisites pass, publication is automatic, without another approval prompt.** This now also includes the expanded language scope below.

The watcher verifies the remote model manifest and weight hash after upload. Network/authentication failures or an interrupted deployment are recorded and never blindly retried. `deployment.json` records an attempt before uploading to prevent duplicate publication. If it reports an uncertain result, inspect the live site and `deploy.log` before clearing the receipt for an intentional retry. A verification failure after a successful upload does not automatically roll back the site.

The frozen old model export in the original `site/model/` is retained. Firebase credentials must remain valid and the installed Node/Firebase paths must remain available. The watcher uses the Python interpreter that started it and the Node/Firebase installation path in `kelus/auto_deploy.py`; update those paths if the installations move.

## Expanded language release requirement

The user subsequently requested 23 coding languages and seven chat language/script targets. Every newly prepared candidate records the hash of `config/languages.json` and receives a blank `language-verification.template.json`. Before automatic publication, complete actual output/behavior reviews and save the results as `language-verification.json`. See [LANGUAGE_TRAINING.md](LANGUAGE_TRAINING.md) for what each check requires. Missing/failed language checks and stale candidate/scope identities block release. The watcher does not perform these reviews or assemble/train the multilingual data itself.
