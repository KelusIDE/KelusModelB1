"""Uploads the Kani API backend as a Hugging Face Docker Space and sets its secret.

    read -s 'HF?Hugging Face write token: ' && print -r -- $HF > ~/.config/kani/hf-token && chmod 600 ~/.config/kani/hf-token
    .venv/bin/python api/deploy_space.py        # prints the Space URL for BACKEND_URL in api/gateway/wrangler.jsonc

The token and the backend secret are read from ~/.config/kani/ and never printed.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path.home() / '.config/kani'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='kani-api', help='the Space name under your Hugging Face account')
    args = parser.parse_args()
    api = HfApi(token=(CONFIG / 'hf-token').read_text().strip())
    repo = f"{api.whoami()['name']}/{args.name}"
    api.create_repo(repo, repo_type='space', space_sdk='docker', exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(['node', str(ROOT / 'api/make_space.mjs'), folder], check=True)
        api.upload_folder(repo_id=repo, repo_type='space', folder_path=folder, commit_message='Kani API backend')
    api.add_space_secret(repo, 'KANI_BACKEND_SECRET', (CONFIG / 'backend-secret').read_text().strip())
    info = api.space_info(repo)
    host = getattr(info, 'host', None) or f"https://{repo.lower().replace('/', '-').replace('_', '-').replace('.', '-')}.hf.space"
    print(f'Space: https://huggingface.co/spaces/{repo}\nBACKEND_URL: {host}')


if __name__ == '__main__':
    main()
