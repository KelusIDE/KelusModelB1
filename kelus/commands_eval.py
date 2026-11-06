"""Does Cide write the right command for macOS, Linux and PowerShell? 72 held-out requests, 24 per system.

    .venv/bin/python -m kelus.commands_eval --checkpoint checkpoints/cide-4-6.npz --out reports/commands.json
    .venv/bin/python -m kelus.commands_eval --check-references   # the test against its own right answers, no model

Nothing a model writes is ever run. A reply passes if it has a code block labelled for that system, the block holds the
command the request needs, it holds no command from another system, and, for macOS and Linux, `bash -n` can parse it
(-n reads without executing). There is no PowerShell on this Mac, so PowerShell code is checked by pattern only. Passing
shows the right command was chosen, not that its flags and arguments are all correct; replies are kept for reading.

"known" requests ask, in new words, for a task data/make_commands.py teaches in its own table. "new" tasks are left out of
that table on purpose, so Cide can only have met them in tldr-pages or the PowerShell docs. No request here may be trained
on: make_commands.py drops any row with the same words.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r'```[ \t]*([A-Za-z0-9+#_-]*)[^\n]*\n(.*?)(?:```|$)', re.S)
LABELS = {'mac': {'zsh', 'bash', 'sh', 'shell', 'console', 'terminal'}, 'linux': {'bash', 'sh', 'shell', 'zsh', 'console', 'terminal'},
          'powershell': {'powershell', 'pwsh', 'ps1', 'ps'}}
CMDLET = r'\b(?:Get|Set|New|Remove|Start|Stop|Invoke|Select|Sort|Where|Test|Copy|Move|Add)-[A-Z][a-z]\w+'
WRONG = {'mac': rf'\b(?:apt|apt-get|dnf|yum|pacman|systemctl|journalctl|xclip|xsel|xdg-open|sha256sum|ufw)\b|{CMDLET}',
         'linux': rf'\b(?:brew|pbcopy|pbpaste|launchctl|diskutil|dscacheutil|sw_vers|mdfind|pmset|screencapture|softwareupdate|hdiutil)\b|{CMDLET}',
         'powershell': r'(?:^|[|;&]\s*)(?:sudo|apt|apt-get|brew|systemctl|chmod|chown|grep|awk|sed)\b'}

# system, group, request, what the code must match, a right answer (--check-references proves the patterns accept it)
TESTS = [
    ('mac', 'known', "I'm on a MacBook. What do I type to see how much battery is left?", r'pmset\s+-g\s+(batt|ps)', 'pmset -g batt'),
    ('mac', 'known', 'macbook terminal: copy everything in todo.txt so I can paste it somewhere', r'\bpbcopy\b', 'pbcopy < todo.txt'),
    ('mac', 'known', 'what do i type on macos to grab a screenshot and save it as shot.png', r'screencapture\b.*shot\.png', 'screencapture shot.png'),
    ('mac', 'known', 'mac: which program is listening on port 8080?', r'lsof\b.*8080', 'lsof -nP -iTCP:8080 -sTCP:LISTEN'),
    ('mac', 'known', 'I need ffmpeg on my Mac. How do I get it from the command line?', r'brew\s+install\s+ffmpeg', 'brew install ffmpeg'),
    ('mac', 'known', 'swap every foo for bar inside config.txt from the Mac terminal, no backup file', r"sed\s+-i\s+(''|\"\")\s+.*foo.*bar", "sed -i '' 's/foo/bar/g' config.txt"),
    ('mac', 'known', "What's the way to get the SHA-256 of a download on macOS?", r'shasum\s+-a\s+256|openssl\s+(dgst\s+-)?sha256', 'shasum -a 256 download.dmg'),
    ('mac', 'known', 'macOS: keep my laptop awake while a long job runs', r'\bcaffeinate\b', 'caffeinate -d'),
    ('mac', 'known', 'clear out the DNS cache on my mac please', r'dscacheutil|mDNSResponder', 'sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder'),
    ('mac', 'known', 'show me which version of macOS this machine runs, from the terminal', r'\bsw_vers\b', 'sw_vers'),
    ('mac', 'known', "what's my local IP address? I'm on a Mac on wifi", r'ipconfig\s+getifaddr|\bifconfig\b', 'ipconfig getifaddr en0'),
    ('mac', 'known', 'macOS terminal, I want to see all the disks attached', r'diskutil\s+list', 'diskutil list'),
    ('mac', 'new', 'On macOS, make Finder show hidden files for good', r'defaults\s+write\s+com\.apple\.finder\s+AppleShowAllFiles', 'defaults write com.apple.finder AppleShowAllFiles -bool true'),
    ('mac', 'new', 'mac command that mounts installer.dmg', r'hdiutil\s+(attach|mount)', 'hdiutil attach installer.dmg'),
    ('mac', 'new', 'turn photo.png into a jpeg from the macOS terminal', r'\bsips\b', 'sips -s format jpeg photo.png --out photo.jpg'),
    ('mac', 'new', 'from the terminal on a mac, remove the quarantine flag from MyApp.app', r'xattr\b.*quarantine', 'xattr -d com.apple.quarantine MyApp.app'),
    ('mac', 'new', 'is Gatekeeper on? how do i check on my mac', r'spctl\s+(--status|--assess|-a)', 'spctl --status'),
    ('mac', 'new', 'macOS: convert settings.plist to XML format', r'plutil\b.*xml1', 'plutil -convert xml1 settings.plist'),
    ('mac', 'new', 'start a Time Machine backup from the mac terminal', r'tmutil\s+startbackup', 'tmutil startbackup'),
    ('mac', 'new', 'list the macOS software updates that are available, command line only', r'softwareupdate\s+(-l|--list)', 'softwareupdate -l'),
    ('mac', 'new', 'make my mac speak the word hello out loud', r'\bsay\b.*hello', 'say hello'),
    ('mac', 'new', "mac: how do I change my computer's name from the command line", r'scutil\s+--set', 'sudo scutil --set ComputerName Studio'),
    ('mac', 'new', 'macOS: which apps are using my network right now', r'\bnettop\b|lsof\s+-i', 'nettop'),
    ('mac', 'new', "macOS: show the code signature of an app", r'codesign\s+(-d|-dv|-dvv|--display|-v|--verify)', 'codesign -dv MyApp.app'),

    ('linux', 'known', 'I run Ubuntu. Get nginx installed for me.', r'apt(-get)?\s+install\b.*nginx', 'sudo apt install nginx'),
    ('linux', 'known', "my linux server needs nginx restarted, what's the command", r'systemctl\s+restart\s+nginx|service\s+nginx\s+restart', 'sudo systemctl restart nginx'),
    ('linux', 'known', 'on linux, how can I read the logs for the ssh service', r'journalctl\b.*-u\s*ssh', 'journalctl -u ssh'),
    ('linux', 'known', 'linux: how much RAM is free right now', r'\bfree\b|/proc/meminfo', 'free -h'),
    ('linux', 'known', 'which process holds port 5432 on linux?', r'\b(ss|netstat|lsof|fuser)\b.*5432', 'sudo ss -ltnp | grep :5432'),
    ('linux', 'known', "copy the text of notes.txt to the clipboard, I'm on Linux", r'\b(xclip|xsel|wl-copy)\b', 'xclip -selection clipboard < notes.txt'),
    ('linux', 'known', 'upgrade every package on my Debian box', r'apt(-get)?\s+(full-|dist-)?upgrade', 'sudo apt update && sudo apt upgrade'),
    ('linux', 'known', "I'm on Fedora, install htop please", r'dnf\s+install\b.*htop', 'sudo dnf install htop'),
    ('linux', 'known', 'arch linux: get vim installed', r'pacman\s+-S\w*\s+.*vim', 'sudo pacman -S vim'),
    ('linux', 'known', 'change foo to bar in config.txt in place, on Linux', r"sed\s+-i\s+['\"]?s.*foo.*bar", "sed -i 's/foo/bar/g' config.txt"),
    ('linux', 'known', 'what distro and version is this linux box?', r'/etc/os-release|lsb_release|hostnamectl', 'cat /etc/os-release'),
    ('linux', 'known', 'let traffic through port 443 on my ubuntu firewall', r'ufw\s+allow\s+443', 'sudo ufw allow 443/tcp'),
    ('linux', 'new', 'add the user ana to the docker group on Linux', r'usermod\s+-a?G\s+docker\s+ana|gpasswd\s+-a\s+ana\s+docker', 'sudo usermod -aG docker ana'),
    ('linux', 'new', 'linux: change the hostname of this machine to web1', r'hostnamectl\s+(set-)?hostname\b.*web1', 'sudo hostnamectl set-hostname web1'),
    ('linux', 'new', 'print the kernel version, linux', r'uname\s+-(r|a)', 'uname -r'),
    ('linux', 'new', 'linux: what PCI hardware do I have?', r'\blspci\b', 'lspci'),
    ('linux', 'new', 'sync my photos folder to backup@nas:/photos on Linux, only sending changes', r'rsync\s+-\w*a\w*', 'rsync -av photos/ backup@nas:/photos/'),
    ('linux', 'new', 'which systemd units failed on this linux machine?', r'systemctl\s+(--failed|list-units\s+.*failed)', 'systemctl --failed'),
    ('linux', 'new', 'mount /dev/sdb1 at /mnt on Linux', r'mount\s+/dev/sdb1\s+/mnt', 'sudo mount /dev/sdb1 /mnt'),
    ('linux', 'new', 'linux: show the kernel ring buffer', r'\bdmesg\b', 'sudo dmesg'),
    ('linux', 'new', 'linux: who is logged in right now', r'^\s*(sudo\s+)?(who|w)\b', 'who'),
    ('linux', 'new', 'linux: how long did each service take at boot', r'systemd-analyze\s+blame', 'systemd-analyze blame'),
    ('linux', 'new', "linux: show the machines in this computer's ARP table", r'ip\s+neigh|\barp\b', 'ip neigh'),
    ('linux', 'new', 'list USB devices on linux', r'\blsusb\b', 'lsusb'),

    ('powershell', 'known', 'pwsh: show me everything in this folder, hidden stuff too', r'(Get-ChildItem|gci|ls|dir)\b.*-Force', 'Get-ChildItem -Force'),
    ('powershell', 'known', 'PowerShell, I want the last 50 lines of app.log', r'(Get-Content|gc|cat|type)\b.*-Tail\s+50', 'Get-Content app.log -Tail 50'),
    ('powershell', 'known', 'using powershell, look for the word timeout in server.log', r'(Select-String|sls)\b.*timeout', 'Select-String -Path server.log -Pattern "timeout"'),
    ('powershell', 'known', 'With PowerShell, make a zip of the folder site', r'Compress-Archive', 'Compress-Archive -Path site -DestinationPath site.zip'),
    ('powershell', 'known', 'in powershell, end every process called chrome', r'(Stop-Process|spps|kill)\b.*chrome', 'Stop-Process -Name chrome'),
    ('powershell', 'known', 'PowerShell: which 5 processes use the most memory?', r'(Sort-Object|sort)\b.*(WorkingSet|WS|PM|Memory).*-First\s+5', 'Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 5'),
    ('powershell', 'known', 'Set NODE_ENV to production just for this PowerShell session', r'\$env:NODE_ENV\s*=\s*[\'"]?production', '$env:NODE_ENV = "production"'),
    ('powershell', 'known', 'pwsh: save https://example.com/a.zip to a.zip', r'(Invoke-WebRequest|iwr|curl|wget)\b.*(-OutFile|-o)\b', 'Invoke-WebRequest https://example.com/a.zip -OutFile a.zip'),
    ('powershell', 'known', 'call a JSON API at https://api.example.com/items from PowerShell', r'(Invoke-RestMethod|irm|Invoke-WebRequest|iwr)\b', 'Invoke-RestMethod https://api.example.com/items'),
    ('powershell', 'known', 'PowerShell way to check port 443 on example.com is open', r'(Test-NetConnection|tnc)\b.*443', 'Test-NetConnection example.com -Port 443'),
    ('powershell', 'known', 'get the sha256 of setup.exe, in powershell', r'Get-FileHash', 'Get-FileHash setup.exe -Algorithm SHA256'),
    ('powershell', 'known', 'pwsh: list the Windows services', r'Get-Service|\bgsv\b', 'Get-Service'),
    ('powershell', 'new', 'powershell: turn the process list into JSON', r'ConvertTo-Json', 'Get-Process | ConvertTo-Json'),
    ('powershell', 'new', 'dump the services into services.csv with pwsh', r'Export-Csv', 'Get-Service | Export-Csv services.csv -NoTypeInformation'),
    ('powershell', 'new', 'load users.csv into objects in PowerShell', r'Import-Csv', 'Import-Csv users.csv'),
    ('powershell', 'new', r'PowerShell: does C:\temp\a.txt exist?', r'Test-Path', r'Test-Path C:\temp\a.txt'),
    ('powershell', 'new', 'make a PowerShell script wait 10 seconds', r'(Start-Sleep|sleep)\b.*10', 'Start-Sleep -Seconds 10'),
    ('powershell', 'new', 'which PowerShell version am I running?', r'\$PSVersionTable|\$Host\.Version|Get-Host', '$PSVersionTable.PSVersion'),
    ('powershell', 'new', 'show the examples for Get-Process in the PowerShell help', r'(Get-Help|help)\b.*Get-Process.*-Examples', 'Get-Help Get-Process -Examples'),
    ('powershell', 'new', 'pwsh: what members does a Get-Date object have?', r'Get-Member|\bgm\b', 'Get-Date | Get-Member'),
    ('powershell', 'new', 'run hostname on server01 remotely with PowerShell', r'(Invoke-Command|icm)\b.*server01', 'Invoke-Command -ComputerName server01 -ScriptBlock { hostname }'),
    ('powershell', 'new', r'how long does .\build.ps1 take? powershell', r'Measure-Command', r'Measure-Command { .\build.ps1 }'),
    ('powershell', 'new', 'powershell: only show services that are running', r'(Where-Object|where|\?).*Running', 'Get-Service | Where-Object Status -eq Running'),
    ('powershell', 'new', 'read a value from the registry with PowerShell', r'Get-ItemProperty(Value)?\b', 'Get-ItemProperty HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'),
]


def normal(text):
    """The words of a request, which is what training rows are compared to the test by."""
    return ' '.join(re.findall(r'[a-z0-9_]+', text.lower()))


def requests():
    return {normal(prompt) for _, _, prompt, _, _ in TESTS}


def parses(code):
    """bash -n only reads. An empty environment keeps BASH_ENV out of it."""
    return subprocess.run(['/bin/bash', '-n'], input=code, capture_output=True, text=True, timeout=20, env={}).returncode == 0


def grade(system, wanted, reply):
    blocks = FENCE.findall(reply)
    language, code = blocks[0] if blocks else ('', '')
    flags = re.I | re.M if system == 'powershell' else re.M
    checks = {'block': bool(blocks), 'label': language.lower() in LABELS[system], 'command': bool(re.search(wanted, code, flags)),
              'no_other_system': not re.search(WRONG[system], code, re.M)}
    checks['parses'] = system == 'powershell' or (bool(code) and parses(code))
    return checks, code


def check_references():
    bad = []
    for system, _, prompt, wanted, answer in TESTS:
        fence = 'powershell' if system == 'powershell' else 'zsh' if system == 'mac' else 'bash'
        checks, _ = grade(system, wanted, f'```{fence}\n{answer}\n```')
        if not all(checks.values()):
            bad.append((prompt, checks))
    print(f'{len(TESTS) - len(bad)}/{len(TESTS)} right answers pass; requests all different: {len(requests()) == len(TESTS)}')
    for row in bad:
        print('FAILS', row)
    return not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint')
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-references', action='store_true')
    args = parser.parse_args()
    if args.check_references:
        raise SystemExit(0 if check_references() else 1)
    from .model import Model
    model, header, _ = Model.load(args.checkpoint)
    started, results = time.monotonic(), []
    for system, group, prompt, wanted, _ in TESTS:
        reply = ''.join(model.generate([{'role': 'user', 'content': prompt}], max_tokens=args.max_tokens, temperature=0)).strip()
        checks, code = grade(system, wanted, reply)
        results.append({'system': system, 'group': group, 'prompt': prompt, 'reply': reply, 'code': code, **checks, 'passed': all(checks.values())})
    score = lambda rows: f"{sum(r['passed'] for r in rows)}/{len(rows)}"
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'), 'passed': score(results),
              **{system: score([r for r in results if r['system'] == system]) for system in LABELS},
              **{group: score([r for r in results if r['group'] == group]) for group in ('known', 'new')},
              'right_command': f"{sum(r['command'] for r in results)}/{len(results)}", 'right_label': f"{sum(r['label'] for r in results)}/{len(results)}",
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
