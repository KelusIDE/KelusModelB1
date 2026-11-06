"""commands-v1: terminal commands for Cide, on macOS, Linux and PowerShell (2026-11-05, user: "make 4.5 do mac cmds and linux and pwsh").

Cide 4.5 scored 0/72 on kelus.commands_eval: it had almost no commands to learn from, and asked for one it wrote Kel or Python.
Three sources:
  - TASKS below, written by hand: the same job on all three systems, so Cide learns that `pbcopy`, `xclip` and `Set-Clipboard`
    answer one request, and which goes with which system. Slots (%file%, %port%, ...) keep it from memorising one example.
  - tldr-pages (https://github.com/tldr-pages/tldr, CC BY 4.0): pages/osx, pages/linux and pages/common.
  - PowerShell-Docs (https://github.com/MicrosoftDocs/PowerShell-Docs, CC BY 4.0, code MIT): the examples of every cmdlet, and the
    PowerShell pages of tldr's pages/windows.
Offensive security tools are left out, as Cide has no need of them. Commands that delete data for good get a warning line. macOS and
Linux commands must pass `bash -n`, which only parses; downloaded text is never run. Downloads live in data/cide-v15-sources/
(git-ignored), so run this with -I.

Requests in kelus.commands_eval are never trained on, and its "new" tasks are kept out of TASKS (this script stops if they get in).

    .venv/bin/python -I data/make_commands.py   # writes data/commands-v1.jsonl
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.commands_eval import TESTS, normal, requests  # noqa: E402

SOURCES = ROOT / 'data/cide-v15-sources'
FENCES = {'mac': 'zsh', 'linux': 'bash', 'powershell': 'powershell'}
NAMES = {'mac': 'macOS', 'linux': 'Linux', 'powershell': 'PowerShell'}
ASKS = {
    'mac': ['On my Mac, how do I {}?', 'How do I {} on macOS?', 'mac terminal: {}', 'macOS command to {}', 'In Terminal on a Mac, {}.',
            'What is the command to {} on my MacBook?', 'how to {} on mac', 'zsh on macOS: how do I {}?', 'Give me the macOS Terminal command to {}.', 'mac: {}'],
    'linux': ['On Linux, how do I {}?', 'How do I {} on Ubuntu?', 'linux terminal: {}', 'Linux command to {}', 'In bash on Linux, {}.',
              'What is the command to {} on my Linux server?', 'how to {} on linux', 'Debian: how do I {}?', 'Give me the Linux command to {}.', 'linux: {}'],
    'powershell': ['In PowerShell, how do I {}?', 'How do I {} with PowerShell?', 'powershell: {}', 'PowerShell command to {}', 'pwsh: {}',
                   'What is the PowerShell cmdlet to {}?', 'how to {} in powershell', 'In Windows PowerShell, how do I {}?', 'Give me the PowerShell command to {}.', 'pwsh command to {}'],
}
SLOTS = {
    'file': ['notes.txt', 'todo.txt', 'report.md', 'data.csv', 'readme.txt', 'config.yaml', 'draft.txt'],
    'log': ['app.log', 'error.log', 'build.log', 'access.log', 'debug.log'],
    'dir': ['photos', 'site', 'docs', 'backup', 'projects', 'logs', 'src'],
    'port': ['3000', '8000', '5000', '9000', '8443', '3306', '6379', '27017'],
    'num': ['5', '10', '20', '30', '100'],
    'word': ['error', 'warning', 'TODO', 'failed', 'login'],
    'host': ['example.com', 'example.org', 'github.com', 'server.example.net'],
    'png': ['screen.png', 'capture.png', 'desktop.png'],
    'cask': ['firefox', 'visual-studio-code', 'iterm2', 'docker'],
    'module': ['Pester', 'PSReadLine', 'ImportExcel', 'Az'],
}
BY_SYSTEM = {  # the same job names different things on each system. Names in the test (ffmpeg, htop, chrome...) stay out, to measure generalising.
    'pkg': {'mac': ['wget', 'git', 'node', 'python3', 'jq', 'tree', 'tmux', 'ripgrep', 'neovim', 'redis'],
            'linux': ['wget', 'git', 'curl', 'python3', 'jq', 'tree', 'tmux', 'ripgrep', 'neovim', 'redis-server'],
            'powershell': ['Git.Git', 'Python.Python.3.12', 'OpenJS.NodeJS', 'jqlang.jq', 'Mozilla.Firefox', '7zip.7zip']},
    'svc': {'mac': ['nginx', 'redis', 'postgresql', 'mysql'], 'linux': ['nginx', 'docker', 'ssh', 'postgresql', 'redis', 'apache2', 'cron'],
            'powershell': ['Spooler', 'W32Time', 'WinRM', 'BITS', 'wuauserv']},
    'proc': {'mac': ['Safari', 'Slack', 'Spotify', 'Notes', 'node'], 'linux': ['firefox', 'node', 'python3', 'gedit', 'slack'],
             'powershell': ['notepad', 'msedge', 'explorer', 'node', 'Teams']},
}
SLOT = re.compile(r'%(file|log|dir|port|num|word|host|png|cask|module|pkg|svc|proc)%')

# (ways to ask, macOS, Linux, PowerShell, notes per system). None: the system has no sensible answer, or the test keeps it out.
TASKS = [
    (['list all files, including hidden ones', 'show hidden files in this folder', 'see every file in the folder with details'], 'ls -la', 'ls -la', 'Get-ChildItem -Force'),
    (['show which folder I am in', 'print the current directory'], 'pwd', 'pwd', 'Get-Location'),
    (['go to my Documents folder', 'change directory to Documents'], 'cd ~/Documents', 'cd ~/Documents', 'Set-Location ~/Documents'),
    (['make a new folder called %dir%', 'create the directory %dir%'], 'mkdir -p %dir%', 'mkdir -p %dir%', 'New-Item -ItemType Directory -Path %dir%'),
    (['create an empty file called %file%', 'make a blank file named %file%'], 'touch %file%', 'touch %file%', 'New-Item -ItemType File -Path %file%'),
    (['copy %file% to backup-%file%', 'duplicate the file %file% as backup-%file%'], 'cp %file% backup-%file%', 'cp %file% backup-%file%', 'Copy-Item %file% backup-%file%'),
    (['copy the folder %dir% with everything in it', 'duplicate the %dir% directory recursively'], 'cp -R %dir% %dir%-copy', 'cp -r %dir% %dir%-copy', 'Copy-Item %dir% %dir%-copy -Recurse'),
    (['rename %file% to final-%file%', 'change the name of %file% to final-%file%'], 'mv %file% final-%file%', 'mv %file% final-%file%', 'Rename-Item %file% final-%file%'),
    (['move %file% into the %dir% folder', 'put %file% in the %dir% directory'], 'mv %file% %dir%/', 'mv %file% %dir%/', 'Move-Item %file% %dir%/'),
    (['delete the file %file%', 'remove %file%'], 'rm %file%', 'rm %file%', 'Remove-Item %file%'),
    (['delete the folder %dir% and everything inside it', 'remove the directory %dir% recursively'], 'rm -r %dir%', 'rm -r %dir%', 'Remove-Item %dir% -Recurse'),
    (['show what is in %file%', 'print the contents of %file%', 'read the file %file%'], 'cat %file%', 'cat %file%', 'Get-Content %file%'),
    (['show the first %num% lines of %file%', 'see the start of %file%'], 'head -n %num% %file%', 'head -n %num% %file%', 'Get-Content %file% -TotalCount %num%'),
    (['show the last %num% lines of %log%', 'see the end of %log%'], 'tail -n %num% %log%', 'tail -n %num% %log%', 'Get-Content %log% -Tail %num%'),
    (['watch %log% as new lines arrive', 'follow %log% live'], 'tail -f %log%', 'tail -f %log%', 'Get-Content %log% -Wait -Tail 10'),
    (['count the lines in %file%', 'find how many lines %file% has'], 'wc -l %file%', 'wc -l %file%', '(Get-Content %file%).Count'),
    (['search %log% for the word %word%', 'find lines containing %word% in %log%'], 'grep "%word%" %log%', 'grep "%word%" %log%', 'Select-String -Path %log% -Pattern "%word%"'),
    (['search every file in this folder for %word%', 'find %word% in all files under this directory'], 'grep -r "%word%" .', 'grep -r "%word%" .', 'Get-ChildItem -Recurse -File | Select-String -Pattern "%word%"'),
    (['find a file named %file%', 'look for %file% under this folder'], 'find . -name "%file%"', 'find . -name "%file%"', 'Get-ChildItem -Recurse -Filter %file%'),
    (['search the whole computer for a file called %file%'], 'mdfind -name %file%', 'locate %file%', r'Get-ChildItem C:\ -Recurse -Filter %file% -ErrorAction SilentlyContinue'),
    (['find files bigger than 100 MB', 'list large files over 100MB'], 'find . -type f -size +100M', 'find . -type f -size +100M', 'Get-ChildItem -Recurse -File | Where-Object Length -GT 100MB'),
    (['see how big the folder %dir% is', 'show the size of the %dir% directory'], 'du -sh %dir%', 'du -sh %dir%', '(Get-ChildItem %dir% -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB'),
    (['check free disk space', 'see how much disk space is left'], 'df -h', 'df -h', 'Get-PSDrive -PSProvider FileSystem'),
    (['zip the folder %dir%', 'make a zip file from %dir%'], 'zip -r %dir%.zip %dir%', 'zip -r %dir%.zip %dir%', 'Compress-Archive -Path %dir% -DestinationPath %dir%.zip'),
    (['unzip %dir%.zip', 'extract the zip file %dir%.zip'], 'unzip %dir%.zip', 'unzip %dir%.zip', 'Expand-Archive %dir%.zip -DestinationPath %dir%'),
    (['make a tar.gz of the folder %dir%', 'compress %dir% into a tarball'], 'tar -czf %dir%.tar.gz %dir%', 'tar -czf %dir%.tar.gz %dir%', 'tar -czf %dir%.tar.gz %dir%'),
    (['extract %dir%.tar.gz', 'unpack the tarball %dir%.tar.gz'], 'tar -xzf %dir%.tar.gz', 'tar -xzf %dir%.tar.gz', 'tar -xzf %dir%.tar.gz'),
    (['make run.sh executable', 'give the script run.sh permission to run'], 'chmod +x run.sh', 'chmod +x run.sh', None),
    (['allow scripts to run', 'change the execution policy so my scripts run'], None, None, 'Set-ExecutionPolicy -Scope CurrentUser RemoteSigned'),
    (['show the permissions of %file%', 'see who can read %file%'], 'ls -l %file%', 'ls -l %file%', 'Get-Acl %file%'),
    (['make only me able to read %file%', 'set %file% to private'], 'chmod 600 %file%', 'chmod 600 %file%', None),
    (['change the owner of %file% to alex', 'make alex the owner of %file%'], 'sudo chown alex %file%', 'sudo chown alex %file%', 'icacls %file% /setowner alex'),
    (['change the owner of the folder %dir% and all its files to alex'], 'sudo chown -R alex %dir%', 'sudo chown -R alex %dir%', None),
    (['open %file% in its default app', 'open the file %file%'], 'open %file%', 'xdg-open %file%', 'Invoke-Item %file%'),
    (['open this folder in the file manager', 'show this folder in a window'], 'open .', 'xdg-open .', 'Invoke-Item .'),
    (['compare the files a.txt and b.txt', 'see the differences between a.txt and b.txt'], 'diff a.txt b.txt', 'diff a.txt b.txt', 'Compare-Object (Get-Content a.txt) (Get-Content b.txt)'),
    (['get the SHA-256 checksum of %file%', 'compute the sha256 hash of %file%'], 'shasum -a 256 %file%', 'sha256sum %file%', 'Get-FileHash %file% -Algorithm SHA256'),
    (['sort the lines of %file%', 'put the lines of %file% in order'], 'sort %file%', 'sort %file%', 'Get-Content %file% | Sort-Object'),
    (['remove duplicate lines from %file%', 'show only the unique lines of %file%'], 'sort %file% | uniq', 'sort %file% | uniq', 'Get-Content %file% | Sort-Object -Unique'),
    (['replace old with new everywhere in %file%', 'change every old to new inside %file% in place'], "sed -i '' 's/old/new/g' %file%", "sed -i 's/old/new/g' %file%", "(Get-Content %file%) -replace 'old', 'new' | Set-Content %file%",
     {'mac': 'The `sed` on macOS needs the empty quotes after `-i`.'}),
    (['make a symbolic link called app to ~/projects/app', 'create a symlink named app that points to ~/projects/app'], 'ln -s ~/projects/app app', 'ln -s ~/projects/app app', 'New-Item -ItemType SymbolicLink -Path app -Target ~/projects/app'),
    (['write hello into %file%', 'save the word hello to %file%'], 'echo "hello" > %file%', 'echo "hello" > %file%', '"hello" | Set-Content %file%'),
    (['append hello to %file%', 'add a line of text to the end of %file%'], 'echo "hello" >> %file%', 'echo "hello" >> %file%', 'Add-Content %file% "hello"'),
    (['list the most recently changed files', 'sort files by newest first'], 'ls -lt', 'ls -lt', 'Get-ChildItem | Sort-Object LastWriteTime -Descending'),
    (['find out how many files are in this folder', 'count the files here'], 'ls | wc -l', 'ls | wc -l', '(Get-ChildItem).Count'),
    (['copy %file% to the clipboard', 'put the contents of %file% on the clipboard'], 'pbcopy < %file%', 'xclip -selection clipboard < %file%', 'Get-Content %file% | Set-Clipboard'),
    (['paste the clipboard into the terminal', 'print what is on the clipboard'], 'pbpaste', 'xclip -selection clipboard -o', 'Get-Clipboard'),
    (['save the output of a command to output.txt', 'send the file list to output.txt'], 'ls > output.txt', 'ls > output.txt', 'Get-ChildItem | Out-File output.txt'),
    (['save the errors of a script to errors.log', 'send error output to errors.log'], './build.sh 2> errors.log', './build.sh 2> errors.log', './build.ps1 2> errors.log'),
    (['loop over every .txt file and print its name', 'print the name of each .txt file in a loop'], 'for f in *.txt; do echo "$f"; done', 'for f in *.txt; do echo "$f"; done', 'Get-ChildItem *.txt | ForEach-Object { $_.Name }'),
    (['loop from 1 to 5 and double each number', 'print each number from 1 to 5 times two'], 'for i in {1..5}; do echo $((i * 2)); done', 'for i in {1..5}; do echo $((i * 2)); done', '1..5 | ForEach-Object { $_ * 2 }'),
    (['run the tests only if the build works', 'run a second command only if the first one succeeds'], 'make && make test', 'make && make test', 'make; if ($?) { make test }'),
    (['ask for a name and keep the answer in a variable', 'read input from the user into a variable'], 'read -r name', 'read -r name', '$name = Read-Host "Name"'),
    (['clear the terminal screen', 'clean up the screen'], 'clear', 'clear', 'Clear-Host'),
    (['show the manual for ls', 'get help for the list command'], 'man ls', 'man ls', 'Get-Help Get-ChildItem'),

    (['list running processes', 'show everything that is running'], 'ps aux', 'ps aux', 'Get-Process'),
    (['see which processes use the most CPU', 'show the top CPU processes'], 'top -o cpu', 'top -o %CPU', 'Get-Process | Sort-Object CPU -Descending | Select-Object -First 10'),
    (['see which processes use the most memory', 'show the biggest memory users'], 'top -o mem', 'ps aux --sort=-%mem | head', 'Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10'),
    (['stop the process called %proc%', 'kill %proc% by name', 'close %proc% from the command line'], 'pkill %proc%', 'pkill %proc%', 'Stop-Process -Name %proc%'),
    (['kill the process with PID 1234', 'stop process id 1234'], 'kill 1234', 'kill 1234', 'Stop-Process -Id 1234'),
    (['see what is using port %port%', 'find which program is on port %port%'], 'lsof -nP -iTCP:%port% -sTCP:LISTEN', 'sudo ss -ltnp | grep :%port%', 'Get-NetTCPConnection -LocalPort %port%'),
    (['see how much memory is free', 'check my RAM usage'], 'vm_stat', 'free -h', 'Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize'),
    (['see my operating system version', 'check which OS version this is'], 'sw_vers', 'cat /etc/os-release', 'Get-ComputerInfo | Select-Object OsName, OsVersion'),
    (['see how long the computer has been on', 'check the uptime'], 'uptime', 'uptime', '(Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime'),
    (['print the name of this computer', 'show the hostname'], 'hostname', 'hostname', '$env:COMPUTERNAME'),
    (['see which user I am logged in as', 'print my username'], 'whoami', 'whoami', '$env:USERNAME'),
    (['see my hardware details', 'show what hardware this machine has'], 'system_profiler SPHardwareDataType', 'sudo lshw -short', 'Get-CimInstance Win32_ComputerSystem'),
    (['see which CPU this machine has', 'show the processor model'], 'sysctl -n machdep.cpu.brand_string', 'lscpu', 'Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores'),
    (['check the battery level', 'see the battery status'], 'pmset -g batt', 'upower -i /org/freedesktop/UPower/devices/battery_BAT0', 'Get-CimInstance Win32_Battery | Select-Object EstimatedChargeRemaining'),
    (['stop the computer going to sleep', 'keep the machine awake'], 'caffeinate -d', 'systemd-inhibit --what=idle sleep 3600', 'powercfg /change standby-timeout-ac 0'),
    (['shut down the computer', 'power off this machine'], 'sudo shutdown -h now', 'sudo shutdown now', 'Stop-Computer'),
    (['restart the computer', 'reboot this machine'], 'sudo shutdown -r now', 'sudo reboot', 'Restart-Computer'),
    (['print the PATH', 'show my PATH variable'], 'echo $PATH', 'echo $PATH', '$env:PATH'),
    (['show all environment variables', 'list my environment variables'], 'env', 'env', 'Get-ChildItem Env:'),
    (['set API_URL to http://localhost:%port% for this session', 'set the environment variable API_URL to http://localhost:%port% for now'], 'export API_URL=http://localhost:%port%', 'export API_URL=http://localhost:%port%', '$env:API_URL = "http://localhost:%port%"'),
    (['set API_URL to http://localhost:3000 permanently', 'save the environment variable API_URL for good'], "echo 'export API_URL=http://localhost:3000' >> ~/.zshrc", "echo 'export API_URL=http://localhost:3000' >> ~/.bashrc", '[Environment]::SetEnvironmentVariable("API_URL", "http://localhost:3000", "User")'),
    (['add the folder ~/bin to my PATH', 'put ~/bin on my PATH for good'], """echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc""", """echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc""", '[Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$HOME\\bin", "User")'),
    (['show my command history', 'see the commands I ran before'], 'history', 'history', 'Get-History'),
    (['see where the git command lives', 'find out which git program runs'], 'which git', 'which git', 'Get-Command git'),
    (['make a shortcut command called ll for listing files', 'create an alias named ll'], "alias ll='ls -la'", "alias ll='ls -la'", 'Set-Alias ll Get-ChildItem'),
    (['run a script in the background', 'keep server.sh running after I log out'], 'nohup ./server.sh &', 'nohup ./server.sh &', 'Start-Job -FilePath ./server.ps1'),
    (['list my background jobs', 'see which jobs are running in this shell'], 'jobs', 'jobs', 'Get-Job'),
    (['schedule a daily job', 'edit my scheduled jobs'], 'crontab -e', 'crontab -e', 'Register-ScheduledTask -TaskName Backup -Action (New-ScheduledTaskAction -Execute "backup.cmd") -Trigger (New-ScheduledTaskTrigger -Daily -At 9am)'),
    (['print the date and time', "show today's date"], 'date', 'date', 'Get-Date'),
    (['print the date as year-month-day', 'show the date in YYYY-MM-DD form'], 'date +%Y-%m-%d', 'date +%Y-%m-%d', 'Get-Date -Format yyyy-MM-dd'),
    (['get the current Unix timestamp', 'print the time in seconds since 1970'], 'date +%s', 'date +%s', '[DateTimeOffset]::Now.ToUnixTimeSeconds()'),
    (['set the time zone to UTC', 'change the time zone to UTC'], 'sudo systemsetup -settimezone UTC', 'sudo timedatectl set-timezone UTC', 'Set-TimeZone -Id UTC'),
    (['read the system log from the last hour', 'see recent system logs'], 'log show --last 1h', 'journalctl --since "1 hour ago"', 'Get-WinEvent -LogName System -MaxEvents 50'),
    (['start a tmux session called work', 'open a session that survives disconnecting'], 'tmux new -s work', 'tmux new -s work', None),
    (['list the groups I belong to', 'see my groups'], 'groups', 'groups', 'whoami /groups'),

    (['list the services', 'see all services on this machine'], 'launchctl list', 'systemctl list-units --type=service', 'Get-Service'),
    (['start the %svc% service', 'turn on the %svc% service'], 'brew services start %svc%', 'sudo systemctl start %svc%', 'Start-Service %svc%'),
    (['stop the %svc% service', 'turn off the %svc% service'], 'brew services stop %svc%', 'sudo systemctl stop %svc%', 'Stop-Service %svc%'),
    (['restart the %svc% service', 'restart %svc%'], 'brew services restart %svc%', 'sudo systemctl restart %svc%', 'Restart-Service %svc%'),
    (['see whether %svc% is running', 'check the status of the %svc% service'], 'brew services info %svc%', 'systemctl status %svc%', 'Get-Service %svc%'),
    (['start %svc% automatically at boot', 'make the %svc% service start on its own'], 'brew services start %svc%', 'sudo systemctl enable %svc%', 'Set-Service %svc% -StartupType Automatic'),
    (['read the logs of the %svc% service', 'see the log output of %svc%'], """log show --predicate 'process == "%svc%"' --last 1h""", 'journalctl -u %svc%', None),
    (['follow the logs of the %svc% service live'], None, 'journalctl -u %svc% -f', None),
    (['reload systemd after I edit a unit file'], None, 'sudo systemctl daemon-reload', None),
    (['list the loaded kernel modules'], None, 'lsmod', None),

    (['install %pkg%', 'get %pkg% installed', 'add the package %pkg%'], 'brew install %pkg%', 'sudo apt install %pkg%', 'winget install %pkg%'),
    (['update all my packages', 'upgrade everything that is installed'], 'brew update && brew upgrade', 'sudo apt update && sudo apt upgrade', 'winget upgrade --all'),
    (['uninstall %pkg%', 'remove the package %pkg%'], 'brew uninstall %pkg%', 'sudo apt remove %pkg%', 'winget uninstall %pkg%'),
    (['search for a package called %pkg%', 'look for %pkg% in the package manager'], 'brew search %pkg%', 'apt search %pkg%', 'winget search %pkg%'),
    (['list the installed packages', 'see what I have installed'], 'brew list', 'apt list --installed', 'winget list'),
    (['install Homebrew', 'set up the brew package manager'], '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', None, None),
    (['install the Xcode command line tools', 'get the developer tools'], 'xcode-select --install', None, None),
    (['install the app %cask% with Homebrew', 'install %cask% as a Homebrew cask'], 'brew install --cask %cask%', None, None),
    (['install the module %module%', 'get the module %module% from the gallery'], None, None, 'Install-Module %module% -Scope CurrentUser'),
    (['list the modules I can use', 'see the available modules'], None, None, 'Get-Module -ListAvailable'),
    (['edit my profile', 'open my PowerShell profile'], None, None, 'notepad $PROFILE'),
    (['list all the commands with the verb Get', 'see every Get command'], None, None, 'Get-Command -Verb Get'),
    (['record everything in this session to session.log', 'start a transcript in session.log'], None, None, 'Start-Transcript -Path session.log'),
    (['open a remote session on server02', 'connect a session to the computer server02'], None, None, 'Enter-PSSession -ComputerName server02'),
    (['list installed hotfixes', 'see which Windows updates are installed'], None, None, 'Get-HotFix'),
    (['list scheduled tasks', 'see the scheduled tasks'], None, None, 'Get-ScheduledTask'),
    (['set the system volume to 50 percent', 'change the sound volume to half'], "osascript -e 'set volume output volume 50'", 'amixer set Master 50%', None),

    (['list the disks', 'show all disks and volumes'], 'diskutil list', 'lsblk', 'Get-Disk'),
    (['eject the disk disk2', 'safely remove the external disk'], 'diskutil eject /dev/disk2', None, None),
    (['unmount the USB drive', 'unmount the disk partition'], 'diskutil unmount /dev/disk2s1', 'sudo umount /dev/sdb1', None),
    (['take a screenshot and save it as %png%', 'capture the screen to %png%'], 'screencapture %png%', 'gnome-screenshot -f %png%', None),
    (['open the app %proc%', 'launch %proc%'], 'open -a %proc%', '%proc% &', 'Start-Process %proc%'),
    (['open https://%host% in the browser', 'open the website %host% from the terminal'], 'open https://%host%', 'xdg-open https://%host%', 'Start-Process https://%host%'),
    (['show a desktop notification saying Done', 'pop up a notification when a job finishes'], """osascript -e 'display notification "Done" with title "Build"'""", 'notify-send "Build" "Done"', None),
    (['restart the Dock', 'reload the Dock'], 'killall Dock', None, None),
    (['restart Finder', 'reload the Finder'], 'killall Finder', None, None),
    (['see the power settings', 'show the power plans'], 'pmset -g', None, 'powercfg /list'),
    (['rebuild the Spotlight index', 'reindex Spotlight'], 'sudo mdutil -E /', None, None),

    (['find my local IP address', 'see my IP on the network'], 'ipconfig getifaddr en0', 'hostname -I', 'Get-NetIPAddress -AddressFamily IPv4'),
    (['find my public IP address', 'see what IP the internet sees'], 'curl -s https://ifconfig.me', 'curl -s https://ifconfig.me', 'Invoke-RestMethod https://ifconfig.me'),
    (['ping %host% four times', 'check if %host% responds'], 'ping -c 4 %host%', 'ping -c 4 %host%', 'Test-Connection %host% -Count 4'),
    (['check whether port %port% is open on %host%', 'test if %host% accepts connections on port %port%'], 'nc -zv %host% %port%', 'nc -zv %host% %port%', 'Test-NetConnection %host% -Port %port%'),
    (['look up the DNS records of %host%', 'do a DNS lookup of %host%'], 'dig %host%', 'dig %host%', 'Resolve-DnsName %host%'),
    (['flush the DNS cache', 'clear the DNS cache'], 'sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder', 'resolvectl flush-caches', 'Clear-DnsClientCache'),
    (['download https://%host%/file.zip', 'save https://%host%/file.zip to disk'], 'curl -LO https://%host%/file.zip', 'wget https://%host%/file.zip', 'Invoke-WebRequest https://%host%/file.zip -OutFile file.zip'),
    (['get JSON from https://%host%/api/items', 'call the API at https://%host%/api/items'], 'curl -s https://%host%/api/items', 'curl -s https://%host%/api/items', 'Invoke-RestMethod https://%host%/api/items'),
    (['send a JSON POST request to https://%host%/api/items', 'post JSON to https://%host%/api/items'], """curl -X POST -H "Content-Type: application/json" -d '{"name":"Ana"}' https://%host%/api/items""", """curl -X POST -H "Content-Type: application/json" -d '{"name":"Ana"}' https://%host%/api/items""", """Invoke-RestMethod -Method Post -Uri https://%host%/api/items -ContentType application/json -Body '{"name":"Ana"}'"""),
    (['connect to %host% over SSH', 'ssh into %host%'], 'ssh alex@%host%', 'ssh alex@%host%', 'ssh alex@%host%'),
    (['copy %file% to my server %host%', 'send %file% to %host% with scp'], 'scp %file% alex@%host%:~/', 'scp %file% alex@%host%:~/', 'scp %file% alex@%host%:~/'),
    (['make a new SSH key', 'generate an SSH key'], 'ssh-keygen -t ed25519', 'ssh-keygen -t ed25519', 'ssh-keygen -t ed25519'),
    (['see the name of my Wi-Fi network', 'show which Wi-Fi I am on'], 'networksetup -getairportnetwork en0', 'nmcli -t -f active,ssid dev wifi', 'netsh wlan show interfaces'),
    (['turn off Wi-Fi', 'switch the Wi-Fi off'], 'networksetup -setairportpower en0 off', 'nmcli radio wifi off', 'Disable-NetAdapter -Name "Wi-Fi"'),
    (['turn on Wi-Fi', 'switch the Wi-Fi on'], 'networksetup -setairportpower en0 on', 'nmcli radio wifi on', 'Enable-NetAdapter -Name "Wi-Fi"'),
    (['list the network adapters', 'see the network interfaces'], 'networksetup -listallhardwareports', 'ip link', 'Get-NetAdapter'),
    (['show the routing table', 'see the network routes'], 'netstat -rn', 'ip route', 'Get-NetRoute'),
    (['see open network connections', 'list TCP connections'], 'netstat -an', 'ss -tan', 'Get-NetTCPConnection'),
    (['check whether the firewall is on', 'see the firewall status'], 'sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate', 'sudo ufw status', 'Get-NetFirewallProfile'),
    (['open port %port% in the firewall', 'allow traffic on port %port% through the firewall'], None, 'sudo ufw allow %port%/tcp', 'New-NetFirewallRule -DisplayName "Port %port%" -Direction Inbound -Protocol TCP -LocalPort %port% -Action Allow'),
    (['turn the firewall on', 'enable the firewall'], None, 'sudo ufw enable', 'Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True'),
    (['list the user accounts', 'see the users on this machine'], 'dscl . list /Users', 'cut -d: -f1 /etc/passwd', 'Get-LocalUser'),
    (['add a new user called alex', 'create the user alex'], 'sudo sysadminctl -addUser alex', 'sudo adduser alex', 'New-LocalUser -Name alex'),
    (['change my password', 'set a new password for my account'], 'passwd', 'passwd', 'Set-LocalUser -Name $env:USERNAME -Password (Read-Host -AsSecureString)'),
]

# One package job, by distribution. Generic "Linux" rows above answer with apt.
DISTROS = {'apt': ['Ubuntu', 'Debian', 'Linux Mint', 'Pop!_OS'], 'dnf': ['Fedora', 'RHEL', 'Rocky Linux', 'AlmaLinux'], 'pacman': ['Arch Linux', 'Manjaro', 'Arch'],
           'zypper': ['openSUSE'], 'apk': ['Alpine', 'Alpine Linux']}
MANAGERS = {'apt': ('sudo apt install %pkg%', 'sudo apt remove %pkg%', 'sudo apt update && sudo apt upgrade', 'apt search %pkg%'),
            'dnf': ('sudo dnf install %pkg%', 'sudo dnf remove %pkg%', 'sudo dnf upgrade', 'dnf search %pkg%'),
            'pacman': ('sudo pacman -S %pkg%', 'sudo pacman -R %pkg%', 'sudo pacman -Syu', 'pacman -Ss %pkg%'),
            'zypper': ('sudo zypper install %pkg%', 'sudo zypper remove %pkg%', 'sudo zypper update', 'zypper search %pkg%'),
            'apk': ('sudo apk add %pkg%', 'sudo apk del %pkg%', 'sudo apk update && sudo apk upgrade', 'apk search %pkg%')}
DISTRO_JOBS = ('install %pkg%', 'uninstall %pkg%', 'update all my packages', 'search for the package %pkg%')
DISTRO_ASKS = ['On {d}, how do I {j}?', '{d}: {j}', 'how to {j} on {d}', "I'm on {d}. {J}.", 'What is the {d} command to {j}?']

# Tools tldr documents that have no place in a coding assistant's teaching.
LEFT_OUT = set('''aircrack-ng airdecap-ng aireplay-ng airmon-ng airodump-ng airbase-ng airolib-ng bettercap crackle crackmapexec crunch dirb dirsearch
 ettercap evil-winrm feroxbuster ffuf fcrackzip gobuster hashcat hashid hping hping3 hydra impacket impacket-* john kerbrute lynis-audit macchanger masscan
 medusa metasploit mimikatz msfconsole msfvenom ncrack netexec nikto ophcrack pdfcrack responder scapy searchsploit slowhttptest sqlmap sublist3r theharvester
 unshadow wafw00f wfuzz wifite wpscan zaproxy'''.split())
LEFT_OUT_MODULES = {'ISE', 'Microsoft.PowerShell.ODataUtils', 'Microsoft.PowerShell.Operation.Validation', 'PSScheduledJob', 'PSWorkflow', 'PSWorkflowUtility', 'PSDiagnostics', 'Microsoft.WSMan.Management'}
# Core tools get every example. Others share a smaller budget, at most two examples per page.
COMMON_CORE = set('''alias awk base64 basename bg cal cat chmod chown cmp comm cp crontab curl cut date df diff dig dirname du echo env export fg file find fold grep
 gunzip gzip head history hostname id join jobs jq kill killall less ln ls make man mkdir mktemp mv nano nc nice nohup nslookup openssl paste ping printf ps pwd read realpath
 rm rmdir rsync scp screen sed seq sftp sleep sort split ssh ssh-copy-id ssh-keygen stat sudo tail tar tee time timeout tmux top touch tr traceroute tree uniq uptime
 vim watch wc wget which who whoami xargs yes zip unzip'''.split())
LINUX_CORE = set('''apt apt-get apt-cache add-apt-repository dpkg dnf yum rpm pacman zypper apk snap flatpak systemctl journalctl service ip ss free lsblk lscpu lsusb lspci
 dmesg mount umount useradd usermod userdel groupadd passwd visudo ufw iptables nft firewall-cmd hostnamectl timedatectl localectl loginctl modprobe lsmod update-alternatives
 xclip xsel wl-copy xdg-open nmcli fdisk parted blkid mkfs swapon swapoff lsof pgrep pkill sysctl systemd-analyze systemd-run uname who w last vmstat iostat sar nproc ldd
 strace ltrace chattr lsattr getfacl setfacl gpasswd crontab at cron tune2fs ethtool ifconfig route arp resolvectl dig'''.split())
DANGER = re.compile(r'\brm\s+-\w*[rf]|Remove-Item\b.*-Recurse|\bdd\s+.*\bof=|\bmkfs\b|\bshred\b|\bwipefs\b|\bfdisk\b|\bparted\b|Format-Volume|Clear-Disk|git\s+(reset\s+--hard|clean\s+-\w*f)|\bdocker\s+system\s+prune')
WARNING = 'Careful: this cannot be undone. Check the path first.'
GENERIC = re.compile(r'^(display|show|print|list|view|get)\b.*\b(help|version|usage|commands|subcommands)\b|^(start|run|stop|create|update|delete|remove|install|build|configure|execute|launch)\b', re.I)


def lower_first(text):
    """Lower-case the first letter, unless it starts a name such as `Get-Process` or DNS."""
    word = text.split(' ', 1)[0]
    return text if word.startswith('`') or sum(c.isupper() for c in word) > 1 else text[:1].lower() + text[1:]


def fill(text, system, values):
    def value(match):
        slot = match.group(1)
        return values[slot] if slot in values else match.group(0)
    return SLOT.sub(value, text)


def pick(system, rng):
    return {**{slot: rng.choice(pool) for slot, pool in SLOTS.items()}, **{slot: rng.choice(pools[system]) for slot, pools in BY_SYSTEM.items()}}


def reply(system, code, note=''):
    parts = [f'```{FENCES[system]}\n{code}\n```']
    if DANGER.search(code):
        parts.append(WARNING)
    if note:
        parts.append(note)
    return '\n\n'.join(parts)


def row(family, source, system, prompt, answer):
    language = 'powershell' if system == 'powershell' else 'shell'
    return {'family': family, 'source': source, 'system': system, 'language': language, 'split': 'train',
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]}


def table_rows(rng, per_task=8):
    rows = []
    for task in TASKS:
        phrases, mac, linux, powershell, *rest = task
        notes = rest[0] if rest else {}
        answers = {'mac': mac, 'linux': linux, 'powershell': powershell}
        for system, answer in answers.items():
            if answer is None:
                continue
            asks = rng.sample(ASKS[system], min(per_task, len(ASKS[system])))
            for i, ask in enumerate(asks):
                values = pick(system, rng)
                prompt = ask.format(fill(phrases[i % len(phrases)], system, values))
                rows.append(row('commands_table', 'make_commands', system, prompt[:1].upper() + prompt[1:] if i % 2 else prompt, reply(system, fill(answer, system, values), notes.get(system, ''))))
        present = [s for s, a in answers.items() if a is not None]
        if len(present) == 3:  # no system named: all three
            for ask in rng.sample(['What is the terminal command to {}?', 'How do I {} from the command line?', 'command to {}', 'How can I {} in a terminal?'], 2):
                values = {s: pick(s, rng) for s in present}
                shared = pick('mac', rng)
                # Slots shared by all three, so one request fits every answer. System-specific slots (services, packages) are skipped here.
                if any(BY in phrases[0] for BY in ('%pkg%', '%svc%', '%proc%')):
                    continue
                body = '\n\n'.join(f"{NAMES[s]}:\n\n{reply(s, fill(answers[s], s, shared))}" for s in present)
                rows.append(row('commands_all', 'make_commands', 'all', ask.format(fill(phrases[0], 'mac', shared)), body))
        # Same job on another system
        for source, target in rng.sample([(a, b) for a in present for b in present if a != b], min(3, len(present) * (len(present) - 1))):
            if any(BY in answers[source] + answers[target] for BY in ('%pkg%', '%svc%', '%proc%')) or answers[source] == answers[target]:
                continue
            shared = pick('mac', rng)
            before, after = fill(answers[source], source, shared), fill(answers[target], target, shared)
            ask = rng.choice([f'What is the {NAMES[target]} version of `{before}`?', f'Convert this {NAMES[source]} command to {NAMES[target]}: `{before}`',
                              f'I know `{before}` on {NAMES[source]}. What do I use on {NAMES[target]}?', f'{NAMES[source]} to {NAMES[target]}: `{before}`'])
            rows.append(row('commands_translate', 'make_commands', target, ask, reply(target, after)))
    for manager, distros in DISTROS.items():
        for job, answer in zip(DISTRO_JOBS, MANAGERS[manager]):
            for distro in distros:
                for ask in rng.sample(DISTRO_ASKS, 3):
                    values = pick('linux', rng)
                    text = fill(job, 'linux', values)
                    rows.append(row('commands_distro', 'make_commands', 'linux', ask.format(d=distro, j=text, J=text[:1].upper() + text[1:]), reply('linux', fill(answer, 'linux', values))))
    return rows


def unwrap(command):
    """tldr's {{placeholders}} and {{[-a|--all]}} choices, as plain text. Returns the command and the placeholders to mention."""
    names = []

    def plain(match):
        inner = match.group(1)
        if re.fullmatch(r'\[.*\]', inner):  # {{[-a|--all]}}: the first choice
            return inner[1:-1].split('|', 1)[0]
        if '|' in inner:  # {{start|stop}}
            return inner.split('|', 1)[0]
        inner = re.sub(r'\s+\.\.\.$', '', inner)
        if ('/' in inner or '_' in inner or inner.isalpha() and inner.islower() and len(inner) > 1) and inner not in names:
            names.append(inner.split(' ')[0])
        return inner
    return re.sub(r'\{\{(.*?)\}\}', plain, command), names


def tldr_pages(folder):
    for path in sorted((SOURCES / 'tldr/pages' / folder).glob('*.md')):
        name = path.stem
        if name in LEFT_OUT or name.startswith('impacket-'):
            continue
        lines = path.read_text(encoding='utf-8').splitlines()
        summary = next((line[2:].strip() for line in lines if line.startswith('> ') and not line.startswith('> More information') and not line.startswith('> See also')), '')
        if 'alias of' in summary.lower():  # nothing to learn from `gwho`, an alias of GNU `who`
            summary = ''
        examples = [(lines[i][2:].rstrip(':').strip(), lines[i + 2].strip().strip('`')) for i in range(len(lines) - 2) if lines[i].startswith('- ') and lines[i + 2].startswith('`')]
        yield name, summary, examples, '\n'.join(lines[:8])


def tldr_row(family, system, name, description, command, rng):
    code, placeholders = unwrap(command)
    task = lower_first(description).replace('`stdin`', 'standard input').replace('`stdout`', 'standard output')
    if len(task.split()) <= 4 or GENERIC.search(task) or rng.random() < 0.35:
        task = f'{task} with `{name}`'
    note = f'Replace `{placeholders[0]}` with your own value.' if placeholders else ''
    return row(family, 'tldr-pages', system, rng.choice(ASKS[system]).format(task), reply(system, code, note))


def tldr_rows(rng):
    rows = []
    for name, summary, examples, _ in tldr_pages('osx'):
        rows += [tldr_row('commands_tldr', 'mac', name, d, c, rng) for d, c in examples]
        if summary:
            rows.append(row('commands_about', 'tldr-pages', 'mac', rng.choice([f'What does `{name}` do on a Mac?', f'macOS: what is `{name}`?', f'what is the {name} command on macos']), summary))
    others = []
    for name, summary, examples, _ in tldr_pages('linux'):
        if name in LINUX_CORE:
            rows += [tldr_row('commands_tldr', 'linux', name, d, c, rng) for d, c in examples]
            if summary:
                rows.append(row('commands_about', 'tldr-pages', 'linux', rng.choice([f'What does `{name}` do on Linux?', f'linux: what is `{name}`?', f'what is the {name} command on linux']), summary))
        else:
            others += [tldr_row('commands_tldr', 'linux', name, d, c, rng) for d, c in rng.sample(examples, min(2, len(examples)))]
    rows += rng.sample(others, min(1500, len(others)))
    common = []
    for name, summary, examples, _ in tldr_pages('common'):
        if name in COMMON_CORE:
            for d, c in examples:
                rows.append(tldr_row('commands_tldr', rng.choice(['mac', 'linux']), name, d, c, rng))
        else:
            common += [(name, d, c) for d, c in rng.sample(examples, min(1, len(examples)))]
    for name, d, c in rng.sample(common, min(1400, len(common))):
        rows.append(tldr_row('commands_tldr', rng.choice(['mac', 'linux']), name, d, c, rng))
    for name, summary, examples, head in tldr_pages('windows'):
        if '-' in name and 'powershell' in head.lower():
            rows += [tldr_row('commands_tldr', 'powershell', name, d, c, rng) for d, c in examples for _ in range(2)]
    return rows


def docs_examples():
    """(cmdlet, title, code, first sentence) from the PowerShell reference: version 7.5, then what only 5.1 has."""
    reference, seen = SOURCES / 'powershell-docs/reference', set()
    for version in ('7.5', '5.1'):
        for path in sorted((reference / version).rglob('*.md')):
            if path.parent.name in LEFT_OUT_MODULES or path.parent.name == 'About' or path.stem.lower().startswith('about_') or path.stem in seen or '-' not in path.stem:
                continue
            seen.add(path.stem)
            text = path.read_text(encoding='utf-8')
            for match in re.finditer(r'^### Example \d+\s*[:\-.]?\s*((?:(?!^###).)+?)\n\s*\n```powershell[^\n]*\n(.*?)```\s*(.*?)(?=\n\s*\n|\Z)', text, re.M | re.S | re.I):
                # A paragraph can sit between the heading and the code: the title is the heading's own line.
                title, code, after = match.group(1).split('\n', 1)[0].strip(), match.group(2).rstrip(), re.sub(r'\s+', ' ', match.group(3)).strip()
                code = re.sub(r'^PS[^>\n]*>\s?', '', code, flags=re.M)
                if not title or len(code) > 500 or code.count('\n') >= 10 or '```' in after:
                    continue
                after = re.sub(r'\*\*(.+?)\*\*', r'\1', re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', after))
                sentence = next(iter(re.findall(r'^.+?[.](?=\s|$)', after)), '')
                yield path.stem, title.rstrip('.'), code, sentence if len(sentence) <= 180 else ''


def docs_rows(rng, wordings=3):
    rows = []
    for cmdlet, title, code, sentence in docs_examples():
        for ask in rng.sample(ASKS['powershell'], wordings):
            task = lower_first(title)
            if cmdlet.lower() not in task.lower() and rng.random() < 0.3:
                task = f'{task} with `{cmdlet}`'
            rows.append(row('commands_docs', 'powershell-docs', 'powershell', ask.format(task), reply('powershell', code, sentence)))
    return rows


def parses(item):
    return subprocess.run(['/bin/bash', '-n'], input=item, capture_output=True, text=True, timeout=20, env={}).returncode == 0


def check_table():
    """The test's "new" tasks must really be missing from TASKS, or its score would claim too much."""
    answers = [(s, a) for task in TASKS for s, a in zip(('mac', 'linux', 'powershell'), task[1:4]) if a]
    bad = [(prompt, a) for system, group, prompt, wanted, _ in TESTS if group == 'new'
           for s, a in answers if s == system and re.search(wanted, a, re.I | re.M if system == 'powershell' else re.M)]
    if bad:
        raise SystemExit(f'"new" test tasks are in TASKS: {bad}')


def build(seed=2461):
    check_table()
    rng = random.Random(seed)
    rows = table_rows(rng) + tldr_rows(rng) + docs_rows(rng)
    held, seen, dropped, kept = requests(), set(), collections.Counter(), []
    for r in rows:
        key = normal(r['messages'][0]['content'])
        code = re.findall(r'```\w*\n(.*?)\n```', r['messages'][1]['content'], re.S)
        if key in held:
            dropped['test request'] += 1
        elif (key, r['messages'][1]['content']) in seen:
            dropped['repeat'] += 1
        elif any('{{' in c or '}}' in c and '{' not in c for c in code):
            dropped['placeholder left'] += 1
        else:
            seen.add((key, r['messages'][1]['content']))
            kept.append((r, code))
    shells = [(r, '\n'.join(code)) for r, code in kept if r['system'] in ('mac', 'linux') and code]
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(parses, [code for _, code in shells]))
    broken = {id(r) for (r, _), ok in zip(shells, verdicts) if not ok}
    dropped['bash -n failed'] += len(broken)
    rows = [r for r, _ in kept if id(r) not in broken]
    rng.shuffle(rows)
    per_system = collections.Counter()
    for r in rows:  # 60 per system to watch validation loss; the test itself is kelus.commands_eval
        if r['family'] in ('commands_table', 'commands_tldr', 'commands_docs') and per_system[r['system']] < 60:
            r['split'] = 'validation'
            per_system[r['system']] += 1
    return rows, dropped


if __name__ == '__main__':
    rows, dropped = build()
    destination = ROOT / 'data/commands-v1.jsonl'
    line = lambda r: json.dumps(r, ensure_ascii=False).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029') + '\n'
    destination.write_text(''.join(line(r) for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    print(f'{destination}: {len(train):,} training rows, {len(rows) - len(train):,} held out; dropped {dict(dropped)}')
    print('by system', dict(collections.Counter(r['system'] for r in train).most_common()))
    print('by family', dict(collections.Counter(r['family'] for r in train).most_common()))
    print('warnings', sum(WARNING in r['messages'][1]['content'] for r in train))
