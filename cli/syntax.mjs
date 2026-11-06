
import { spawn } from 'node:child_process';
import path from 'node:path';
export async function checkSyntax(filename, content, python = 'python3') {
  const ext = path.extname(filename).toLowerCase();
  let command, args;
  if (ext === '.py') {
    command = python; args = ['-I', '-c', 'import ast,sys; ast.parse(sys.stdin.read())'];
  } else if (['.js','.mjs','.cjs'].includes(ext)) {
    command = process.execPath; args = ['--check', '--input-type=' + (ext === '.cjs' ? 'commonjs' : 'module')];
  } else return { checked: false, message: 'No syntax checker for this file type; inspect the diff carefully.' };
  return new Promise((resolve,reject) => {
    const child = spawn(command,args,{stdio:['pipe','ignore','pipe']});
    let errors='',timedOut=false;
    const timer=setTimeout(()=>{timedOut=true;child.kill();},10000);
    child.stderr.on('data',part=>{if(errors.length<12000)errors+=part.toString();});
    child.stdin.on('error',()=>{});
    child.on('error',error=>{
      clearTimeout(timer);
      if(error.code==='ENOENT') resolve({checked:false,message:'Syntax checker unavailable; source was not executed.'});
      else reject(error);
    });
    child.on('close',code=>{
      clearTimeout(timer);
      if(timedOut) reject(new Error('Syntax check timed out. No proposal saved.'));
      else if(code===0) resolve({checked:true,message:'Syntax parse passed. Behavior is untested; source was not executed.'});
      else reject(new Error('Generated code failed syntax parsing. No proposal saved.\n'+errors.trim()));
    });
    child.stdin.end(content);
  });
}
