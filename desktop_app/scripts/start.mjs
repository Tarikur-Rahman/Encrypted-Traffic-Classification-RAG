import {spawn} from 'node:child_process';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
process.chdir(fileURLToPath(new URL('..',import.meta.url)));
await import('./build-electron.mjs');
const dev=process.argv.includes('--dev');
let vite;
if(dev){const {createServer}=await import('vite');vite=await createServer({configFile:'frontend/vite.config.ts'});await vite.listen();}
const environment={...process.env,PG_DEV:dev?'1':'0'};delete environment.ELECTRON_RUN_AS_NODE;
const child=spawn(require('electron'),['.'],{stdio:'inherit',env:environment});
child.on('exit',async code=>{await vite?.close();process.exit(code??0)});
process.on('SIGINT',()=>child.kill());
