import {app,BrowserWindow,dialog,ipcMain,shell,session} from 'electron';
import {spawn,ChildProcess} from 'node:child_process';
import {randomBytes} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

let backend:ChildProcess;let win:BrowserWindow;let port=0;
const token=randomBytes(32).toString('hex');
const project=path.resolve(process.env.PG_PROJECT_ROOT || (app.isPackaged ? path.join(path.dirname(process.env.PORTABLE_EXECUTABLE_FILE || app.getPath('exe')),'../..') : path.join(__dirname,'../..')));
const appDirectory=path.join(project,'desktop_app');
const runtime=path.join(appDirectory,'runtime');
fs.mkdirSync(runtime,{recursive:true});
app.setPath('userData',path.join(runtime,'electron'));
app.setPath('sessionData',path.join(runtime,'electron-session'));
app.setPath('logs',path.join(runtime,'logs'));
app.setPath('crashDumps',path.join(runtime,'crashes'));
const dev=process.env.PG_DEV==='1';
const renderer=path.join(__dirname,'../frontend/dist/index.html');

function trusted(event:Electron.IpcMainInvokeEvent){
 if(event.sender!==win.webContents || event.senderFrame!==win.webContents.mainFrame)throw new Error('Untrusted desktop frame.');
 const url=event.senderFrame.url;
 if(dev ? !url.startsWith('http://127.0.0.1:5173/') : !url.startsWith('file://'))throw new Error('Untrusted desktop origin.');
}
async function request(route:string,method='GET',body?:unknown){
 if(!/^\/api\/(health|llm\/status|input\/(validate|inspect)|analysis\/create|analysis\/[A-F0-9]{12}\/(phase1|status|summary|comparison|flows|audit|export)(\/[^?#]*)?)(\?[^#]*)?$/.test(route)||!['GET','POST'].includes(method))throw new Error('Unsupported local operation.');
 const response=await fetch(`http://127.0.0.1:${port}${route}`,{method,headers:{authorization:`Bearer ${token}`,'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
 const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;
}
async function startBackend(){
 // The association bundle is frozen against this exact V5f interpreter. Do not
 // resolve `python` from PATH or fall back to the older GUI environment: its
 // Python/pandas identity is intentionally rejected by the fail-closed bundle
 // validator.
 const python=path.join(appDirectory,'runtime','v5f_exact_env','Scripts','python.exe');
 if(!fs.existsSync(python))throw new Error('Validated V5f Python environment is missing: '+python);
 const log=fs.openSync(path.join(runtime,'python-service.log'),'a');
 backend=spawn(python,['-B','-m','backend.main'],{cwd:appDirectory,windowsHide:true,env:{...process.env,PG_API_TOKEN:token,PG_PROJECT_ROOT:project,PYTHONDONTWRITEBYTECODE:'1',TEMP:runtime,TMP:runtime},stdio:['ignore','pipe',log]});
 await new Promise<void>((resolve,reject)=>{
  const timer=setTimeout(()=>reject(new Error('Python service did not start within 30 seconds. See runtime/python-service.log.')),30000);
  const lines=readline.createInterface({input:backend.stdout!});
  lines.on('line',line=>{try{const message=JSON.parse(line);if(message.ready&&Number.isInteger(message.port)){port=message.port;clearTimeout(timer);resolve();}}catch{}});
  backend.once('error',error=>{clearTimeout(timer);reject(error)});
  backend.once('exit',code=>{clearTimeout(timer);reject(new Error(`Python service stopped (${code}). See runtime/python-service.log.`))});
 });
 for(let i=0;i<100;i++){try{await request('/api/health');return;}catch{await new Promise(r=>setTimeout(r,100))}}
 throw new Error('Backend health check timed out.');
}
async function destination(format:string){
 if(!['json','csv'].includes(format))throw new Error('Unsupported export format.');
 const selected=await dialog.showSaveDialog(win,{title:'Export inside desktop_app',defaultPath:path.join(runtime,`ProvenanceGuard-${Date.now()}.${format}`),filters:[{name:format.toUpperCase(),extensions:[format]}]});
 if(selected.canceled||!selected.filePath)return null;
 const target=path.resolve(selected.filePath);
 if(fs.existsSync(target))throw new Error('Choose a new filename. Existing files are never overwritten.');
 // Resolve the real parent so junctions cannot escape the app output boundary.
 const parent=fs.realpathSync(path.dirname(target));
 const appRelative=path.relative(fs.realpathSync(appDirectory),parent);
 if(appRelative==='..'||appRelative.startsWith('..'+path.sep)||path.isAbsolute(appRelative))throw new Error('Exports must stay inside desktop_app.');
 return path.join(parent,path.basename(target));
}
app.whenReady().then(async()=>{
 await startBackend();
 win=new BrowserWindow({width:1440,height:900,minWidth:1080,minHeight:680,backgroundColor:'#0B1220',title:'ProvenanceGuard ETI',show:false,webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true,webSecurity:true}});
 win.removeMenu();
 session.defaultSession.setPermissionRequestHandler((_w,_p,callback)=>callback(false));
 win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
 win.webContents.on('will-navigate',event=>event.preventDefault());
 win.webContents.on('will-attach-webview',event=>event.preventDefault());
 const handle=(channel:string,fn:(...args:any[])=>unknown)=>ipcMain.handle(channel,(event,...args)=>{trusted(event);return fn(...args)});
 handle('api-request',request);
 handle('app-version',()=>app.getVersion());
 handle('select-file',async(kind?:string)=>{if(kind!==undefined&&!['csv','pcap'].includes(kind))throw new Error('Unsupported input type.');const value=await dialog.showOpenDialog(win,{title:'Select traffic or research CSV',properties:['openFile'],filters:[{name:'Traffic inputs',extensions:kind==='csv'?['csv']:kind==='pcap'?['pcap','pcapng']:['csv','pcap','pcapng']}]});if(value.canceled)return null;const file=value.filePaths[0];return {name:path.basename(file),path:file,extension:path.extname(file).toLowerCase(),size:fs.statSync(file).size};});
 handle('select-export',destination);
 handle('export-analysis',async(id:string,format:string)=>{const target=await destination(format);if(!target)return null;const data=await request(`/api/analysis/${id}/export`,'POST',{format});fs.writeFileSync(target,data.content,{encoding:'utf8',flag:'wx'});return target;});
 handle('open-document',async(value:string)=>{const url=new URL(value);if(url.protocol!=='https:'||url.username||url.password)throw new Error('Only HTTPS evidence documents can be opened.');await shell.openExternal(url.href)});
 if(dev)await win.loadURL('http://127.0.0.1:5173/');else await win.loadFile(renderer);
 win.show();
 backend.on('exit',()=>{if(!win.isDestroyed())win.webContents.send('backend-stopped')});
}).catch(error=>{dialog.showErrorBox('ProvenanceGuard could not start',String(error.message));app.quit()});
app.on('before-quit',()=>{backend?.kill()});
app.on('window-all-closed',()=>app.quit());
